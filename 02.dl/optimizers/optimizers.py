"""
Gradient-Descent Optimizers (from scratch)
==========================================
Every deep net is trained by some variant of gradient descent. The differences
between them — momentum, per-parameter adaptive learning rates, bias correction —
are small algebraically but huge in practice. This module implements the classic
optimizers in pure NumPy (each `step()` is the literal update rule), watches them
minimize a 2D test function so you can SEE the trajectories, and cross-checks each
against its `torch.optim` equivalent on the same problem.

Variants implemented here:
    - SGD (vanilla gradient descent)
    - Momentum (heavy ball)
    - Nesterov accelerated gradient (lookahead momentum)
    - AdaGrad (accumulate squared grads)
    - RMSProp (exponential moving average of squared grads)
    - Adam (momentum + RMSProp + bias correction)
    - AdamW (Adam with *decoupled* weight decay)

Training techniques demonstrated:
    - Adaptive learning rates & bias correction (the Adam construction)
    - Decoupled weight decay (AdamW vs L2-in-the-gradient)
    - (LR scheduling/warmup is the transformer file; see training-techniques/README.md)

References:
    - Polyak (1964, momentum); Nesterov (1983); Duchi et al. (2011, AdaGrad)
    - Tieleman & Hinton (2012, RMSProp); Kingma & Ba (2014, Adam)
    - Loshchilov & Hutter (2017, AdamW / decoupled weight decay)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Test functions (objective + analytic gradient) used to race the optimizers
# ---------------------------------------------------------------------------
# An ILL-CONDITIONED quadratic bowl  f(x,y) = 0.5*(a*x^2 + b*y^2), min at (0,0).
# With a >> b the level sets are long ellipses: plain SGD zig-zags across the
# steep x-axis while crawling along the shallow y-axis. This is the textbook
# picture for why momentum and adaptive methods help. Condition number = a/b.
QUAD_A, QUAD_B = 20.0, 1.0


def quad(p):
    x, y = p
    return 0.5 * (QUAD_A * x ** 2 + QUAD_B * y ** 2)


def quad_grad(p):
    x, y = p
    return np.array([QUAD_A * x, QUAD_B * y])


def beale(p):
    """Beale function: a classic non-convex 2D test, min 0 at (3, 0.5)."""
    x, y = p
    return ((1.5 - x + x * y) ** 2
            + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


def beale_grad(p):
    """Analytic gradient of the Beale function (chain rule on each term)."""
    x, y = p
    a = 1.5 - x + x * y
    b = 2.25 - x + x * y ** 2
    c = 2.625 - x + x * y ** 3
    dx = 2 * a * (y - 1) + 2 * b * (y ** 2 - 1) + 2 * c * (y ** 3 - 1)
    dy = 2 * a * x + 2 * b * (2 * x * y) + 2 * c * (3 * x * y ** 2)
    return np.array([dx, dy])


# ---------------------------------------------------------------------------
# 1. NumPy optimizers (from scratch) — each `step` IS the update rule
# ---------------------------------------------------------------------------
class Optimizer:
    """Base: holds params (a single flat vector here) and applies an update."""

    name = "optimizer"

    def __init__(self, params: np.ndarray, lr: float):
        self.p = np.array(params, dtype=float)   # parameter vector theta
        self.lr = lr
        self.t = 0                               # timestep (for bias correction)

    def step(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SGD(Optimizer):
    r"""Vanilla gradient descent:  theta <- theta - lr * g."""
    name = "sgd"

    def step(self, g):
        self.p -= self.lr * g
        return self.p


class Momentum(Optimizer):
    r"""Heavy ball: accumulate a velocity (EMA of gradients).
        v <- mu*v + g ;  theta <- theta - lr*v.
    The velocity averages out oscillation across steep directions and builds up
    speed along consistent (low-curvature) directions."""
    name = "momentum"

    def __init__(self, params, lr, mu=0.9):
        super().__init__(params, lr)
        self.mu = mu
        self.v = np.zeros_like(self.p)

    def step(self, g):
        self.v = self.mu * self.v + g
        self.p -= self.lr * self.v
        return self.p


class Nesterov(Optimizer):
    r"""Nesterov accelerated gradient: evaluate the gradient at the *lookahead*
    point theta - lr*mu*v (where momentum is about to carry us), giving a
    correction term. Equivalent reformulation used here:
        v       <- mu*v + g
        update  <- mu*v + g            (= mu*v_new + g, the lookahead-corrected step)
        theta   <- theta - lr*update.
    Note: the gradient g is supplied at the lookahead point by the caller."""
    name = "nesterov"

    def __init__(self, params, lr, mu=0.9):
        super().__init__(params, lr)
        self.mu = mu
        self.v = np.zeros_like(self.p)

    def lookahead(self):
        """Point at which the gradient should be evaluated for this step."""
        return self.p - self.lr * self.mu * self.v

    def step(self, g):
        # g is grad at the lookahead point self.lookahead()
        self.v = self.mu * self.v + g
        self.p -= self.lr * self.v
        return self.p


class AdaGrad(Optimizer):
    r"""Per-parameter learning rate that shrinks with accumulated gradient energy:
        G <- G + g^2 ;  theta <- theta - lr * g / (sqrt(G)+eps).
    Great for sparse features, but G only grows -> the effective LR decays to 0."""
    name = "adagrad"

    def __init__(self, params, lr, eps=1e-8):
        super().__init__(params, lr)
        self.eps = eps
        self.G = np.zeros_like(self.p)

    def step(self, g):
        self.G += g ** 2
        self.p -= self.lr * g / (np.sqrt(self.G) + self.eps)
        return self.p


class RMSProp(Optimizer):
    r"""Fix AdaGrad's monotonic decay by using an EMA of squared gradients:
        E <- rho*E + (1-rho)*g^2 ;  theta <- theta - lr * g / (sqrt(E)+eps).
    The window 'forgets' old gradients so the effective LR stays alive."""
    name = "rmsprop"

    def __init__(self, params, lr, rho=0.9, eps=1e-8):
        super().__init__(params, lr)
        self.rho, self.eps = rho, eps
        self.E = np.zeros_like(self.p)

    def step(self, g):
        self.E = self.rho * self.E + (1 - self.rho) * g ** 2
        self.p -= self.lr * g / (np.sqrt(self.E) + self.eps)
        return self.p


class Adam(Optimizer):
    r"""Adam = Momentum (1st moment m) + RMSProp (2nd moment v) + bias correction.
        m <- b1*m + (1-b1)*g                 (EMA of gradient)
        v <- b2*v + (1-b2)*g^2               (EMA of squared gradient)
        m_hat = m/(1-b1^t),  v_hat = v/(1-b2^t)   (correct the zero-init bias)
        theta <- theta - lr * m_hat / (sqrt(v_hat)+eps).
    Bias correction matters because m,v start at 0 and are badly under-estimated
    for small t; dividing by (1-b^t) -> 1 as t grows removes that startup bias."""
    name = "adam"

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = np.zeros_like(self.p)
        self.v = np.zeros_like(self.p)

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        self.p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return self.p


class AdamW(Adam):
    r"""AdamW DECOUPLES weight decay from the gradient. Classic L2 adds wd*theta
    to the gradient (so it gets scaled by 1/sqrt(v) like everything else). AdamW
    instead applies the decay directly to the weights:
        theta <- theta - lr*( m_hat/(sqrt(v_hat)+eps) + wd*theta ).
    This makes the regularization strength independent of the adaptive scaling —
    the fix that made Adam competitive with SGD on generalization."""
    name = "adamw"

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        super().__init__(params, lr, b1, b2, eps)
        self.wd = wd

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        # decoupled decay: applied to theta, NOT routed through the adaptive term
        self.p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * self.p)
        return self.p


def optimize(opt_cls, start, grad_fn, steps=400, **kw):
    """Run an optimizer on grad_fn from `start`, returning the trajectory.

    Nesterov is special: it needs the gradient evaluated at a lookahead point.
    """
    opt = opt_cls(np.array(start, dtype=float), **kw)
    traj = [opt.p.copy()]
    for _ in range(steps):
        if isinstance(opt, Nesterov):
            g = grad_fn(opt.lookahead())
        else:
            g = grad_fn(opt.p)
        opt.step(g)
        traj.append(opt.p.copy())
    return np.array(traj)


# Hyper-parameters tuned for the ill-conditioned quadratic bowl (all converge,
# so you can fairly compare *how* they get there). torch_optimize mirrors these.
NUMPY_OPTS = {
    "sgd": (SGD, dict(lr=0.05)),
    "momentum": (Momentum, dict(lr=0.02, mu=0.9)),
    "nesterov": (Nesterov, dict(lr=0.02, mu=0.9)),
    "adagrad": (AdaGrad, dict(lr=0.5)),
    "rmsprop": (RMSProp, dict(lr=0.1, rho=0.9)),
    "adam": (Adam, dict(lr=0.1, b1=0.9, b2=0.999)),
    "adamw": (AdamW, dict(lr=0.1, wd=0.0)),
}


# ---------------------------------------------------------------------------
# 2. PyTorch equivalents — same update rules via torch.optim
# ---------------------------------------------------------------------------
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_optimize(name, start, steps=400):
    """Minimize the (autograd-differentiated) quadratic bowl with torch.optim,
    returning the trajectory. Mirrors NUMPY_OPTS hyper-parameters exactly, so the
    NumPy update rules can be validated against the framework."""
    dev = get_device()
    theta = torch.tensor(start, dtype=torch.float64, device=dev, requires_grad=True)
    builders = {
        "sgd": lambda: torch.optim.SGD([theta], lr=0.05),
        "momentum": lambda: torch.optim.SGD([theta], lr=0.02, momentum=0.9),
        "nesterov": lambda: torch.optim.SGD([theta], lr=0.02, momentum=0.9, nesterov=True),
        "adagrad": lambda: torch.optim.Adagrad([theta], lr=0.5),
        "rmsprop": lambda: torch.optim.RMSprop([theta], lr=0.1, alpha=0.9),
        "adam": lambda: torch.optim.Adam([theta], lr=0.1, betas=(0.9, 0.999)),
        "adamw": lambda: torch.optim.AdamW([theta], lr=0.1, weight_decay=0.0),
    }
    opt = builders[name]()
    traj = [theta.detach().cpu().numpy().copy()]
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * (QUAD_A * theta[0] ** 2 + QUAD_B * theta[1] ** 2)
        loss.backward()
        opt.step()
        traj.append(theta.detach().cpu().numpy().copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# 3. Demo — race the optimizers on Beale; compare NumPy vs torch.optim
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    start = (2.0, 2.0)
    target = np.array([0.0, 0.0])           # min of the quadratic bowl

    print(f"Ill-conditioned quadratic (cond = {QUAD_A/QUAD_B:.0f}) from (2, 2); "
          "min at (0, 0).")
    print(f"{'optimizer':10s} {'steps':>6s} {'f(final)':>12s} {'dist-to-min':>12s}")
    trajs = {}
    for nm, (cls, kw) in NUMPY_OPTS.items():
        traj = optimize(cls, start, quad_grad, steps=300, **kw)
        trajs[nm] = traj
        final = traj[-1]
        print(f"{nm:10s} {len(traj)-1:6d} {quad(final):12.4e} "
              f"{np.linalg.norm(final - target):12.4e}")

    # Bias-correction illustration. m,v are EMAs initialised at 0, so for small t
    # they are biased toward 0 -- and crucially v is biased MORE than m (b2 > b1),
    # so the ratio m/sqrt(v) is mis-scaled early. Feed a CONSTANT gradient g, whose
    # true moments are mean(g)=g and mean(g^2)=g^2: a correct estimator must give an
    # update of size exactly lr at every step. Watch the corrected version do that.
    print("\nAdam bias correction (constant gradient; ideal |update| = lr = 0.1):")
    print(f"  {'t':>3s} {'|update| uncorrected':>22s} {'|update| corrected':>20s}")
    b1, b2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    m = np.zeros(2); v = np.zeros(2)
    g = np.array([1.0, -1.0])               # constant unit gradient
    for t in range(1, 6):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        raw = lr * m / (np.sqrt(v) + eps)               # NO correction
        cor = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)  # corrected
        per = np.abs(raw[0]); pec = np.abs(cor[0])      # symmetric coords
        print(f"  {t:3d} {per:22.4f} {pec:20.4f}")
    print("  -> uncorrected step is MIS-SCALED early (v under-estimated more than m,")
    print("     so m/sqrt(v) is off and only drifts toward lr slowly); the 1/(1-b^t)")
    print("     factors give the correct ~lr step from the very first iteration.")

    # cross-check NumPy vs torch.optim on the same problem. SGD/Momentum/Adam/AdamW
    # use identical formulas so they match to machine precision; torch's Nesterov &
    # the Adagrad/RMSprop eps-placement differ slightly, so we just report those.
    print("\nNumPy vs torch.optim final point (max coord diff over 300 steps):")
    for nm in NUMPY_OPTS:
        cls, kw = NUMPY_OPTS[nm]
        np_traj = optimize(cls, start, quad_grad, steps=300, **kw)
        t_traj = torch_optimize(nm, start, steps=300)
        diff = np.max(np.abs(np_traj[-1] - t_traj[-1]))
        print(f"  {nm:9s}: {diff:.2e}")
    print("  -> SGD/Momentum/Adam/AdamW match exactly (same update rule);")
    print("     small diffs for Nesterov/Adagrad/RMSProp = framework formulation details.")


if __name__ == "__main__":
    demo()

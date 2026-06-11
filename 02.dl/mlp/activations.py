"""
Activation Functions (forward + derivative, from scratch)
=========================================================
A neural network is just an affine map composed with a pointwise nonlinearity.
The nonlinearity is what lets stacked layers represent non-linear functions, and
its *derivative* is what backprop multiplies through at every layer. This module
implements the classic activations in pure NumPy — each with an explicit forward
AND derivative — and mirrors them with their torch equivalents, then measures the
single property that decides whether deep nets train: how much the activation
*saturates* (drives its derivative toward 0), which is the root of the
vanishing-gradient problem.

Variants implemented here:
    - Sigmoid (logistic)
    - Tanh
    - ReLU
    - LeakyReLU (slope alpha for x<0)
    - ELU (exponential linear unit)
    - GELU (Gaussian error linear unit; tanh approximation)
    - Swish / SiLU (x * sigmoid(beta*x))
    - Softmax (vector-valued; full Jacobian)

Training techniques demonstrated:
    - VANISHING GRADIENTS link: saturating activations (sigmoid/tanh) have
      derivatives bounded well below 1, so the product of Jacobians shrinks
      geometrically with depth. Non-saturating activations (ReLU family, GELU,
      Swish) keep the derivative near 1 over the active region. We MEASURE the
      max derivative and the fraction of "dead"/saturated inputs.
      (see training-techniques/README.md -> Vanishing / exploding gradients)

References:
    - Glorot & Bengio (2010); Nair & Hinton (2010, ReLU); Clevert et al. (2015, ELU)
    - Hendrycks & Gimpel (2016, GELU); Ramachandran et al. (2017, Swish/SiLU)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch) — each activation: forward + derivative
# ---------------------------------------------------------------------------
class Activation:
    """Base class: subclasses define forward `f(x)` and elementwise `df(x)`.

    Convention: `df` returns df/dx evaluated at the *input* x (not at f(x)),
    which is what the backward pass multiplies: delta_in = delta_out * df(x).
    """

    name = "activation"

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def df(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # convenience aliases
    def __call__(self, x): return self.forward(x)


class Sigmoid(Activation):
    r"""sigma(x) = 1/(1+e^{-x});  sigma'(x) = sigma(x)(1-sigma(x)) in [0, 1/4]."""
    name = "sigmoid"

    def forward(self, x):
        # clip to avoid overflow in exp; output in (0,1)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def df(self, x):
        s = self.forward(x)
        return s * (1.0 - s)               # max 0.25 at x=0 -> SATURATES


class Tanh(Activation):
    r"""tanh(x);  tanh'(x) = 1 - tanh(x)^2 in [0, 1]."""
    name = "tanh"

    def forward(self, x):
        return np.tanh(x)

    def df(self, x):
        return 1.0 - np.tanh(x) ** 2       # max 1 at x=0 -> still saturates far out


class ReLU(Activation):
    r"""ReLU(x) = max(0,x);  ReLU'(x) = 1 if x>0 else 0 (undefined at 0 -> 0)."""
    name = "relu"

    def forward(self, x):
        return np.maximum(0.0, x)

    def df(self, x):
        return (x > 0).astype(x.dtype)     # 0 or 1 -> no shrinkage on active side


class LeakyReLU(Activation):
    r"""LeakyReLU(x) = x if x>0 else alpha*x;  derivative 1 or alpha."""
    name = "leaky_relu"

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def df(self, x):
        return np.where(x > 0, 1.0, self.alpha)   # never exactly 0 -> no "dead" units


class ELU(Activation):
    r"""ELU(x) = x if x>0 else alpha*(e^x - 1).
    Derivative: 1 if x>0 else alpha*e^x = ELU(x)+alpha on the negative side."""
    name = "elu"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -50, 50)) - 1.0))

    def df(self, x):
        return np.where(x > 0, 1.0, self.alpha * np.exp(np.clip(x, -50, 50)))


class GELU(Activation):
    r"""GELU(x) = x * Phi(x), Phi = standard normal CDF.
    We use the tanh approximation (as in the original paper / BERT/GPT):
        GELU(x) ~= 0.5 x (1 + tanh[ sqrt(2/pi) (x + 0.044715 x^3) ]).
    Derivative obtained by the product + chain rule on that closed form."""
    name = "gelu"
    _c = np.sqrt(2.0 / np.pi)              # sqrt(2/pi)
    _a = 0.044715

    def _inner(self, x):
        return self._c * (x + self._a * x ** 3)

    def forward(self, x):
        return 0.5 * x * (1.0 + np.tanh(self._inner(x)))

    def df(self, x):
        u = self._inner(x)
        t = np.tanh(u)
        # du/dx = c (1 + 3 a x^2)
        du = self._c * (1.0 + 3.0 * self._a * x ** 2)
        sech2 = 1.0 - t ** 2               # d/du tanh(u)
        # product rule on 0.5 x (1 + tanh u)
        return 0.5 * (1.0 + t) + 0.5 * x * sech2 * du


class Swish(Activation):
    r"""Swish / SiLU(x) = x * sigma(beta*x).  With beta=1 this is SiLU.
    Derivative: sigma(bx) + x * beta * sigma(bx)(1 - sigma(bx))
              = sigma(bx) + b*x*sigma(bx) - b*x*sigma(bx)^2
              = beta*Swish(x) + sigma(bx)(1 - beta*Swish(x))."""
    name = "swish"

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def _sig(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(self.beta * x, -50, 50)))

    def forward(self, x):
        return x * self._sig(x)

    def df(self, x):
        s = self._sig(x)
        return s + self.beta * x * s * (1.0 - s)


class Softmax:
    r"""Vector-valued activation over the last axis.
        softmax(z)_i = e^{z_i} / sum_j e^{z_j}  (shift by max for stability).
    Its Jacobian for a single sample is
        J_ij = s_i (delta_ij - s_j),
    i.e. diag(s) - s s^T. We expose both the forward and the per-sample Jacobian.
    """
    name = "softmax"

    def forward(self, z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def __call__(self, z): return self.forward(z)

    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Batched Jacobian, shape (N, K, K) for input (N, K)."""
        s = self.forward(z)                          # (N, K)
        N, K = s.shape
        diag = np.einsum("nk,kj->nkj", s, np.eye(K))  # diag(s)
        outer = np.einsum("ni,nj->nij", s, s)         # s s^T
        return diag - outer

    def vjp(self, z: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Vector-Jacobian product g @ J without forming J: efficient backward.
        dL/dz = s * (g - sum_j g_j s_j)."""
        s = self.forward(z)
        return s * (g - (g * s).sum(axis=-1, keepdims=True))


# registry of the elementwise activations (instantiated with defaults)
ELEMENTWISE = {
    "sigmoid": Sigmoid(),
    "tanh": Tanh(),
    "relu": ReLU(),
    "leaky_relu": LeakyReLU(0.1),
    "elu": ELU(1.0),
    "gelu": GELU(),
    "swish": Swish(1.0),
}


def check_gradient(act: Activation, x: np.ndarray, eps: float = 1e-5) -> float:
    """Max abs error between analytic df and a central finite difference.
    Numerically verifies every hand-derived derivative above."""
    num = (act.forward(x + eps) - act.forward(x - eps)) / (2 * eps)
    return float(np.max(np.abs(num - act.df(x))))


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (idiomatic)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


TORCH_ACT = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.1),
    "elu": lambda x: F.elu(x, alpha=1.0),
    "gelu": lambda x: F.gelu(x, approximate="tanh"),
    "swish": F.silu,                       # SiLU == Swish(beta=1)
    "softmax": lambda x: F.softmax(x, dim=-1),
}


def torch_forward_and_grad(name: str, x_np: np.ndarray):
    """Run the torch activation and obtain its derivative via autograd.
    Returns (forward, derivative) as numpy arrays, on the chosen device."""
    dev = get_device()
    x = torch.tensor(x_np, dtype=torch.float64, device=dev, requires_grad=True)
    y = TORCH_ACT[name](x)
    # sum() so we get dy_i/dx_i on the diagonal for elementwise functions
    y.sum().backward()
    return y.detach().cpu().numpy(), x.grad.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo — verify derivatives & MEASURE saturation (the vanishing-grad link)
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    x = np.linspace(-6, 6, 2001)
    # grid that avoids the kink at x=0 (where ReLU/LeakyReLU are non-differentiable
    # and a central finite difference straddles the corner)
    x_smooth = np.linspace(-6, 6, 2000) + 1e-3

    # (a) verify every analytic derivative against finite differences
    print("Analytic derivative vs finite-difference (max abs error):")
    for nm, act in ELEMENTWISE.items():
        err = check_gradient(act, x_smooth)
        print(f"  {nm:11s}: {err:.2e}")

    # (b) cross-check NumPy forward & derivative against PyTorch autograd
    print("\nNumPy vs PyTorch (max abs error on f and f'):")
    for nm, act in ELEMENTWISE.items():
        tf, tg = torch_forward_and_grad(nm, x)
        ef = float(np.max(np.abs(act.forward(x) - tf)))
        eg = float(np.max(np.abs(act.df(x) - tg)))
        print(f"  {nm:11s}: |df_f|={ef:.2e}  |df_grad|={eg:.2e}")

    # softmax forward agreement
    z = np.random.randn(5, 4)
    sm = Softmax()
    tsm, _ = torch_forward_and_grad("softmax", z)
    print(f"  softmax    : |df_f|={np.max(np.abs(sm.forward(z) - tsm)):.2e}")
    # verify the efficient VJP against the explicit Jacobian
    g = np.random.randn(*z.shape)
    vjp_fast = sm.vjp(z, g)
    vjp_full = np.einsum("ni,nij->nj", g, sm.jacobian(z))
    print(f"  softmax vjp vs Jacobian: {np.max(np.abs(vjp_fast - vjp_full)):.2e}")

    # (c) MEASURE saturation — the vanishing-gradient link
    print("\nSaturation report (root cause of vanishing gradients):")
    print("  activation   max f'    mean f' over [-6,6]   %|f'|<0.01 (saturated/dead)")
    for nm, act in ELEMENTWISE.items():
        d = act.df(x)
        dead = 100.0 * np.mean(np.abs(d) < 0.01)
        print(f"  {nm:11s} {d.max():7.3f}   {d.mean():10.3f}            {dead:6.1f}%")
    print("  -> sigmoid/tanh: max f' <= 1 and most inputs saturate, so a product")
    print("     of L such factors shrinks geometrically => VANISHING gradients.")
    print("     ReLU family/GELU/Swish keep f' ~ 1 on the active region => deep")
    print("     nets stay trainable (see training-techniques/README.md).")


if __name__ == "__main__":
    demo()

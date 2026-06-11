"""
Generalized Linear Models (GLMs)
================================
Ordinary linear regression assumes Gaussian, constant-variance noise. GLMs lift
that restriction: the response follows an *exponential-family* distribution whose
mean is connected to a linear predictor $\eta=Xw$ through an invertible **link
function** $g$, i.e. $g(\mu)=\eta$. Picking the family + link recovers familiar
models (Gaussian+identity = OLS, Bernoulli+logit = logistic regression) and gives
us new ones for counts (**Poisson**, log link) and positive skewed data
(**Gamma**, log link). We fit by both Iteratively Reweighted Least Squares (IRLS,
i.e. Fisher scoring / Newton's method) and plain gradient descent, in NumPy, then
mirror the gradient-descent fit in PyTorch with autograd.

Variants implemented here:
    - Poisson regression (log link) — counts / rates
    - Gamma regression (log link)   — positive continuous, constant CV
    - Exponential-family / link-function framework (general IRLS + GD)

Training techniques demonstrated:
    - IRLS / Fisher scoring (Newton's method with the Fisher information)
    - Gradient descent on the negative log-likelihood (compare with IRLS)

References:
    - Nelder & Wedderburn (1972), "Generalized Linear Models"
    - McCullagh & Nelder (1989), "Generalized Linear Models" (2nd ed.)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
# A GLM with the natural/canonical setup we use here (log link) needs, per family:
#   - inv_link(eta) = mu                          (mean response)
#   - the gradient of the NEGATIVE log-likelihood wrt w
#   - the IRLS "working weights" w_i and "working response" z_i
#
# For an exponential-family GLM the score (gradient of the LOG-likelihood) is
#       dL/dw = X^T (y - mu) * (dmu/deta) / Var(mu) ... [general]
# With the LOG link mu = exp(eta) and the natural variance functions below, this
# simplifies. We write each family explicitly so the math is visible.
class GLMNumPy:
    r"""
    Generalized Linear Model with a log link, fit by IRLS or gradient descent.

    Linear predictor:  eta = X w           (X includes a bias column of ones)
    Mean:              mu  = g^{-1}(eta) = exp(eta)         (log link)

    POISSON  y ~ Poisson(mu):
        log-lik (drop const):  L = sum_i [ y_i eta_i - exp(eta_i) ]
        gradient (of -L):      -X^T (y - mu)
        IRLS weights:          W = mu      (variance fn V(mu)=mu, log link)

    GAMMA  y ~ Gamma(mean=mu, shape=nu), log link:
        log-lik (drop const):  L = sum_i [ -y_i/mu_i - eta_i ]  (times nu)
        gradient (of -L):      -X^T (y - mu)/mu
        IRLS weights:          W = 1       (variance fn V(mu)=mu^2, log link)
    """

    def __init__(self, family="poisson", fit_method="irls", lr=0.01,
                 n_iters=100, l2=0.0, seed=SEED):
        assert family in ("poisson", "gamma")
        assert fit_method in ("irls", "gd")
        self.family = family
        self.fit_method = fit_method
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2                       # ridge penalty on weights (not bias)
        self.seed = seed
        self.w = None
        self.history = []                  # negative log-likelihood per iter

    # --- design matrix: prepend a column of ones for the intercept ----------
    @staticmethod
    def _design(X):
        X = np.asarray(X, float)
        return np.hstack([np.ones((len(X), 1)), X])

    # --- inverse link: log link => mu = exp(eta) ----------------------------
    def _inv_link(self, eta):
        return np.exp(np.clip(eta, -30, 30))

    # --- per-family negative log-likelihood (for monitoring) ----------------
    def _nll(self, eta, y):
        mu = self._inv_link(eta)
        if self.family == "poisson":
            # -[y*eta - mu]  (drop log(y!) constant, independent of w)
            return np.mean(mu - y * eta)
        else:  # gamma, unit shape -> deviance-like NLL: y/mu + eta
            return np.mean(y / mu + eta)

    # --- gradient of the NEGATIVE log-likelihood wrt w ----------------------
    def _grad(self, Xb, eta, y):
        mu = self._inv_link(eta)
        if self.family == "poisson":
            # dL/dw = X^T (y - mu);  grad of -L = -X^T (y - mu)
            g = -Xb.T @ (y - mu) / len(y)
        else:  # gamma with log link: dL/dw = X^T (y - mu)/mu
            g = -Xb.T @ ((y - mu) / mu) / len(y)
        if self.l2:
            reg = self.l2 * self.w
            reg[0] = 0.0                   # don't penalize the intercept
            g = g + reg
        return g

    # --- IRLS: Fisher scoring solves a weighted least squares each step -----
    def _irls(self, Xb, y):
        n, d = Xb.shape
        w = np.zeros(d)
        # warm start the intercept at log(mean(y)) so exp(eta0)=mean(y)
        w[0] = np.log(max(y.mean(), 1e-3))
        for _ in range(self.n_iters):
            eta = Xb @ w
            mu = self._inv_link(eta)
            # Working weights W and working response z (Fisher scoring):
            #   z = eta + (y - mu) * (deta/dmu)
            # log link => deta/dmu = 1/mu.
            if self.family == "poisson":
                W = mu                       # V(mu)=mu
                z = eta + (y - mu) / mu
            else:  # gamma: V(mu)=mu^2, W = (dmu/deta)^2 / V = mu^2/mu^2 = 1
                W = np.ones_like(mu)
                z = eta + (y - mu) / mu
            # Solve weighted normal equations: (X^T W X + l2 I) w = X^T W z
            WX = Xb * W[:, None]
            A = Xb.T @ WX
            if self.l2:
                R = self.l2 * n * np.eye(d); R[0, 0] = 0.0
                A = A + R
            b = Xb.T @ (W * z)
            w_new = np.linalg.solve(A, b)
            self.history.append(self._nll(Xb @ w_new, y))
            if np.linalg.norm(w_new - w) < 1e-8:
                w = w_new
                break
            w = w_new
        return w

    # --- plain gradient descent on the NLL ----------------------------------
    def _gd(self, Xb, y):
        rng = np.random.default_rng(self.seed)
        w = np.zeros(Xb.shape[1])
        w[0] = np.log(max(y.mean(), 1e-3))
        self.w = w
        for _ in range(self.n_iters):
            eta = Xb @ w
            self.history.append(self._nll(eta, y))
            g = self._grad(Xb, eta, y)
            w = w - self.lr * g
            self.w = w
        return w

    def fit(self, X, y):
        y = np.asarray(y, float)
        Xb = self._design(X)
        self.w = np.zeros(Xb.shape[1])
        self.history = []
        self.w = self._irls(Xb, y) if self.fit_method == "irls" else self._gd(Xb, y)
        return self

    def predict(self, X):
        """Return the predicted mean response mu = exp(Xw)."""
        return self._inv_link(self._design(X) @ self.w)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (idiomatic — autograd on the NLL)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GLMTorch(nn.Module):
    """
    GLM (log link) trained by minimizing the negative log-likelihood with autograd.

    The forward pass produces the linear predictor eta = Xw + b; the loss is the
    per-family NLL. Autograd then reproduces exactly the hand-derived gradients.
    """

    def __init__(self, n_features, family="poisson"):
        super().__init__()
        assert family in ("poisson", "gamma")
        self.family = family
        self.linear = nn.Linear(n_features, 1)

    def forward(self, X):
        return self.linear(X).squeeze(-1)          # eta

    def nll(self, eta, y):
        mu = torch.exp(torch.clamp(eta, -30, 30))
        if self.family == "poisson":
            return (mu - y * eta).mean()           # drop log(y!) const
        return (y / mu + eta).mean()               # gamma, unit shape

    def fit(self, X, y, lr=0.05, n_iters=300, l2=0.0):
        dev = get_device(); self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(y, dtype=torch.float32, device=dev)
        # warm start intercept at log(mean(y))
        with torch.no_grad():
            self.linear.bias.fill_(float(np.log(max(y.mean().item(), 1e-3))))
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        self.history = []
        for _ in range(n_iters):
            opt.zero_grad()
            loss = self.nll(self(X), y)
            loss.backward(); opt.step()
            self.history.append(loss.item())
        return self

    @torch.no_grad()
    def predict(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return torch.exp(self(X)).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def _make_poisson(rng, n=400, d=3):
    """Synthetic counts: y ~ Poisson(exp(Xw + b))."""
    X = rng.normal(size=(n, d)) * 0.5
    w_true = np.array([0.6, -0.4, 0.3])
    eta = 0.5 + X @ w_true
    y = rng.poisson(np.exp(eta)).astype(float)
    return X, y, np.r_[0.5, w_true]


def _make_gamma(rng, n=400, d=3, shape=5.0):
    """Synthetic positive responses: y ~ Gamma(mean=exp(Xw+b))."""
    X = rng.normal(size=(n, d)) * 0.5
    w_true = np.array([0.5, 0.2, -0.3])
    mu = np.exp(0.2 + X @ w_true)
    # numpy gamma is parameterized by (shape k, scale theta), mean = k*theta
    y = rng.gamma(shape=shape, scale=mu / shape)
    return X, y, np.r_[0.2, w_true]


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("=== Poisson regression (log link) ===")
    Xp, yp, w_true_p = _make_poisson(rng)
    irls = GLMNumPy(family="poisson", fit_method="irls", n_iters=50).fit(Xp, yp)
    gd = GLMNumPy(family="poisson", fit_method="gd", lr=0.2, n_iters=400).fit(Xp, yp)
    tt = GLMTorch(Xp.shape[1], family="poisson").fit(Xp, yp, lr=0.05, n_iters=600)
    print(f"  true coefs (b,w):  {np.round(w_true_p, 3)}")
    print(f"  IRLS coefs:        {np.round(irls.w, 3)}  (final NLL {irls.history[-1]:.4f})")
    print(f"  GD   coefs:        {np.round(gd.w, 3)}  (final NLL {gd.history[-1]:.4f})")
    tw = np.r_[tt.linear.bias.item(), tt.linear.weight.detach().cpu().numpy().ravel()]
    print(f"  Torch coefs:       {np.round(tw, 3)}")
    print(f"  IRLS converged in {len(irls.history)} iters (Newton is fast)")

    print("\n=== Gamma regression (log link) ===")
    Xg, yg, w_true_g = _make_gamma(rng)
    irls_g = GLMNumPy(family="gamma", fit_method="irls", n_iters=50).fit(Xg, yg)
    gd_g = GLMNumPy(family="gamma", fit_method="gd", lr=0.2, n_iters=400).fit(Xg, yg)
    tt_g = GLMTorch(Xg.shape[1], family="gamma").fit(Xg, yg, lr=0.05, n_iters=600)
    print(f"  true coefs (b,w):  {np.round(w_true_g, 3)}")
    print(f"  IRLS coefs:        {np.round(irls_g.w, 3)}")
    print(f"  GD   coefs:        {np.round(gd_g.w, 3)}")
    twg = np.r_[tt_g.linear.bias.item(), tt_g.linear.weight.detach().cpu().numpy().ravel()]
    print(f"  Torch coefs:       {np.round(twg, 3)}")

    # quality: mean absolute relative error on the mean
    mu_hat = irls.predict(Xp)
    print(f"\n  Poisson IRLS mean |y-mu|: {np.mean(np.abs(yp - mu_hat)):.3f}")


if __name__ == "__main__":
    demo()

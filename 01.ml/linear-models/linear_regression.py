"""
Linear Regression
=================
Fit a linear map y ≈ Xw + b by minimizing squared error. The workhorse of
classical ML and the cleanest place to see "define a loss, take its gradient,
descend" — the pattern every later model reuses.

Variants implemented here:
    - OLS (ordinary least squares)        — closed form + gradient descent
    - Ridge (L2 penalty)                  — shrinks weights, fixes ill-conditioning
    - Lasso (L1 penalty)                  — sparse weights (subgradient / soft-threshold)
    - ElasticNet (L1 + L2)
    - Polynomial features                 — linear model, nonlinear fit

Training techniques demonstrated:
    - Feature standardization (so gradient descent / penalties behave)
    - Gradient descent vs. the normal equations (closed form)

References:
    - Hastie, Tibshirani, Friedman, "The Elements of Statistical Learning", ch. 3
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation — the math made explicit
# ---------------------------------------------------------------------------
class LinearRegressionNumPy:
    r"""
    Model:      ŷ = X w + b
    Loss (MSE): L(w,b) = (1/2n) * ||Xw + b - y||^2  +  reg(w)

    Penalties:
        ridge:       (λ/2) ||w||_2^2     -> grad  λ w
        lasso:       λ ||w||_1           -> subgrad λ sign(w)
        elasticnet:  λ[α||w||_1 + (1-α)/2 ||w||_2^2]

    Gradient of the MSE term:
        dL/dw = (1/n) X^T (Xw + b - y)
        dL/db = (1/n) sum(Xw + b - y)
    """

    def __init__(self, penalty=None, lam=0.0, l1_ratio=0.5,
                 lr=0.1, n_iters=2000, fit_closed_form=False):
        self.penalty = penalty            # None | "ridge" | "lasso" | "elasticnet"
        self.lam = lam
        self.l1_ratio = l1_ratio          # only for elasticnet
        self.lr = lr
        self.n_iters = n_iters
        self.fit_closed_form = fit_closed_form
        self.w = None
        self.b = 0.0
        self.history = []

    # --- closed form (normal equations) -- only OLS / Ridge have one ---------
    def _fit_closed_form(self, X, y):
        n, d = X.shape
        # Augment with a bias column of ones: X_aug = [X | 1]
        X_aug = np.hstack([X, np.ones((n, 1))])
        A = X_aug.T @ X_aug
        if self.penalty == "ridge":
            # Don't regularize the bias term -> zero its diagonal entry.
            R = self.lam * np.eye(d + 1)
            R[-1, -1] = 0.0
            A = A + R
        # Solve A theta = X^T y   (theta = [w; b])
        theta = np.linalg.solve(A, X_aug.T @ y)
        self.w, self.b = theta[:-1], theta[-1]
        return self

    # --- penalty gradient ----------------------------------------------------
    def _reg_grad(self, w):
        if self.penalty == "ridge":
            return self.lam * w
        if self.penalty == "lasso":
            return self.lam * np.sign(w)            # subgradient
        if self.penalty == "elasticnet":
            a = self.l1_ratio
            return self.lam * (a * np.sign(w) + (1 - a) * w)
        return 0.0

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        if self.fit_closed_form and self.penalty in (None, "ridge"):
            return self._fit_closed_form(X, y)

        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iters):
            pred = X @ self.w + self.b               # forward
            err = pred - y                           # residual
            grad_w = (X.T @ err) / n + self._reg_grad(self.w)
            grad_b = err.mean()                      # bias unregularized
            self.w -= self.lr * grad_w               # gradient descent step
            self.b -= self.lr * grad_b
            self.history.append(0.5 * np.mean(err ** 2))
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.w + self.b


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — idiomatic, autograd does the calculus
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LinearRegressionTorch(nn.Module):
    def __init__(self, in_features, penalty=None, lam=0.0, l1_ratio=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)   # holds w and b
        self.penalty, self.lam, self.l1_ratio = penalty, lam, l1_ratio

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def _penalty(self):
        w = self.linear.weight
        if self.penalty == "ridge":
            return self.lam * (w ** 2).sum()
        if self.penalty == "lasso":
            return self.lam * w.abs().sum()
        if self.penalty == "elasticnet":
            a = self.l1_ratio
            return self.lam * (a * w.abs().sum() + (1 - a) * (w ** 2).sum())
        return torch.zeros((), device=w.device)

    def fit(self, X, y, lr=0.1, n_iters=2000):
        dev = get_device(); self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(y, dtype=torch.float32, device=dev).ravel()
        opt = torch.optim.SGD(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(n_iters):
            opt.zero_grad()
            loss = 0.5 * loss_fn(self(X), y) + self._penalty() / len(X)
            loss.backward()
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return self(X).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def _toy(n=120, d=1, noise=0.5):
    rng = np.random.default_rng(SEED)
    X = rng.uniform(-3, 3, size=(n, d))
    w_true = rng.normal(size=d)
    y = X @ w_true + 1.2 + noise * rng.normal(size=n)
    return X, y, w_true


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    X, y, w_true = _toy()
    # standardize features so lr/penalties behave the same across scales
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xz = (X - mu) / sd

    print("true w:", np.round(w_true, 3), " true b: 1.200")

    # NumPy: closed form OLS vs gradient descent
    ols_cf = LinearRegressionNumPy(fit_closed_form=True).fit(Xz, y)
    ols_gd = LinearRegressionNumPy(lr=0.2, n_iters=3000).fit(Xz, y)
    print(f"NumPy OLS  closed-form: w={ols_cf.w.round(3)} b={ols_cf.b:.3f}")
    print(f"NumPy OLS  grad-desc  : w={ols_gd.w.round(3)} b={ols_gd.b:.3f}")

    ridge = LinearRegressionNumPy(penalty="ridge", lam=5.0, lr=0.2, n_iters=3000).fit(Xz, y)
    lasso = LinearRegressionNumPy(penalty="lasso", lam=0.3, lr=0.2, n_iters=3000).fit(Xz, y)
    print(f"NumPy Ridge          : w={ridge.w.round(3)} b={ridge.b:.3f}")
    print(f"NumPy Lasso          : w={lasso.w.round(3)} b={lasso.b:.3f}")

    # PyTorch OLS — should match NumPy
    t = LinearRegressionTorch(Xz.shape[1]).fit(Xz, y, lr=0.2, n_iters=3000)
    w_t = t.linear.weight.detach().cpu().numpy().ravel()
    b_t = float(t.linear.bias.detach().cpu())
    print(f"Torch OLS            : w={w_t.round(3)} b={b_t:.3f}")

    mse = np.mean((ols_cf.predict(Xz) - y) ** 2)
    print(f"\nOLS train MSE: {mse:.4f}")
    assert np.allclose(ols_cf.w, w_t, atol=0.05), "NumPy and Torch should agree"
    print("NumPy ≈ PyTorch ✓")


if __name__ == "__main__":
    demo()

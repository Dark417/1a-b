"""
Support Vector Regression (SVR)
===============================
SVR ports the max-margin idea to regression. Instead of fitting every point, it
fits a "tube" of half-width $\varepsilon$ around the function and ignores errors
that fall *inside* the tube (the **epsilon-insensitive loss**). Only points
outside the tube (the support vectors) shape the solution, giving a sparse,
robust regressor. As with classification SVMs, nonlinearity comes from a kernel /
feature map. We implement the primal SVR with a sub-gradient optimizer in NumPy
(linear and RBF via random Fourier features), then the same objective in PyTorch
with autograd.

Variants implemented here:
    - Linear SVR
    - RBF SVR (via random Fourier features / explicit feature map)
    - Primal sub-gradient optimization (NumPy) and autograd (PyTorch)

Training techniques demonstrated:
    - Epsilon-insensitive loss (the regression hinge) — a non-smooth convex loss
    - Sub-gradient descent on a non-differentiable objective
    - Random Fourier features to approximate the RBF kernel cheaply

References:
    - Drucker, Burges, Kaufman, Smola, Vapnik (1997), "Support Vector Regression Machines"
    - Smola & Schölkopf (2004), "A tutorial on support vector regression"
    - Rahimi & Recht (2007), "Random Features for Large-Scale Kernel Machines"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Random Fourier features: z(x) approximates an RBF kernel, <z(x),z(x')> ~ K(x,x')
# ---------------------------------------------------------------------------
def make_rff(n_features, n_rff=200, gamma=0.5, seed=SEED):
    """Return a map phi(X)->(n, n_rff) with E[<phi(x),phi(x')>] = exp(-gamma||x-x'||^2)."""
    rng = np.random.default_rng(seed)
    # Bochner's theorem: RBF kernel <-> Gaussian spectral density.
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(n_features, n_rff))
    b = rng.uniform(0, 2 * np.pi, size=n_rff)

    def phi(X):
        X = np.asarray(X, float)
        return np.sqrt(2.0 / n_rff) * np.cos(X @ W + b)
    return phi


# ---------------------------------------------------------------------------
# 1. NumPy implementation — primal SVR, sub-gradient descent
# ---------------------------------------------------------------------------
class SVRNumPy:
    r"""
    Primal epsilon-insensitive SVR.

    Objective (epsilon-insensitive loss + L2 margin term):
        min_{w,b}  (1/2)||w||^2 + C sum_i L_eps( y_i - (w^T phi(x_i)+b) )
    where the eps-insensitive loss is
        L_eps(r) = max(0, |r| - eps)        (free inside the tube |r| <= eps).

    Sub-gradient wrt the residual r_i = y_i - f(x_i):
        dL/dr = 0                if |r_i| <= eps      (inside tube, no push)
              = -sign(r_i)       if |r_i| >  eps      (pull f toward y_i)
    Chain through f = w^T phi + b:
        grad_w = w + C * sum_i s_i * phi(x_i),   grad_b = C * sum_i s_i,
        where s_i = -sign(r_i) * 1[|r_i|>eps]  (i.e. s_i = +1 above tube, -1 below).
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel="linear", n_rff=200, gamma=0.5,
                 lr=0.01, n_iters=500, seed=SEED):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.n_rff = n_rff
        self.gamma = gamma
        self.lr = lr
        self.n_iters = n_iters
        self.seed = seed

    def _phi(self, X):
        return self._map(X) if self.kernel == "rbf" else np.asarray(X, float)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if self.kernel == "rbf":
            self._map = make_rff(X.shape[1], self.n_rff, self.gamma, self.seed)
        Z = self._phi(X)                          # feature representation
        n, d = Z.shape
        w = np.zeros(d)
        b = 0.0
        self.history = []
        for _ in range(self.n_iters):
            f = Z @ w + b                         # current predictions
            r = y - f                             # residuals
            outside = np.abs(r) > self.epsilon    # points outside the tube
            # s_i = +1 if prediction is below y (r>0), -1 if above; 0 inside tube
            s = np.where(outside, -np.sign(r), 0.0)
            grad_w = w + self.C * (Z.T @ s)       # d/dw of 1/2||w||^2 + C*loss
            grad_b = self.C * s.sum()
            w -= self.lr * grad_w / n
            b -= self.lr * grad_b / n
            loss = 0.5 * w @ w + self.C * np.maximum(0, np.abs(r) - self.epsilon).sum()
            self.history.append(loss)
        self.w, self.b = w, b
        self.n_support_ = int(outside.sum())      # points outside the tube
        return self

    def predict(self, X):
        return self._phi(X) @ self.w + self.b


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — same objective, autograd
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SVRTorch(nn.Module):
    r"""
    Primal SVR in PyTorch. Forward gives f(x)=w^T phi(x)+b; the loss is
        (1/2)||w||^2 + C * sum_i max(0, |y_i - f(x_i)| - eps).
    Autograd supplies the sub-gradient of the eps-insensitive loss automatically.
    RBF nonlinearity uses explicit random Fourier features (same as NumPy).
    """

    def __init__(self, n_features, C=1.0, epsilon=0.1, kernel="linear",
                 n_rff=200, gamma=0.5, seed=SEED):
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        if kernel == "rbf":
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("Wrf", torch.randn(n_features, n_rff, generator=g) * (2 * gamma) ** 0.5)
            self.register_buffer("brf", torch.rand(n_rff, generator=g) * 2 * np.pi)
            self.n_rff = n_rff
            self.linear = nn.Linear(n_rff, 1)
        else:
            self.linear = nn.Linear(n_features, 1)

    def _phi(self, X):
        if self.kernel == "rbf":
            return (2.0 / self.n_rff) ** 0.5 * torch.cos(X @ self.Wrf + self.brf)
        return X

    def forward(self, X):
        return self.linear(self._phi(X)).squeeze(-1)

    def fit(self, X, y, lr=0.05, n_iters=500):
        dev = get_device(); self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(y, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.history = []
        for _ in range(n_iters):
            opt.zero_grad()
            r = (y - self(X)).abs()
            eps_loss = torch.clamp(r - self.epsilon, min=0).sum()
            reg = 0.5 * (self.linear.weight ** 2).sum()
            loss = reg + self.C * eps_loss
            loss.backward(); opt.step()
            self.history.append(loss.item())
        return self

    @torch.no_grad()
    def predict(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return self(X).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    # Tiny problems: a single BLAS thread avoids CPU thread-oversubscription
    # overhead that otherwise dominates these many small matmuls.
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)

    # --- linear target with noise ---
    n = 200
    Xl = rng.uniform(-3, 3, size=(n, 1))
    yl = (1.7 * Xl[:, 0] - 0.5 + rng.normal(scale=0.3, size=n))
    Xtr, ytr, Xte, yte = Xl[:160], yl[:160], Xl[160:], yl[160:]

    print("=== Linear SVR ===")
    lin = SVRNumPy(C=1.0, epsilon=0.1, kernel="linear", lr=0.05, n_iters=600).fit(Xtr, ytr)
    print(f"  NumPy  test RMSE = {_rmse(lin.predict(Xte), yte):.3f}  "
          f"(pts outside tube = {lin.n_support_}/{len(Xtr)})")
    tl = SVRTorch(1, C=1.0, epsilon=0.1, kernel="linear").fit(Xtr, ytr, lr=0.05, n_iters=600)
    print(f"  Torch  test RMSE = {_rmse(tl.predict(Xte), yte):.3f}")

    # --- nonlinear target: needs the RBF feature map ---
    Xn = rng.uniform(-3, 3, size=(n, 1))
    yn = np.sin(Xn[:, 0]) + 0.3 * Xn[:, 0] + rng.normal(scale=0.1, size=n)
    Xn = (Xn - Xn.mean(0)) / Xn.std(0)
    Xtr, ytr, Xte, yte = Xn[:160], yn[:160], Xn[160:], yn[160:]

    print("\n=== Nonlinear target (y = sin x + 0.3x) ===")
    lin_n = SVRNumPy(C=1.0, epsilon=0.05, kernel="linear", lr=0.05, n_iters=600).fit(Xtr, ytr)
    print(f"  NumPy linear SVR test RMSE = {_rmse(lin_n.predict(Xte), yte):.3f}  <- underfits")
    rbf_n = SVRNumPy(C=2.0, epsilon=0.05, kernel="rbf", n_rff=200, gamma=1.0,
                     lr=0.1, n_iters=1000).fit(Xtr, ytr)
    print(f"  NumPy RBF SVR    test RMSE = {_rmse(rbf_n.predict(Xte), yte):.3f}")
    rbf_t = SVRTorch(1, C=2.0, epsilon=0.05, kernel="rbf", n_rff=200, gamma=1.0).fit(
        Xtr, ytr, lr=0.1, n_iters=600)
    print(f"  Torch RBF SVR    test RMSE = {_rmse(rbf_t.predict(Xte), yte):.3f}")


if __name__ == "__main__":
    demo()

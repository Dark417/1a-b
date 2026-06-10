"""
Support Vector Machine (SVM)
============================
A maximum-margin classifier: among all hyperplanes that separate two classes,
pick the one whose distance to the nearest points (the *margin*) is largest. The
soft-margin version tolerates a few violations via slack variables. The dual
formulation exposes the data only through inner products, so swapping in a
**kernel** $K(x,x')$ silently lifts the model into a high- (even infinite-)
dimensional feature space — the *kernel trick* — giving nonlinear boundaries at
linear cost. We implement the dual soft-margin SVM from scratch with a simplified
SMO solver (linear / RBF / polynomial kernels), then an idiomatic PyTorch primal
SVM trained on the hinge loss.

Variants implemented here:
    - Linear kernel
    - RBF (Gaussian) kernel
    - Polynomial kernel
    - Soft-margin via the dual (simplified SMO) — NumPy
    - Soft-margin in the primal (hinge loss + L2) — PyTorch

Training techniques demonstrated:
    - Sequential Minimal Optimization (SMO): coordinate ascent on the dual
    - L2 regularization as the margin term; hinge loss as a convex surrogate
    - Sub-gradient / autograd optimization of a non-smooth loss

References:
    - Cortes & Vapnik (1995), "Support-Vector Networks"
    - Platt (1998), "Sequential Minimal Optimization"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Kernels: K(X, Z) returns the (n, m) Gram matrix of inner products in feature space
# ---------------------------------------------------------------------------
def linear_kernel(X, Z):
    # <x, z>  — the plain dot product (no lifting)
    return X @ Z.T


def rbf_kernel(X, Z, gamma=0.5):
    # exp(-gamma * ||x - z||^2)  — infinite-dimensional feature space
    x2 = (X ** 2).sum(1)[:, None]
    z2 = (Z ** 2).sum(1)[None, :]
    sq = x2 + z2 - 2 * X @ Z.T
    return np.exp(-gamma * np.maximum(sq, 0))


def poly_kernel(X, Z, degree=3, coef0=1.0, gamma=1.0):
    # (gamma <x, z> + coef0)^degree  — all monomials up to `degree`
    return (gamma * (X @ Z.T) + coef0) ** degree


def _make_kernel(kernel, **params):
    funcs = {"linear": linear_kernel, "rbf": rbf_kernel, "poly": poly_kernel}
    f = funcs[kernel]
    return lambda X, Z: f(X, Z, **params)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — dual soft-margin SVM via simplified SMO
# ---------------------------------------------------------------------------
class SVMNumPy:
    r"""
    Soft-margin SVM solved in the DUAL.

    Primal (soft margin):
        min_{w,b,xi}  (1/2)||w||^2 + C sum_i xi_i
        s.t.  y_i (w^T phi(x_i) + b) >= 1 - xi_i,   xi_i >= 0.

    Lagrangian dual (alpha_i are multipliers on the margin constraints):
        max_alpha  sum_i alpha_i - (1/2) sum_{i,j} alpha_i alpha_j y_i y_j K(x_i,x_j)
        s.t.  0 <= alpha_i <= C,   sum_i alpha_i y_i = 0.

    KKT: alpha_i = 0           -> point outside the margin (correct, ignored)
         0 < alpha_i < C       -> point ON the margin (a "free" support vector)
         alpha_i = C           -> point inside margin / misclassified
    The decision function uses only support vectors (alpha_i > 0):
        f(x) = sum_i alpha_i y_i K(x_i, x) + b.

    We optimize the dual with a SIMPLIFIED SMO: repeatedly pick a pair
    (alpha_i, alpha_j), solve that 2-variable subproblem in closed form while
    keeping sum_i alpha_i y_i = 0, and clip to [0, C].
    """

    def __init__(self, C=1.0, kernel="rbf", n_iters=100, tol=1e-3, seed=SEED, **kparams):
        self.C = C
        self.kernel_name = kernel
        self.kparams = kparams
        self.n_iters = n_iters          # max passes over the data
        self.tol = tol                  # KKT tolerance
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)         # labels must be in {-1, +1}
        n = len(X)
        self._kf = _make_kernel(self.kernel_name, **self.kparams)
        K = self._kf(X, X)               # (n, n) Gram matrix
        alpha = np.zeros(n)
        b = 0.0
        rng = np.random.default_rng(self.seed)

        # f(x_k) given current alpha, b:  sum_i alpha_i y_i K(i,k) + b
        def decision(k):
            return (alpha * y) @ K[:, k] + b

        passes = 0
        while passes < self.n_iters:
            num_changed = 0
            for i in range(n):
                Ei = decision(i) - y[i]                       # prediction error
                # Does i violate the KKT conditions (within tol)?
                if (y[i] * Ei < -self.tol and alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and alpha[i] > 0):
                    j = i
                    while j == i:                             # pick a partner j != i
                        j = rng.integers(n)
                    Ej = decision(j) - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    # Box constraints L<=alpha_j<=H that keep sum alpha y fixed
                    if y[i] != y[j]:
                        L = max(0, aj_old - ai_old); H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0, ai_old + aj_old - self.C); H = min(self.C, ai_old + aj_old)
                    if L == H:
                        continue
                    # eta = 2K_ij - K_ii - K_jj is the second derivative (<=0)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    # Unconstrained optimum for alpha_j, then clip to [L, H]
                    alpha[j] = aj_old - y[j] * (Ei - Ej) / eta
                    alpha[j] = min(H, max(L, alpha[j]))
                    if abs(alpha[j] - aj_old) < 1e-5:
                        continue
                    # Move alpha_i the opposite way to preserve sum alpha y = 0
                    alpha[i] = ai_old + y[i] * y[j] * (aj_old - alpha[j])
                    # Update bias b from the KKT stationarity at i and j
                    b1 = b - Ei - y[i] * (alpha[i] - ai_old) * K[i, i] \
                         - y[j] * (alpha[j] - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alpha[i] - ai_old) * K[i, j] \
                         - y[j] * (alpha[j] - aj_old) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    num_changed += 1
            passes = passes + 1 if num_changed == 0 else 0     # stop when a full sweep changes nothing

        # Keep only support vectors (alpha > 0) for prediction
        sv = alpha > 1e-6
        self.alpha = alpha[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]
        self.b = b
        self.w_norm2 = float((self.alpha * self.sv_y) @ self._kf(self.sv_X, self.sv_X) @ (self.alpha * self.sv_y))
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        # f(x) = sum_sv alpha_i y_i K(sv_i, x) + b
        K = self._kf(self.sv_X, X)                     # (n_sv, m)
        return (self.alpha * self.sv_y) @ K + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — primal soft-margin SVM with the hinge loss
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SVMTorch(nn.Module):
    r"""
    Primal soft-margin SVM:
        min_{w,b}  (1/2)||w||^2 + C sum_i max(0, 1 - y_i (w^T x_i + b)).
    The hinge term is the convex surrogate for the 0/1 loss; ||w||^2 is the
    (inverse) margin. We optimize directly with autograd (sub-gradient of hinge).

    For nonlinear boundaries we optionally map X through explicit RBF random
    features z(x) = sqrt(2/D) cos(W x + b_rf)  (Rahimi & Recht), so the linear
    model in z-space approximates an RBF-kernel SVM.
    """

    def __init__(self, n_features, C=1.0, feature_map="linear", n_rff=200,
                 gamma=0.5, seed=SEED):
        super().__init__()
        self.C = C
        self.feature_map = feature_map
        if feature_map == "rff":
            g = torch.Generator().manual_seed(seed)
            # Random Fourier features approximating the RBF kernel
            self.register_buffer("Wrf", torch.randn(n_features, n_rff, generator=g) * (2 * gamma) ** 0.5)
            self.register_buffer("brf", torch.rand(n_rff, generator=g) * 2 * np.pi)
            self.n_rff = n_rff
            self.linear = nn.Linear(n_rff, 1)
        else:
            self.linear = nn.Linear(n_features, 1)

    def _phi(self, X):
        if self.feature_map == "rff":
            return (2.0 / self.n_rff) ** 0.5 * torch.cos(X @ self.Wrf + self.brf)
        return X

    def forward(self, X):
        return self.linear(self._phi(X)).squeeze(-1)        # w^T phi(x) + b

    def fit(self, X, y, lr=0.05, n_iters=300):
        dev = get_device(); self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(y, dtype=torch.float32, device=dev)   # in {-1, +1}
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.history = []
        for _ in range(n_iters):
            opt.zero_grad()
            margins = y * self(X)
            hinge = torch.clamp(1 - margins, min=0).sum()
            reg = 0.5 * (self.linear.weight ** 2).sum()
            loss = reg + self.C * hinge
            loss.backward(); opt.step()
            self.history.append(loss.item())
        return self

    @torch.no_grad()
    def decision_function(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return self(X).cpu().numpy()

    @torch.no_grad()
    def predict(self, X):
        return np.sign(self.decision_function(X))


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_moons, make_blobs

    # --- linearly separable-ish: blobs ---
    Xb, yb = make_blobs(n_samples=200, centers=2, cluster_std=1.2, random_state=SEED)
    yb = np.where(yb == 0, -1, 1).astype(float)
    Xb = (Xb - Xb.mean(0)) / Xb.std(0)
    Xtr, ytr, Xte, yte = Xb[:160], yb[:160], Xb[160:], yb[160:]

    print("=== Linear SVM on blobs ===")
    lin = SVMNumPy(C=1.0, kernel="linear", n_iters=50).fit(Xtr, ytr)
    print(f"  NumPy dual (linear)  test acc = {np.mean(lin.predict(Xte) == yte):.3f}"
          f"  (#SV = {len(lin.alpha)})")
    tl = SVMTorch(2, C=1.0, feature_map="linear").fit(Xtr, ytr, lr=0.05, n_iters=300)
    print(f"  Torch primal (hinge) test acc = {np.mean(tl.predict(Xte) == yte):.3f}")

    # --- nonlinear: moons need a kernel ---
    Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=SEED)
    ym = np.where(ym == 0, -1, 1).astype(float)
    Xm = (Xm - Xm.mean(0)) / Xm.std(0)
    Xtr, ytr, Xte, yte = Xm[:240], ym[:240], Xm[240:], ym[240:]

    print("\n=== Kernel SVM on moons (nonlinear) ===")
    for name, kw in [("rbf", dict(kernel="rbf", gamma=1.0)),
                     ("poly", dict(kernel="poly", degree=3, coef0=1.0, gamma=1.0))]:
        m = SVMNumPy(C=1.0, n_iters=100, **kw).fit(Xtr, ytr)
        print(f"  NumPy dual ({name:4s}) test acc = {np.mean(m.predict(Xte) == yte):.3f}"
              f"  (#SV = {len(m.alpha)})")

    lin_m = SVMNumPy(C=1.0, kernel="linear", n_iters=100).fit(Xtr, ytr)
    print(f"  NumPy dual (linear) test acc = {np.mean(lin_m.predict(Xte) == yte):.3f}"
          f"   <- linear can't separate moons")

    rff = SVMTorch(2, C=1.0, feature_map="rff", n_rff=300, gamma=1.0).fit(
        Xtr, ytr, lr=0.05, n_iters=400)
    print(f"  Torch RFF (approx RBF) test acc = {np.mean(rff.predict(Xte) == yte):.3f}")


if __name__ == "__main__":
    demo()

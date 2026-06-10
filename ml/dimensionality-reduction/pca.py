"""
Principal Component Analysis (PCA)
==================================
Find the orthogonal directions of maximum variance and project onto the top
few. The canonical linear dimensionality-reduction / compression method, and a
beautiful application of the eigen/SVD decomposition.

Variants implemented here:
    - PCA via eigendecomposition of the covariance matrix
    - PCA via SVD (numerically preferred)
    - Whitening (unit-variance components)
    - Kernel PCA (RBF) for nonlinear structure

Training techniques demonstrated:
    - Centering (and why it's mandatory)
    - Choosing #components via explained-variance ratio

References:
    - Jolliffe, "Principal Component Analysis"; Bishop PRML ch. 12
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation
# ---------------------------------------------------------------------------
class PCANumPy:
    r"""
    Center X. The covariance is  C = (1/n) X^T X.  Its eigenvectors are the
    principal directions; eigenvalues are the variance along them.
    Equivalently, SVD  X = U S V^T  gives components V and variances S^2/n.
    """

    def __init__(self, n_components=2, method="svd", whiten=False):
        self.n_components, self.method, self.whiten = n_components, method, whiten

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_                              # centering is mandatory
        n = len(X)
        if self.method == "eig":
            C = (Xc.T @ Xc) / n                          # covariance
            vals, vecs = np.linalg.eigh(C)              # ascending
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]
            self.components_ = vecs[:, :self.n_components].T
            self.explained_variance_ = vals[:self.n_components]
            total = vals.sum()
        else:  # SVD — more stable, no explicit covariance
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.components_ = Vt[:self.n_components]
            self.explained_variance_ = (S[:self.n_components] ** 2) / n
            total = (S ** 2).sum() / n
        self.explained_variance_ratio_ = self.explained_variance_ / total
        return self

    def transform(self, X):
        Xc = np.asarray(X, float) - self.mean_
        Z = Xc @ self.components_.T
        if self.whiten:                                 # unit variance per comp
            Z = Z / np.sqrt(self.explained_variance_ + 1e-12)
        return Z

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_ + 1e-12)
        return Z @ self.components_ + self.mean_


class KernelPCANumPy:
    """Nonlinear PCA in feature space via the kernel trick (RBF)."""

    def __init__(self, n_components=2, gamma=1.0):
        self.n_components, self.gamma = n_components, gamma

    def _kernel(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(2)
        return np.exp(-self.gamma * d2)

    def fit_transform(self, X):
        X = np.asarray(X, float); self.X_fit = X
        n = len(X)
        K = self._kernel(X, X)
        one = np.ones((n, n)) / n
        Kc = K - one @ K - K @ one + one @ K @ one      # center in feature space
        vals, vecs = np.linalg.eigh(Kc)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        # eigenvectors scaled by 1/sqrt(eigenvalue) give unit-norm projections
        alphas = vecs[:, :self.n_components] / np.sqrt(vals[:self.n_components] + 1e-12)
        self.alphas_, self.Kc_ = alphas, Kc
        return Kc @ alphas


# ---------------------------------------------------------------------------
# 2. PyTorch implementation
# ---------------------------------------------------------------------------
import torch


def pca_torch(X, n_components=2):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Xc = Xt - Xt.mean(0, keepdim=True)
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    Z = Xc @ comps.T
    return Z.cpu().numpy(), comps.cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import load_iris, make_circles

    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)

    pe = PCANumPy(2, method="eig").fit(X)
    ps = PCANumPy(2, method="svd").fit(X)
    print("explained variance ratio (eig):", np.round(pe.explained_variance_ratio_, 3))
    print("explained variance ratio (svd):", np.round(ps.explained_variance_ratio_, 3))
    print("eig ≈ svd components:", np.allclose(np.abs(pe.components_), np.abs(ps.components_), atol=1e-4))

    Z, _ = pca_torch(X, 2)
    print("torch top-2 shape:", Z.shape)

    recon = ps.inverse_transform(ps.transform(X))
    print(f"reconstruction MSE (2 of 4 dims): {np.mean((recon - X) ** 2):.4f}")

    # kernel PCA untangles concentric circles that linear PCA cannot
    Xc, yc = make_circles(n_samples=300, factor=0.3, noise=0.05, random_state=SEED)
    Zk = KernelPCANumPy(2, gamma=10).fit_transform(Xc)
    sep = abs(Zk[yc == 0, 0].mean() - Zk[yc == 1, 0].mean())
    print(f"kernel-PCA class separation on 1st component: {sep:.3f}")


if __name__ == "__main__":
    demo()

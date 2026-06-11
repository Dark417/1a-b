"""
Fisher's Linear Discriminant Analysis (LDA)
===========================================
A *supervised* linear dimensionality reduction: find the directions that
maximize between-class separation relative to within-class scatter. Unlike PCA
(which only looks at total variance, ignoring labels), LDA actively projects the
data so that classes are pushed apart while each class stays compact. The
projection axes solve a generalized eigenvalue problem on the scatter matrices.

Variants implemented here:
    - Two-class Fisher discriminant (the classic w ∝ S_W^{-1}(m1 - m0))
    - Multi-class LDA via the generalized eigenproblem  S_B w = λ S_W w
    - Shrinkage / regularized within-class scatter (S_W + γI) for stability
    - LDA as a Gaussian classifier (shared covariance ⇒ linear decision rule)

Training techniques demonstrated:
    - Whitening the within-class scatter to turn a generalized eigenproblem
      into a standard symmetric one (numerically robust)
    - Regularization (shrinkage) when S_W is singular / ill-conditioned

References:
    - Fisher (1936), "The use of multiple measurements in taxonomic problems"
    - Bishop, PRML §4.1.4–4.1.6; Hastie/Tibshirani/Friedman ESL §4.3
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
class LDANumPy:
    r"""
    Fisher LDA. With class means :math:`m_c`, global mean :math:`m`, and
    per-class counts :math:`n_c`:

        within-class scatter   S_W = sum_c sum_{i in c} (x_i - m_c)(x_i - m_c)^T
        between-class scatter   S_B = sum_c n_c (m_c - m)(m_c - m)^T

    We seek directions :math:`w` maximizing the Rayleigh quotient

        J(w) = (w^T S_B w) / (w^T S_W w).

    Stationarity gives the generalized eigenproblem  S_B w = λ S_W w; the top
    eigenvectors (largest λ) are the discriminant axes. At most C-1 of them are
    nonzero because S_B has rank ≤ C-1.
    """

    def __init__(self, n_components=None, shrinkage=0.0):
        # shrinkage adds gamma*I to S_W (regularization toward a sphere)
        self.n_components = n_components
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        mean_global = X.mean(0)

        S_W = np.zeros((n_features, n_features))   # within-class scatter
        S_B = np.zeros((n_features, n_features))   # between-class scatter
        self.means_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            m_c = Xc.mean(0)
            self.means_[c] = m_c
            Xc_centered = Xc - m_c
            S_W += Xc_centered.T @ Xc_centered          # pooled scatter within c
            diff = (m_c - mean_global).reshape(-1, 1)
            S_B += len(Xc) * (diff @ diff.T)            # n_c * outer(mean diff)

        # Shrinkage / ridge: keep S_W invertible (vital when n < d or collinear).
        if self.shrinkage > 0:
            S_W = S_W + self.shrinkage * np.trace(S_W) / n_features * np.eye(n_features)

        # Solve S_B w = λ S_W w. Whiten by S_W: let A = S_W^{-1/2} S_B S_W^{-1/2},
        # a SYMMETRIC matrix, so eigh is stable; map eigenvectors back by S_W^{-1/2}.
        # eigh on S_W gives S_W = U diag(s) U^T, so S_W^{-1/2} = U diag(s^{-1/2}) U^T.
        s, U = np.linalg.eigh(S_W)
        s = np.maximum(s, 1e-12)                        # guard tiny/neg eigenvalues
        S_W_inv_half = U @ np.diag(1.0 / np.sqrt(s)) @ U.T
        A = S_W_inv_half @ S_B @ S_W_inv_half
        A = (A + A.T) / 2                               # symmetrize against fp drift
        eigvals, eigvecs = np.linalg.eigh(A)            # ascending order

        order = np.argsort(eigvals)[::-1]               # largest λ first
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        W = S_W_inv_half @ eigvecs                      # back to original space
        # normalize columns to unit length for interpretability
        W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)

        max_comp = len(self.classes_) - 1               # rank(S_B) <= C-1
        k = self.n_components or max_comp
        k = min(k, max_comp, n_features)
        self.scalings_ = W[:, :k]
        self.eigenvalues_ = eigvals[:k]
        # explained "discriminability" ratio (separation captured per axis)
        pos = np.maximum(eigvals, 0)
        self.explained_variance_ratio_ = (pos[:k] / (pos.sum() + 1e-12))
        return self

    def transform(self, X):
        return np.asarray(X, float) @ self.scalings_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        """LDA as a classifier: nearest class mean in the *projected* space
        (equivalent to a shared-covariance Gaussian / linear discriminant)."""
        Z = self.transform(X)
        proj_means = np.stack([self.means_[c] @ self.scalings_ for c in self.classes_])
        d2 = ((Z[:, None, :] - proj_means[None, :, :]) ** 2).sum(2)
        return self.classes_[d2.argmin(1)]


def fisher_two_class(X, y):
    r"""Closed-form two-class Fisher direction  w ∝ S_W^{-1}(m_1 - m_0).

    For two classes the generalized eigenproblem collapses to this single
    direction (S_B has rank 1), which is why it has a tidy closed form."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    classes = np.unique(y)
    assert len(classes) == 2, "two-class only"
    m0, m1 = X[y == classes[0]].mean(0), X[y == classes[1]].mean(0)
    S_W = np.zeros((X.shape[1],) * 2)
    for c, m in zip(classes, (m0, m1)):
        Xc = X[y == c] - m
        S_W += Xc.T @ Xc
    w = np.linalg.solve(S_W, m1 - m0)                   # S_W^{-1} (m1 - m0)
    return w / np.linalg.norm(w)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation
# ---------------------------------------------------------------------------
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lda_torch(X, y, n_components=None, shrinkage=0.0):
    """LDA via the generalized eigenproblem using torch.linalg.

    Same whitening trick as the NumPy version: form S_W^{-1/2} S_B S_W^{-1/2}
    and use the symmetric eigensolver (eigh)."""
    dev = get_device()
    Xt = torch.as_tensor(np.asarray(X, float), dtype=torch.float64, device=dev)
    y = np.asarray(y)
    classes = np.unique(y)
    d = Xt.shape[1]
    mean_global = Xt.mean(0)
    S_W = torch.zeros((d, d), dtype=torch.float64, device=dev)
    S_B = torch.zeros((d, d), dtype=torch.float64, device=dev)
    for c in classes:
        Xc = Xt[torch.as_tensor(y == c, device=dev)]
        m_c = Xc.mean(0)
        Xc_c = Xc - m_c
        S_W += Xc_c.T @ Xc_c
        diff = (m_c - mean_global).reshape(-1, 1)
        S_B += Xc.shape[0] * (diff @ diff.T)
    if shrinkage > 0:
        S_W = S_W + shrinkage * torch.trace(S_W) / d * torch.eye(d, dtype=torch.float64, device=dev)

    s, U = torch.linalg.eigh(S_W)
    s = torch.clamp(s, min=1e-12)
    S_W_inv_half = U @ torch.diag(1.0 / torch.sqrt(s)) @ U.T
    A = S_W_inv_half @ S_B @ S_W_inv_half
    A = (A + A.T) / 2
    eigvals, eigvecs = torch.linalg.eigh(A)
    order = torch.argsort(eigvals, descending=True)
    W = S_W_inv_half @ eigvecs[:, order]
    W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    k = (n_components or (len(classes) - 1))
    k = min(k, len(classes) - 1, d)
    W = W[:, :k]
    Z = Xt @ W
    return Z.cpu().numpy(), W.cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    from sklearn.datasets import load_iris
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X, y = load_iris(return_X_y=True)

    lda = LDANumPy(n_components=2).fit(X, y)
    Z = lda.transform(X)
    print("LDA eigenvalues (separation per axis):", np.round(lda.eigenvalues_, 3))
    print("explained discriminability ratio:", np.round(lda.explained_variance_ratio_, 3))

    # Class separation in the 1D Fisher projection vs raw: ratio of between/within var.
    acc = (lda.predict(X) == y).mean()
    print(f"LDA-as-classifier training accuracy: {acc:.3f}")

    # Compare with sklearn (sign/scale may differ; compare absolute correlation of axis 1).
    sk = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
    Zsk = sk.transform(X)
    corr = abs(np.corrcoef(Z[:, 0], Zsk[:, 0])[0, 1])
    print(f"|corr| of 1st LDA axis vs sklearn: {corr:.3f}")
    print(f"sklearn LDA training accuracy:      {sk.score(X, y):.3f}")

    # Two-class closed form (setosa vs rest collapsed to two classes).
    yb = (y > 0).astype(int)
    w = fisher_two_class(X, yb)
    proj = X @ w
    sep = abs(proj[yb == 0].mean() - proj[yb == 1].mean())
    within = proj[yb == 0].std() + proj[yb == 1].std()
    print(f"two-class Fisher separation/within-spread: {sep / (within + 1e-9):.3f}")

    # Torch path agrees with NumPy on the projected coordinates (up to sign).
    Zt, _ = lda_torch(X, y, n_components=2)
    agree = abs(np.corrcoef(Z[:, 0], Zt[:, 0])[0, 1])
    print(f"|corr| numpy vs torch 1st axis: {agree:.3f}")


if __name__ == "__main__":
    demo()

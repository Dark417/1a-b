"""
Truncated SVD / Low-Rank Matrix Factorization
==============================================
The Singular Value Decomposition factorizes ANY matrix as A = U Σ Vᵀ. Keeping
only the top-k singular triplets gives the best rank-k approximation of A in
both the Frobenius and spectral norms (the Eckart–Young theorem). This is the
engine behind dimensionality reduction (truncated SVD, a.k.a. PCA without
centering), low-rank compression, image compression, and Latent Semantic
Analysis (LSA) for text.

Variants implemented here:
    - Full SVD (NumPy linalg) and the geometry of U, Σ, V
    - Truncated SVD (top-k) for dimensionality reduction (sklearn-style)
    - Randomized SVD (fast approximate top-k via a random projection sketch)
    - Low-rank approximation + Eckart–Young error check
    - LSA note: TruncatedSVD on a term–document (TF-IDF-like) matrix

Training techniques demonstrated:
    - Randomized range finding (Halko, Martinsson, Tropp) with power iterations
    - Choosing rank k via the singular-value spectrum / explained variance

References:
    - Eckart & Young (1936); Golub & Van Loan, "Matrix Computations"
    - Halko, Martinsson & Tropp (2011), "Finding Structure with Randomness"
    - Deerwester et al. (1990), "Indexing by Latent Semantic Analysis"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch where it teaches; linalg for the core)
# ---------------------------------------------------------------------------
class TruncatedSVDNumPy:
    r"""
    Truncated SVD for dimensionality reduction.

    Decompose  A = U Σ Vᵀ  with singular values σ_1 ≥ σ_2 ≥ ... ≥ 0. Keep the
    top k:  A_k = U_k Σ_k V_kᵀ.  The reduced representation of the rows of A is

        Z = A V_k = U_k Σ_k      (an n×k embedding),

    and the rank-k reconstruction is  Â = Z V_kᵀ.  Unlike PCA we do NOT center
    (so it works directly on sparse term–document matrices — that is LSA).
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, A):
        A = np.asarray(A, float)
        # Economy SVD: U (n×r), s (r,), Vt (r×d) with r = min(n, d).
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        k = self.n_components
        self.components_ = Vt[:k]                  # V_kᵀ : right singular vectors
        self.singular_values_ = s[:k]
        # explained variance ratio = σ_i² / Σ σ² (variance of A·v_i)
        var = s ** 2
        self.explained_variance_ratio_ = var[:k] / (var.sum() + 1e-12)
        self._U, self._s = U, s
        return self

    def transform(self, A):
        # Z = A V_k  (project rows onto the top-k right singular vectors)
        return np.asarray(A, float) @ self.components_.T

    def fit_transform(self, A):
        self.fit(A)
        # equals U_k Σ_k; compute directly for numerical consistency
        return self._U[:, : self.n_components] * self._s[: self.n_components]

    def inverse_transform(self, Z):
        return Z @ self.components_                 # back to original space


def low_rank_approx(A, k):
    r"""
    Best rank-k approximation A_k = Σ_{i≤k} σ_i u_i v_iᵀ.

    Eckart–Young: among all rank-k matrices B,
        ||A - A_k||_F = sqrt(Σ_{i>k} σ_i²)   is the minimum,
        ||A - A_k||_2 = σ_{k+1}              is the minimum (spectral norm).
    """
    A = np.asarray(A, float)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    frob_err = np.sqrt((s[k:] ** 2).sum())          # predicted Frobenius error
    spec_err = s[k] if k < len(s) else 0.0          # predicted spectral error
    return Ak, frob_err, spec_err


def randomized_svd(A, k, n_oversamples=10, n_power_iter=2, seed=SEED):
    r"""
    Fast approximate top-k SVD via a random projection sketch.

    Idea: draw a random Gaussian Ω (d×(k+p)); the sketch Y = A Ω captures the
    dominant range of A. Orthonormalize Q = qr(Y); then B = Qᵀ A is small
    ((k+p)×d), and SVD(B) is cheap. Power iterations (A Aᵀ)^q sharpen the
    spectrum so leading directions dominate the sketch.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float)
    n, d = A.shape
    p = k + n_oversamples
    Omega = rng.standard_normal((d, p))
    Y = A @ Omega
    for _ in range(n_power_iter):                   # power iteration for accuracy
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)                          # orthonormal basis of range
    B = Q.T @ A                                     # project to a small matrix
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub                                      # lift back
    return U[:, :k], s[:k], Vt[:k]


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


def truncated_svd_torch(A, k=2):
    """Top-k truncated SVD using torch.linalg.svd (LAPACK-backed)."""
    dev = get_device()
    At = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=dev)
    U, s, Vt = torch.linalg.svd(At, full_matrices=False)
    Z = U[:, :k] * s[:k]                            # n×k embedding (U_k Σ_k)
    return Z.cpu().numpy(), Vt[:k].cpu().numpy(), s[:k].cpu().numpy()


def low_rank_sgd_torch(A, k=2, n_iter=300, lr=1e-1, seed=SEED):
    r"""
    Low-rank factorization A ≈ W Hᵀ learned by gradient descent (W:n×k, H:d×k).
    Minimizes ||A - W Hᵀ||_F² — a non-convex objective whose global optima span
    the same subspace as the top-k SVD. Illustrates SVD as matrix factorization.
    """
    dev = get_device()
    At = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=dev)
    n, d = At.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = (torch.randn(n, k, generator=g) * 0.1).to(dev).requires_grad_(True)
    H = (torch.randn(d, k, generator=g) * 0.1).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([W, H], lr=lr)
    loss = None
    for _ in range(n_iter):
        loss = ((At - W @ H.T) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return W.detach().cpu().numpy(), H.detach().cpu().numpy(), float(loss.detach())


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    from sklearn.datasets import load_digits

    digits = load_digits()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(digits.data), 300, replace=False)
    A, y = digits.data[idx], digits.target[idx]    # 300 × 64 (no centering: SVD/LSA)

    # Truncated SVD reduction --------------------------------------------
    svd = TruncatedSVDNumPy(n_components=10).fit(A)
    print("top singular values:", np.round(svd.singular_values_[:5], 1), "...")
    print("explained variance ratio (top10 sum):",
          round(float(svd.explained_variance_ratio_.sum()), 3))

    # Eckart–Young low-rank approximation error ---------------------------
    for k in (5, 10, 20):
        Ak, pred_frob, pred_spec = low_rank_approx(A, k)
        actual = np.linalg.norm(A - Ak)
        print(f"rank {k:2d}: actual Frob err={actual:8.1f}  "
              f"predicted (√Σσ²)={pred_frob:8.1f}  spectral={pred_spec:6.1f}")

    # Randomized SVD matches the deterministic top-k -----------------------
    Ur, sr, Vtr = randomized_svd(A, k=10)
    _, sd, _ = np.linalg.svd(A, full_matrices=False)
    rel = np.abs(sr - sd[:10]) / (sd[:10] + 1e-12)
    print(f"randomized vs exact singular values max rel-err: {rel.max():.4f}")

    # Torch truncated SVD agrees on the spectrum --------------------------
    _, _, st = truncated_svd_torch(A, k=10)
    print("torch vs numpy top-5 σ close:",
          np.allclose(st[:5], sd[:5], rtol=1e-3, atol=1e-1))

    # SGD low-rank factorization reaches the SVD optimum (smaller subset so the
    # autograd loop is quick on CPU) ---------------------------------------
    A_sgd = A[:150]
    W, H, fit_err = low_rank_sgd_torch(A_sgd, k=10, n_iter=150, lr=0.5)
    Ak, _, _ = low_rank_approx(A_sgd, 10)
    svd_err = (np.linalg.norm(A_sgd - Ak)) ** 2
    print(f"SGD factorization ‖A-WHᵀ‖²={fit_err:.0f}  vs  SVD optimum={svd_err:.0f}")

    # LSA note: TruncatedSVD on a tiny term–document matrix ---------------
    # rows = terms, cols = documents; latent factors group co-occurring terms.
    docs = np.array([
        [3, 0, 1, 0],   # "money"
        [2, 0, 1, 0],   # "bank"
        [0, 3, 0, 2],   # "river"
        [0, 2, 0, 3],   # "water"
    ], float)
    Zlsa = TruncatedSVDNumPy(n_components=2).fit_transform(docs)
    print("LSA latent coords (terms):\n", np.round(Zlsa, 2))
    print("-> 'money'/'bank' cluster apart from 'river'/'water' in latent space")


if __name__ == "__main__":
    demo()

"""
Spectral Clustering
===================
Cluster by the *geometry of a similarity graph* rather than raw coordinates.
Build an affinity graph over the points, form the (normalized) graph Laplacian,
embed the points using its smallest non-trivial eigenvectors, then run k-means in
that low-dimensional embedding. Because the embedding "unrolls" manifolds, this
separates non-convex shapes (moons, rings) that k-means cannot.

Pipeline:
    1. Affinity W (RBF kernel, optionally k-NN sparsified).
    2. Degree D = diag(row sums); Laplacian L = D - W (or a normalized variant).
    3. Take the eigenvectors of the k smallest eigenvalues -> embedding U (n x k).
    4. (For the normalized symmetric variant) row-normalize U, then k-means on U.

Variants implemented here:
    - Unnormalized Laplacian          L     = D - W
    - Symmetric normalized (Ng-Jordan-Weiss)  L_sym = I - D^-1/2 W D^-1/2
    - Random-walk normalized (Shi-Malik)      L_rw  = I - D^-1 W
    - NumPy (np.linalg.eigh) + PyTorch (torch.linalg.eigh, torch.cdist affinity)

Training techniques demonstrated:
    - Relaxing the NP-hard normalized-cut to an eigenproblem (Rayleigh quotient)
    - Laplacian eigenmaps: eigenvectors as a smooth low-dim embedding
    - Affinity scale (gamma) and graph sparsification choices

References:
    - Shi & Malik (2000), "Normalized Cuts and Image Segmentation"
    - Ng, Jordan & Weiss (2002), "On Spectral Clustering"
    - von Luxburg (2007), "A Tutorial on Spectral Clustering"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _kmeans(X, k, n_iters=100, n_init=10, seed=SEED):
    """Small self-contained k-means++ for the embedding step (no sibling import)."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    best_inertia, best_labels = np.inf, None
    for _ in range(n_init):
        C = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = ((X[:, None, :] - np.array(C)[None, :, :]) ** 2).sum(2).min(1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C, float)
        for _ in range(n_iters):
            D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
            lab = D.argmin(1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                C = newC; break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, lab
    return best_labels


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
class SpectralClusteringNumPy:
    r"""
    Graph-cut view. Partition the similarity graph to minimize the **normalized
    cut**. With cluster indicator vectors this is NP-hard; relaxing the indicators
    to real values turns it into a generalized eigenproblem on the Laplacian.

    Laplacians (W = affinity, D = degree diag):
      unnormalized: L      = D - W
      symmetric:    L_sym  = I - D^{-1/2} W D^{-1/2}
      random walk:  L_rw   = I - D^{-1} W   (eigvecs = those of  D^{-1}W)

    The k eigenvectors with the smallest eigenvalues give a piecewise-constant
    embedding on which ordinary k-means recovers the clusters. For L_sym the
    rows are renormalized to unit length (Ng-Jordan-Weiss).
    """

    def __init__(self, n_clusters=2, affinity="rbf", gamma=1.0, n_neighbors=10,
                 laplacian="sym", seed=SEED):
        assert laplacian in ("unnormalized", "sym", "rw")
        self.n_clusters, self.affinity, self.gamma = n_clusters, affinity, gamma
        self.n_neighbors, self.laplacian, self.seed = n_neighbors, laplacian, seed

    def _affinity(self, X):
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(2)
        if self.affinity == "knn":
            # symmetric k-NN graph: connect i~j if either is in the other's kNN
            W = np.zeros_like(d2)
            idx = np.argsort(d2, axis=1)[:, 1:self.n_neighbors + 1]
            for i in range(len(X)):
                W[i, idx[i]] = np.exp(-self.gamma * d2[i, idx[i]])
            W = np.maximum(W, W.T)                     # make symmetric
        else:                                         # full RBF (Gaussian) affinity
            W = np.exp(-self.gamma * d2)
            np.fill_diagonal(W, 0.0)                   # no self-loops
        return W

    def fit(self, X):
        X = np.asarray(X, float)
        W = self._affinity(X)
        deg = W.sum(1)                                 # node degrees
        D = np.diag(deg)

        if self.laplacian == "unnormalized":
            L = D - W
            vals, vecs = np.linalg.eigh(L)
            U = vecs[:, :self.n_clusters]              # smallest eigenvalues
        elif self.laplacian == "sym":
            d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-12)
            Lsym = np.eye(len(X)) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
            vals, vecs = np.linalg.eigh(Lsym)
            U = vecs[:, :self.n_clusters]
            U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)  # row-normalize
        else:  # random walk: solve L_rw u = lambda u  <=>  generalized (D-W)u=lambda D u
            d_inv = 1.0 / (deg + 1e-12)
            Lrw = np.eye(len(X)) - d_inv[:, None] * W
            vals, vecs = np.linalg.eig(Lrw)            # Lrw is not symmetric
            order = np.argsort(vals.real)
            U = vecs[:, order[:self.n_clusters]].real

        self.embedding_ = U
        self.eigenvalues_ = np.sort(vals.real)[:self.n_clusters + 1]
        self.affinity_matrix_ = W
        self.labels_ = _kmeans(U, self.n_clusters, seed=self.seed)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (GPU-friendly affinity + eigendecomposition)
# ---------------------------------------------------------------------------
import torch


def get_device():
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def spectral_torch(X, n_clusters=2, gamma=1.0, seed=SEED):
    """Symmetric-normalized spectral embedding via torch, then NumPy k-means.

    Heavy linear algebra (affinity via torch.cdist, eigh of the Laplacian) runs
    on the device; only the tiny final k-means runs on CPU/NumPy.
    """
    dev = get_device()
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev)
    D2 = torch.cdist(Xt, Xt) ** 2                      # squared distances
    W = torch.exp(-gamma * D2)
    W.fill_diagonal_(0.0)
    deg = W.sum(1)
    d_inv_sqrt = (deg + 1e-12).rsqrt()
    Lsym = torch.eye(len(Xt), device=dev) - d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
    vals, vecs = torch.linalg.eigh(Lsym)               # ascending eigenvalues
    U = vecs[:, :n_clusters]
    U = U / (U.norm(dim=1, keepdim=True) + 1e-12)
    labels = _kmeans(U.cpu().numpy(), n_clusters, seed=seed)
    return labels, U.cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_moons, make_circles, make_blobs
    from sklearn.metrics import adjusted_rand_score

    # two moons: spectral separates the non-convex shapes; k-means cannot
    Xm, ym = make_moons(n_samples=300, noise=0.06, random_state=SEED)
    sc = SpectralClusteringNumPy(2, affinity="rbf", gamma=15, laplacian="sym").fit(Xm)
    km = _kmeans(Xm, 2)
    print(f"Moons:   spectral ARI={adjusted_rand_score(ym, sc.labels_):.3f}  "
          f"k-means ARI={adjusted_rand_score(ym, km):.3f}")

    # concentric circles: same story
    Xc, yc = make_circles(n_samples=300, factor=0.4, noise=0.05, random_state=SEED)
    scc = SpectralClusteringNumPy(2, affinity="rbf", gamma=15, laplacian="sym").fit(Xc)
    print(f"Circles: spectral ARI={adjusted_rand_score(yc, scc.labels_):.3f}  "
          f"k-means ARI={adjusted_rand_score(yc, _kmeans(Xc, 2)):.3f}")

    print("\nLaplacian variants on moons (ARI):")
    for lap in ("unnormalized", "sym", "rw"):
        s = SpectralClusteringNumPy(2, gamma=15, laplacian=lap).fit(Xm)
        print(f"  {lap:12s}: {adjusted_rand_score(ym, s.labels_):.3f}")

    lab, U = spectral_torch(Xm, n_clusters=2, gamma=15)
    print(f"\nTorch spectral (sym) ARI={adjusted_rand_score(ym, lab):.3f}  device={get_device()}")

    # the "eigengap" hints at the number of clusters
    Xb, yb = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=SEED)
    sb = SpectralClusteringNumPy(4, gamma=2, laplacian="sym").fit(Xb)
    print(f"\nBlobs(k=4) smallest eigenvalues: {np.round(sb.eigenvalues_, 3)} "
          f"(gap after #clusters)")


if __name__ == "__main__":
    demo()

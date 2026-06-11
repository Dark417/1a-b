"""
k-Means Clustering
==================
Partition data into k clusters by alternating: (E) assign each point to its
nearest centroid, (M) move each centroid to the mean of its points. This is
coordinate descent on the within-cluster sum of squares (inertia).

Variants implemented here:
    - Lloyd's algorithm (vanilla)
    - k-means++ smart initialization
    - Mini-batch k-means (scales to large n)
    - Model selection: elbow (inertia) and silhouette

Training techniques demonstrated:
    - Sensitivity to initialization → k-means++ and multiple restarts
    - Choosing k (elbow / silhouette)

References:
    - Lloyd (1957/1982); Arthur & Vassilvitskii (2007), "k-means++"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _dist2(X, C):
    """Squared Euclidean distance, X:(n,d) C:(k,d) -> (n,k)."""
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)


# ---------------------------------------------------------------------------
# 1. NumPy implementation
# ---------------------------------------------------------------------------
class KMeansNumPy:
    def __init__(self, k=3, init="kmeans++", n_iters=100, n_init=10, tol=1e-6, seed=SEED):
        self.k, self.init, self.n_iters = k, init, n_iters
        self.n_init, self.tol, self.seed = n_init, tol, seed
        self.centroids = None; self.labels_ = None; self.inertia_ = np.inf

    def _init_centroids(self, X, rng):
        if self.init == "random":
            return X[rng.choice(len(X), self.k, replace=False)]
        # --- k-means++: pick spread-out seeds proportional to D^2 ---
        C = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = _dist2(X, np.array(C)).min(1)        # dist to nearest chosen
            probs = d2 / d2.sum()                     # farther -> likelier
            C.append(X[rng.choice(len(X), p=probs)])
        return np.array(C)

    def _fit_once(self, X, rng):
        C = self._init_centroids(X, rng).astype(float)
        for _ in range(self.n_iters):
            labels = _dist2(X, C).argmin(1)           # E-step: assign
            newC = np.array([X[labels == j].mean(0) if np.any(labels == j)
                             else C[j] for j in range(self.k)])   # M-step: update
            if np.linalg.norm(newC - C) < self.tol:
                C = newC; break
            C = newC
        inertia = _dist2(X, C)[np.arange(len(X)), labels].sum()
        return C, labels, inertia

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.n_init):                  # multiple restarts
            C, labels, inertia = self._fit_once(X, rng)
            if inertia < self.inertia_:
                self.centroids, self.labels_, self.inertia_ = C, labels, inertia
        return self

    def predict(self, X):
        return _dist2(np.asarray(X, float), self.centroids).argmin(1)


class MiniBatchKMeansNumPy:
    """Streaming/online update — each step uses a random mini-batch."""

    def __init__(self, k=3, batch=64, n_iters=300, seed=SEED):
        self.k, self.batch, self.n_iters, self.seed = k, batch, n_iters, seed

    def fit(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        C = X[rng.choice(len(X), self.k, replace=False)].astype(float)
        counts = np.zeros(self.k)
        for _ in range(self.n_iters):
            b = X[rng.choice(len(X), self.batch, replace=False)]
            lab = _dist2(b, C).argmin(1)
            for j in range(self.k):
                pts = b[lab == j]
                for x in pts:                         # per-center running mean
                    counts[j] += 1
                    C[j] += (x - C[j]) / counts[j]
        self.centroids = C
        self.labels_ = _dist2(X, C).argmin(1)
        return self

    def predict(self, X):
        return _dist2(np.asarray(X, float), self.centroids).argmin(1)


def silhouette_score(X, labels):
    """Mean silhouette: (b - a) / max(a, b). +1 good, 0 overlapping, -1 wrong."""
    X = np.asarray(X, float); labels = np.asarray(labels)
    D = np.sqrt(np.maximum(_dist2(X, X), 0))
    sil = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]; same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = min((D[i, labels == c].mean()
                 for c in np.unique(labels) if c != labels[i]), default=0.0)
        sil[i] = (b - a) / (max(a, b) + 1e-12)
    return sil.mean()


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (GPU-friendly Lloyd)
# ---------------------------------------------------------------------------
import torch


def kmeans_torch(X, k=3, n_iters=100, seed=SEED):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    idx = torch.randperm(len(Xt), generator=g)[:k]
    C = Xt[idx].clone()
    for _ in range(n_iters):
        D = torch.cdist(Xt, C)                        # (n, k)
        labels = D.argmin(1)
        newC = torch.stack([Xt[labels == j].mean(0) if (labels == j).any() else C[j]
                            for j in range(k)])
        if torch.allclose(newC, C, atol=1e-6):
            C = newC; break
        C = newC
    return C.cpu().numpy(), labels.cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_blobs

    X, ytrue = make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=SEED)

    km = KMeansNumPy(k=4).fit(X)
    print(f"k-means++   inertia={km.inertia_:.1f}  silhouette={silhouette_score(X, km.labels_):.3f}")

    mb = MiniBatchKMeansNumPy(k=4).fit(X)
    mb_in = _dist2(X, mb.centroids)[np.arange(len(X)), mb.labels_].sum()
    print(f"mini-batch  inertia={mb_in:.1f}")

    C, lab = kmeans_torch(X, k=4)
    t_in = ((X - C[lab]) ** 2).sum()
    print(f"torch       inertia={t_in:.1f}")

    print("\nElbow (inertia vs k):")
    for k in range(2, 7):
        print(f"  k={k}: inertia={KMeansNumPy(k=k, n_init=3).fit(X).inertia_:.1f}")


if __name__ == "__main__":
    demo()

"""
k-Nearest Neighbours (kNN)
==========================
The simplest non-parametric model: to predict a point, look at its k closest
neighbours in the training set and let them vote (classification) or average
(regression). No training — all the work is at query time ("lazy learning").

Variants implemented here:
    - Classification (majority vote) and regression (mean of neighbours)
    - Uniform vs distance-weighted voting
    - Distance metrics: Euclidean (L2), Manhattan (L1), Minkowski-p
    - Brute force vs a simple KD-tree (for low dimensions)

Training techniques demonstrated:
    - The bias–variance trade-off via k (small k = high variance)
    - Why feature scaling matters for distance-based methods

References:
    - Cover & Hart (1967), "Nearest neighbor pattern classification"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _pairwise(A, B, p=2):
    """||a - b||_p for every pair. A:(n,d) B:(m,d) -> (n,m)."""
    if p == 2:  # ||a-b||^2 = |a|^2 + |b|^2 - 2 a·b  (fast, vectorized)
        a2 = (A**2).sum(1)[:, None]
        b2 = (B**2).sum(1)[None, :]
        d2 = np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)
        return np.sqrt(d2)
    return np.power(np.abs(A[:, None, :] - B[None, :, :]) ** p, 1).sum(2) ** (1 / p)


# ---------------------------------------------------------------------------
# 1. NumPy implementation
# ---------------------------------------------------------------------------
class KNNNumPy:
    def __init__(self, k=5, task="classification", weights="uniform", p=2):
        self.k, self.task, self.weights, self.p = k, task, weights, p

    def fit(self, X, y):                 # "training" = remember the data
        self.X = np.asarray(X, float)
        self.y = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        D = _pairwise(X, self.X, self.p)             # (n_query, n_train)
        idx = np.argpartition(D, self.k, axis=1)[:, :self.k]   # k smallest
        neigh_y = self.y[idx]
        neigh_d = np.take_along_axis(D, idx, axis=1)
        if self.weights == "distance":
            w = 1.0 / (neigh_d + 1e-12)
        else:
            w = np.ones_like(neigh_d)

        out = []
        for i in range(len(X)):
            if self.task == "classification":
                classes = np.unique(neigh_y[i])
                scores = {c: w[i][neigh_y[i] == c].sum() for c in classes}
                out.append(max(scores, key=scores.get))
            else:  # regression: weighted average
                out.append(np.average(neigh_y[i], weights=w[i]))
        return np.array(out)


# ---------------------------------------------------------------------------
# 2. A simple KD-tree (teaching version, low-dim) + PyTorch batched distances
# ---------------------------------------------------------------------------
class KDTree:
    """Minimal KD-tree for exact nearest-neighbour search in low dimensions."""

    class _Node:
        __slots__ = ("point", "idx", "axis", "left", "right")

    def __init__(self, X):
        self.X = np.asarray(X, float)
        self.root = self._build(np.arange(len(X)), depth=0)

    def _build(self, idx, depth):
        if len(idx) == 0:
            return None
        axis = depth % self.X.shape[1]
        order = idx[np.argsort(self.X[idx, axis])]
        mid = len(order) // 2
        node = self._Node()
        node.idx = order[mid]; node.point = self.X[order[mid]]; node.axis = axis
        node.left = self._build(order[:mid], depth + 1)
        node.right = self._build(order[mid + 1:], depth + 1)
        return node

    def query(self, q, k=1):
        import heapq
        heap = []  # max-heap of (-dist, idx)

        def visit(node):
            if node is None:
                return
            d = np.linalg.norm(q - node.point)
            if len(heap) < k:
                heapq.heappush(heap, (-d, int(node.idx)))
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, int(node.idx)))
            diff = q[node.axis] - node.point[node.axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(near)
            if len(heap) < k or abs(diff) < -heap[0][0]:   # hypersphere crosses plane
                visit(far)

        visit(self.root)
        return sorted([(-nd, i) for nd, i in heap])


import torch


def knn_torch(X_train, y_train, X_query, k=5, task="classification"):
    """Vectorized kNN with torch.cdist (works on GPU)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y_train, device=dev)
    Xq = torch.as_tensor(X_query, dtype=torch.float32, device=dev)
    D = torch.cdist(Xq, Xt)                       # (q, n)
    idx = D.topk(k, largest=False).indices        # k nearest
    neigh = yt[idx]
    if task == "classification":
        out = torch.mode(neigh, dim=1).values
    else:
        out = neigh.float().mean(1)
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=300, noise=0.25, random_state=SEED)
    mu, sd = X.mean(0), X.std(0); X = (X - mu) / sd
    n_tr = 220
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    for k in (1, 5, 15):
        m = KNNNumPy(k=k).fit(Xtr, ytr)
        acc = np.mean(m.predict(Xte) == yte)
        print(f"k={k:2d}  test acc={acc:.3f}")

    acc_w = np.mean(KNNNumPy(k=15, weights="distance").fit(Xtr, ytr).predict(Xte) == yte)
    print(f"k=15 distance-weighted acc={acc_w:.3f}")

    acc_t = np.mean(knn_torch(Xtr, ytr, Xte, k=5) == yte)
    print(f"torch kNN (k=5) acc={acc_t:.3f}")

    # KD-tree vs brute force agree on nearest neighbour
    tree = KDTree(Xtr)
    q = Xte[0]
    kd = [i for _, i in tree.query(q, k=3)]
    bf = np.argsort(_pairwise(q[None], Xtr)[0])[:3].tolist()
    print(f"KD-tree NN {kd}  ==  brute force {bf}  -> {set(kd) == set(bf)}")


if __name__ == "__main__":
    demo()

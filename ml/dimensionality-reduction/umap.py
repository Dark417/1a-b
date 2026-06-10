"""
UMAP — Uniform Manifold Approximation and Projection (simplified)
=================================================================
A nonlinear manifold-learning / visualization method. Like t-SNE it preserves
local neighborhood structure, but it is built on a different mathematical story:
it models the data as a **fuzzy simplicial set** (a weighted k-nearest-neighbor
graph whose edge weights are *membership strengths*), and then lays out a
low-dimensional graph whose fuzzy structure matches the high-dimensional one,
minimizing a fuzzy-set cross-entropy via attractive/repulsive forces.

This is a faithful but *simplified* re-implementation: we build the fuzzy
kNN graph exactly as UMAP does (local connectivity `rho` + bandwidth `sigma`
calibrated so each row sums to log2(k)), and optimize the embedding with the
attractive/repulsive SGD using the smooth low-dim membership curve
`1 / (1 + a * d^{2b})` and negative sampling.

Variants implemented here:
    - NumPy fuzzy-simplicial-set construction (rho/sigma calibration, symmetrize)
    - NumPy attractive/repulsive SGD layout with negative sampling
    - PyTorch autograd layout optimizing the fuzzy cross-entropy directly

Training techniques demonstrated:
    - Negative sampling (repulsive forces on random non-neighbors)
    - Smooth approximation of the membership curve via fitted (a, b)
    - Learning-rate decay over epochs

References:
    - McInnes, Healy & Melville (2018), "UMAP: Uniform Manifold Approximation
      and Projection for Dimension Reduction" (arXiv:1802.03426)
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _pairwise_sq_dists(X):
    """||x_i - x_j||^2 for all pairs."""
    sq = (X ** 2).sum(1)
    D = sq[:, None] - 2 * X @ X.T + sq[None, :]
    return np.maximum(D, 0)


def _fit_ab(min_dist=0.1, spread=1.0):
    r"""
    UMAP's low-dim membership is approximated by the smooth curve
        phi(d) = 1 / (1 + a * d^{2b}).
    We fit (a, b) by least squares so phi matches the target piecewise curve
        psi(d) = 1                       if d <= min_dist
                 exp(-(d - min_dist)/spread) otherwise.
    """
    from scipy.optimize import curve_fit

    xv = np.linspace(0, spread * 3, 300)
    yv = np.where(xv <= min_dist, 1.0, np.exp(-(xv - min_dist) / spread))

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    (a, b), _ = curve_fit(curve, xv, yv, p0=(1.0, 1.0), maxfev=10000)
    return float(a), float(b)


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
class UMAPNumPy:
    r"""
    Step 1 — Fuzzy simplicial set (high-dim graph).
      For each point i, let rho_i be the distance to its nearest neighbor
      (local connectivity), and find sigma_i (a bandwidth) by binary search so
      that
          sum_{j in kNN(i)} exp(-(d_ij - rho_i)_+ / sigma_i) = log2(k).
      The directed membership strength is
          w_{j|i} = exp(-(d_ij - rho_i)_+ / sigma_i).
      Symmetrize with the probabilistic t-conorm (fuzzy union):
          w_{ij} = w_{j|i} + w_{i|j} - w_{j|i} w_{i|j}.

    Step 2 — Low-dim layout.
      Low-dim membership of an edge of length d is phi(d) = 1/(1 + a d^{2b}).
      Minimize the fuzzy cross-entropy between high-dim weights w_ij and
      low-dim memberships, which decomposes into an *attractive* term on graph
      edges and a *repulsive* term on (sampled) non-edges. We follow the
      gradients with SGD + negative sampling.
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0,
                 n_epochs=200, lr=1.0, n_negative=5, seed=SEED):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.lr = lr
        self.n_negative = n_negative
        self.seed = seed

    # --- fuzzy simplicial set -------------------------------------------
    def _fuzzy_graph(self, X):
        n = len(X)
        k = min(self.n_neighbors, n - 1)
        D = np.sqrt(_pairwise_sq_dists(X))
        # k nearest neighbors (exclude self at index 0 after sorting)
        knn_idx = np.argsort(D, axis=1)[:, 1:k + 1]
        knn_d = np.take_along_axis(D, knn_idx, axis=1)

        target = np.log2(k)
        W = np.zeros((n, n))
        for i in range(n):
            di = knn_d[i]
            rho = di[di > 0].min() if np.any(di > 0) else 0.0   # local connectivity
            # binary search sigma so the row sums to log2(k)
            lo, hi, sigma = 0.0, np.inf, 1.0
            for _ in range(64):
                psum = np.exp(-np.maximum(di - rho, 0) / sigma).sum()
                if abs(psum - target) < 1e-5:
                    break
                if psum > target:
                    hi = sigma
                    sigma = (lo + hi) / 2
                else:
                    lo = sigma
                    sigma = sigma * 2 if hi == np.inf else (lo + hi) / 2
            w = np.exp(-np.maximum(di - rho, 0) / sigma)
            W[i, knn_idx[i]] = w
        # symmetrize via probabilistic t-conorm (fuzzy union)
        W = W + W.T - W * W.T
        return W

    def fit_transform(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.a, self.b = _fit_ab(self.min_dist, self.spread)

        W = self._fuzzy_graph(X)
        # edge list with weights (upper triangle is enough; graph is symmetric)
        ii, jj = np.where(W > 1e-3)
        mask = ii < jj
        ii, jj = ii[mask], jj[mask]
        wij = W[ii, jj]

        # spectral-ish init: a small random embedding (cheap, deterministic)
        Y = rng.normal(0, 10.0, (n, self.n_components))

        a, b = self.a, self.b
        n_edges = len(ii)
        for epoch in range(self.n_epochs):
            alpha = self.lr * (1.0 - epoch / self.n_epochs)   # LR decay
            # process edges in random order
            order = rng.permutation(n_edges)
            for e in order:
                i, j = ii[e], jj[e]
                # --- attractive force on the connected edge (i, j) ---
                diff = Y[i] - Y[j]
                d2 = (diff ** 2).sum() + 1e-12
                # d/dd2 of cross-entropy attractive term:
                #   grad_coef = -2 a b d2^{b-1} / (1 + a d2^b)
                grad_coef = (-2.0 * a * b * d2 ** (b - 1.0)) / (1.0 + a * d2 ** b)
                grad = np.clip(grad_coef * diff, -4, 4) * wij[e]
                Y[i] += alpha * grad
                Y[j] -= alpha * grad
                # --- repulsive forces against negative samples ---
                for _ in range(self.n_negative):
                    kk = rng.integers(n)
                    if kk == i:
                        continue
                    diff = Y[i] - Y[kk]
                    d2 = (diff ** 2).sum() + 1e-12
                    # repulsive gradient coefficient
                    grad_coef = (2.0 * b) / ((1e-3 + d2) * (1.0 + a * d2 ** b))
                    grad = np.clip(grad_coef * diff, -4, 4)
                    Y[i] += alpha * grad
        self.embedding_ = Y
        self.graph_ = W
        return Y


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (autograd on the fuzzy cross-entropy)
# ---------------------------------------------------------------------------
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def umap_torch(X, n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0,
               n_iter=300, lr=1e-1, seed=SEED):
    r"""
    Optimize the fuzzy-set cross-entropy directly with autograd.

    With high-dim memberships w_ij (from the fuzzy graph) and low-dim
    memberships q_ij = 1/(1 + a ||y_i - y_j||^{2b}), the UMAP objective is the
    fuzzy cross-entropy
        CE = - sum_ij [ w_ij log q_ij + (1 - w_ij) log(1 - q_ij) ].
    We minimize a subsampled version (all edges attract, all pairs repel) with
    Adam — autograd handles the messy gradient.
    """
    dev = get_device()
    X = np.asarray(X, float)
    n = len(X)
    a, b = _fit_ab(min_dist, spread)

    builder = UMAPNumPy(n_components, n_neighbors, min_dist, spread, seed=seed)
    W = builder._fuzzy_graph(X)
    Wt = torch.as_tensor(W, dtype=torch.float32, device=dev)
    eye = torch.eye(n, device=dev, dtype=torch.bool)

    g = torch.Generator(device="cpu").manual_seed(seed)
    Y = (torch.randn(n, n_components, generator=g) * 10.0).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([Y], lr=lr)

    for _ in range(n_iter):
        d2 = torch.cdist(Y, Y) ** 2
        q = 1.0 / (1.0 + a * d2.clamp_min(1e-9) ** b)
        q = q.masked_fill(eye, 0.0).clamp(1e-6, 1 - 1e-6)
        ce = -(Wt * q.log() + (1.0 - Wt) * (1.0 - q).log())
        ce = ce.masked_fill(eye, 0.0)
        loss = ce.sum() / (n * n)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return Y.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    from sklearn.datasets import load_digits
    from sklearn.manifold import trustworthiness

    digits = load_digits()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(digits.data), 300, replace=False)
    X, y = digits.data[idx], digits.target[idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    um = UMAPNumPy(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=120)
    Y = um.fit_transform(X)
    print(f"fitted membership curve: a={um.a:.3f}  b={um.b:.3f}")
    tw = trustworthiness(X, Y, n_neighbors=5)
    print(f"NumPy UMAP trustworthiness={tw:.3f}")

    # cluster structure: inter/intra class distance ratio in the map
    from itertools import combinations
    cen = np.stack([Y[y == c].mean(0) for c in np.unique(y)])
    intra = np.mean([np.linalg.norm(Y[i] - cen[y[i]]) for i in range(len(Y))])
    inter = np.mean([np.linalg.norm(cen[p] - cen[q]) for p, q in combinations(range(len(cen)), 2)])
    print(f"map intra={intra:.2f} inter={inter:.2f} ratio={inter/(intra+1e-9):.2f}")

    Yt = umap_torch(X, n_neighbors=15, n_iter=200)
    twt = trustworthiness(X, Yt, n_neighbors=5)
    print(f"Torch UMAP trustworthiness={twt:.3f}")


if __name__ == "__main__":
    demo()

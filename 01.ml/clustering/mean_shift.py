"""
Mean Shift — mode-seeking on a kernel density estimate
======================================================
Mean shift is a non-parametric clustering method: it places a kernel density
estimate (KDE) over the data, then moves every point *uphill* along the density
gradient until it reaches a mode (local maximum). Points that climb to the same
mode form a cluster. The number of clusters is not specified — it emerges from
the bandwidth `h` (the only real hyperparameter).

The update each iteration replaces a point by the (kernel-weighted) mean of its
neighbors — this is exactly a normalized gradient-ascent step on the KDE.

Variants implemented here:
    - Flat (uniform) kernel: mean of points within radius h
    - Gaussian (RBF) kernel: weighted mean with weight exp(-||x-y||^2 / 2h^2)
    - NumPy from-scratch + a PyTorch (torch.cdist) vectorized version

Training techniques demonstrated:
    - The mean-shift vector as a gradient-ascent step on a KDE (derivation in nb)
    - Bandwidth selection and mode-merging tolerance
    - Seeding from all points vs a subset (binning) for speed

References:
    - Fukunaga & Hostetler (1975); Cheng (1995)
    - Comaniciu & Meer (2002), "Mean Shift: A Robust Approach Toward Feature Space Analysis"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
class MeanShiftNumPy:
    r"""
    KDE with kernel profile k:   f(x) = (c / n h^d) sum_i k(||(x - x_i)/h||^2).
    Its gradient points toward higher density; setting it proportional to the
    *mean shift vector*

        m(x) = [ sum_i x_i g(||(x-x_i)/h||^2) / sum_i g(...) ] - x,

    where g = -k'. The fixed-point iteration x <- x + m(x) climbs to a mode.

      - Gaussian kernel:  k(t) = exp(-t/2)  -> g(t) = exp(-t/2) (weights = RBF).
      - Flat kernel:      k(t) = 1[t<=1]    -> g is the indicator; m(x) = mean of
                          neighbors within radius h, minus x.
    """

    def __init__(self, bandwidth=1.0, kernel="gaussian", max_iter=300,
                 tol=1e-4, cluster_eps=None):
        self.bandwidth, self.kernel = bandwidth, kernel
        self.max_iter, self.tol = max_iter, tol
        # modes closer than cluster_eps are merged into one cluster
        self.cluster_eps = cluster_eps if cluster_eps is not None else bandwidth / 2

    def _shift(self, y, X):
        # one mean-shift step for a single point y given all data X
        d2 = ((X - y) ** 2).sum(1)                       # ||x_i - y||^2
        h2 = self.bandwidth ** 2
        if self.kernel == "flat":
            w = (d2 <= h2).astype(float)                 # uniform within radius
        else:                                            # gaussian
            w = np.exp(-0.5 * d2 / h2)
        s = w.sum()
        if s == 0:
            return y
        return (w[:, None] * X).sum(0) / s               # weighted mean (the new y)

    def fit(self, X):
        X = np.asarray(X, float)
        modes = X.copy()                                 # seed a trajectory per point
        for i in range(len(X)):
            y = modes[i]
            for _ in range(self.max_iter):
                y_new = self._shift(y, X)
                if np.linalg.norm(y_new - y) < self.tol:
                    y = y_new; break
                y = y_new
            modes[i] = y

        # merge nearby modes into cluster centers (greedy within cluster_eps)
        centers = []
        labels = np.full(len(X), -1)
        for i, m in enumerate(modes):
            for c, center in enumerate(centers):
                if np.linalg.norm(m - center) < self.cluster_eps:
                    labels[i] = c
                    break
            else:
                labels[i] = len(centers)
                centers.append(m)
        self.cluster_centers_ = np.array(centers)
        self.labels_ = labels
        self.n_clusters_ = len(centers)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        d2 = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(2)
        return d2.argmin(1)


def estimate_bandwidth(X, quantile=0.3, n_samples=100, seed=SEED):
    """Median-style bandwidth heuristic: average of the `quantile` nearest distances."""
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    d2 = ((X[idx][:, None, :] - X[None, :, :]) ** 2).sum(2)
    d = np.sqrt(np.maximum(d2, 0))
    d.sort(axis=1)
    k = max(1, int(quantile * len(X)))
    return d[:, :k].max(1).mean()


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (GPU-friendly, vectorized over all seeds)
# ---------------------------------------------------------------------------
import torch


def get_device():
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MeanShiftTorch:
    """All points shifted in parallel; torch.cdist gives the pairwise distances."""

    def __init__(self, bandwidth=1.0, kernel="gaussian", max_iter=300,
                 tol=1e-4, cluster_eps=None, device=None):
        self.bandwidth, self.kernel = bandwidth, kernel
        self.max_iter, self.tol = max_iter, tol
        self.cluster_eps = cluster_eps if cluster_eps is not None else bandwidth / 2
        self.device = device or get_device()

    def fit(self, X):
        Xt = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        Y = Xt.clone()                                   # one seed per data point
        h2 = self.bandwidth ** 2
        for _ in range(self.max_iter):
            D2 = torch.cdist(Y, Xt) ** 2                 # (n, n) squared distances
            if self.kernel == "flat":
                W = (D2 <= h2).float()
            else:
                W = torch.exp(-0.5 * D2 / h2)
            Wsum = W.sum(1, keepdim=True).clamp_min(1e-12)
            Y_new = (W @ Xt) / Wsum                      # weighted means (all at once)
            if (Y_new - Y).norm(dim=1).max() < self.tol:
                Y = Y_new; break
            Y = Y_new

        modes = Y.cpu().numpy()
        centers, labels = [], np.full(len(modes), -1)
        for i, m in enumerate(modes):
            for c, center in enumerate(centers):
                if np.linalg.norm(m - center) < self.cluster_eps:
                    labels[i] = c; break
            else:
                labels[i] = len(centers); centers.append(m)
        self.cluster_centers_ = np.array(centers)
        self.labels_ = labels
        self.n_clusters_ = len(centers)
        return self


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    X, ytrue = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=SEED)

    # the flat kernel uses h as a hard radius; the Gaussian kernel uses it as a
    # std, so it needs a smaller h to resolve the same clusters.
    h_flat = estimate_bandwidth(X, quantile=0.3)
    h_gauss = h_flat / 2.5
    print(f"estimated bandwidth: flat h={h_flat:.3f}  gaussian h={h_gauss:.3f}")

    bw = {"gaussian": h_gauss, "flat": h_flat}
    for kern in ("gaussian", "flat"):
        ms = MeanShiftNumPy(bandwidth=bw[kern], kernel=kern).fit(X)
        print(f"NumPy mean-shift ({kern:8s}): clusters={ms.n_clusters_}  "
              f"ARI={adjusted_rand_score(ytrue, ms.labels_):.3f}")

    mt = MeanShiftTorch(bandwidth=h_gauss, kernel="gaussian").fit(X)
    print(f"Torch mean-shift (gaussian): clusters={mt.n_clusters_}  "
          f"ARI={adjusted_rand_score(ytrue, mt.labels_):.3f}  device={mt.device}")

    # bandwidth controls cluster count: small h -> many modes, large h -> few
    print("\nBandwidth vs #clusters (gaussian):")
    for hh in (0.3, 0.5, h_gauss, 1.5, 3.0):
        n = MeanShiftNumPy(bandwidth=hh, kernel="gaussian").fit(X).n_clusters_
        print(f"  h={hh:4.2f}: {n} clusters")


if __name__ == "__main__":
    demo()

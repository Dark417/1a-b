"""
DBSCAN — Density-Based Spatial Clustering of Applications with Noise
====================================================================
Cluster points by *density* rather than by distance to a centroid: a cluster is
a maximal set of points connected through dense neighborhoods. DBSCAN needs no
preset number of clusters, finds arbitrarily shaped clusters (moons, rings), and
labels low-density points as **noise** (-1). Two hyperparameters: a radius `eps`
and a minimum neighbor count `min_samples`.

Point types:
    - core point:   has >= min_samples points within eps (incl. itself)
    - border point: within eps of a core point but not itself core
    - noise point:  neither (label -1)

Variants implemented here:
    - DBSCAN (NumPy, from scratch via BFS over the eps-neighborhood graph)
    - DBSCAN (PyTorch, torch.cdist neighbor matrix, GPU-friendly)
    - OPTICS-style reachability ordering (conceptual cousin that avoids a single
      global eps by producing a reachability plot you can cut at multiple scales)

Training techniques demonstrated:
    - k-distance plot heuristic for choosing eps (the "elbow")
    - Density-reachability / density-connectivity as the cluster definition

References:
    - Ester, Kriegel, Sander & Xu (1996), "A Density-Based Algorithm ... (DBSCAN)"
    - Ankerst, Breunig, Kriegel & Sander (1999), "OPTICS"
"""

from __future__ import annotations

import numpy as np

SEED = 0

NOISE = -1
UNVISITED = -2


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
class DBSCANNumPy:
    r"""
    Definitions (metric d, radius eps, threshold m = min_samples):
      - eps-neighborhood:  N_eps(p) = { q : d(p,q) <= eps }.
      - p is a *core* point iff |N_eps(p)| >= m.
      - q is *directly density-reachable* from core p iff q in N_eps(p).
      - *density-reachable*: transitive closure of directly-reachable via cores.
      - *density-connected*: p, q both density-reachable from some core o.
    A cluster is a maximal density-connected set. DBSCAN realizes this by, for
    every unvisited core point, growing its cluster with a BFS that only expands
    through *core* points (border points join but do not seed further growth).
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps, self.min_samples = eps, min_samples

    def _neighbors(self, X):
        # pairwise squared distances -> boolean adjacency within eps
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(2)
        return [np.where(row <= self.eps ** 2)[0] for row in d2]

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        neigh = self._neighbors(X)
        n_neigh = np.array([len(a) for a in neigh])
        is_core = n_neigh >= self.min_samples          # core-point mask

        labels = np.full(n, UNVISITED)
        cluster = 0
        for p in range(n):
            if labels[p] != UNVISITED:                 # already processed
                continue
            if not is_core[p]:
                labels[p] = NOISE                      # tentatively noise
                continue
            # --- BFS to grow a new cluster from core seed p ---
            labels[p] = cluster
            queue = list(neigh[p])
            qi = 0
            while qi < len(queue):
                q = queue[qi]; qi += 1
                if labels[q] == NOISE:
                    labels[q] = cluster                # border reclaimed from noise
                if labels[q] != UNVISITED:
                    continue
                labels[q] = cluster
                if is_core[q]:                         # only cores expand the frontier
                    queue.extend(neigh[q])
            cluster += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        self.n_clusters_ = cluster
        # classify point types for teaching/inspection
        self.point_types_ = np.where(
            is_core, "core",
            np.where(labels == NOISE, "noise", "border"))
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


def k_distance(X, k=4):
    """Sorted distance to the k-th nearest neighbor — the eps-selection heuristic.

    Plot the returned (descending) curve; the 'elbow' is a good eps: most points
    have a small k-distance (inside clusters), noise points have a large one.
    """
    X = np.asarray(X, float)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(2)
    d = np.sqrt(np.maximum(d2, 0))
    d.sort(axis=1)                                     # ascending per row
    kth = d[:, k]                                      # 0 is the point itself
    return np.sort(kth)[::-1]


# ---------------------------------------------------------------------------
# OPTICS-style reachability ordering (conceptual cousin of DBSCAN)
# ---------------------------------------------------------------------------
def optics_reachability(X, min_samples=5, max_eps=np.inf):
    r"""
    OPTICS orders points so that spatially close points are neighbors in the
    ordering, recording each point's *reachability distance*. The reachability
    plot (valleys = clusters) lets you extract DBSCAN clusterings for *every*
    eps at once, sidestepping a single global eps.

    core-distance(p)        = distance to the min_samples-th nearest neighbor
    reachability-dist(o,p)  = max(core-distance(p), d(p, o))
    """
    X = np.asarray(X, float)
    n = len(X)
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(2), 0))
    Dsorted = np.sort(D, axis=1)
    core_dist = Dsorted[:, min_samples - 1]            # min_samples-th neighbor incl self
    core_dist[Dsorted[:, min_samples - 1] > max_eps] = np.inf

    processed = np.zeros(n, bool)
    reach = np.full(n, np.inf)
    order = []
    for start in range(n):
        if processed[start]:
            continue
        seeds = [start]
        while seeds:
            # pick the unprocessed seed with smallest reachability
            seeds.sort(key=lambda i: reach[i])
            p = seeds.pop(0)
            if processed[p]:
                continue
            processed[p] = True
            order.append(p)
            if np.isinf(core_dist[p]):
                continue
            for o in range(n):                         # update neighbors within max_eps
                if processed[o] or D[p, o] > max_eps:
                    continue
                new_reach = max(core_dist[p], D[p, o])
                if new_reach < reach[o]:
                    reach[o] = new_reach
                    if o not in seeds:
                        seeds.append(o)
    order = np.array(order)
    return order, reach[order]


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (GPU-friendly neighbor computation)
# ---------------------------------------------------------------------------
import torch


def get_device():
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DBSCANTorch:
    """DBSCAN whose heavy step — the eps-neighborhood graph — uses torch.cdist."""

    def __init__(self, eps=0.5, min_samples=5, device=None):
        self.eps, self.min_samples = eps, min_samples
        self.device = device or get_device()

    def fit(self, X):
        Xt = torch.as_tensor(np.asarray(X, np.float32), device=self.device)
        D = torch.cdist(Xt, Xt)                        # (n, n) pairwise distances
        adj = D <= self.eps                            # boolean neighborhood graph
        n_neigh = adj.sum(1)
        is_core = (n_neigh >= self.min_samples).cpu().numpy()
        adj = adj.cpu().numpy()
        n = len(Xt)

        labels = np.full(n, UNVISITED)
        cluster = 0
        for p in range(n):
            if labels[p] != UNVISITED:
                continue
            if not is_core[p]:
                labels[p] = NOISE
                continue
            labels[p] = cluster
            queue = list(np.where(adj[p])[0]); qi = 0
            while qi < len(queue):
                q = queue[qi]; qi += 1
                if labels[q] == NOISE:
                    labels[q] = cluster
                if labels[q] != UNVISITED:
                    continue
                labels[q] = cluster
                if is_core[q]:
                    queue.extend(np.where(adj[q])[0])
            cluster += 1
        self.labels_ = labels
        self.n_clusters_ = cluster
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.metrics import adjusted_rand_score

    # two interleaving moons: density-based clustering separates them, k-means can't
    X, ytrue = make_moons(n_samples=400, noise=0.06, random_state=SEED)

    db = DBSCANNumPy(eps=0.2, min_samples=5).fit(X)
    ari = adjusted_rand_score(ytrue, db.labels_)
    n_noise = int((db.labels_ == NOISE).sum())
    print(f"NumPy DBSCAN moons: clusters={db.n_clusters_}  noise={n_noise}  "
          f"ARI={ari:.3f}  cores={len(db.core_sample_indices_)}")

    dt = DBSCANTorch(eps=0.2, min_samples=5).fit(X)
    print(f"Torch DBSCAN moons: clusters={dt.n_clusters_}  "
          f"ARI={adjusted_rand_score(ytrue, dt.labels_):.3f}  device={dt.device}")

    # blobs with background noise: DBSCAN flags outliers as -1
    Xb, yb = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=SEED)
    noise = np.random.uniform(Xb.min(0), Xb.max(0), size=(40, 2))
    Xn = np.vstack([Xb, noise])
    dbn = DBSCANNumPy(eps=0.6, min_samples=6).fit(Xn)
    print(f"\nNumPy DBSCAN blobs+noise: clusters={dbn.n_clusters_}  "
          f"flagged_noise={int((dbn.labels_ == NOISE).sum())} (injected 40)")

    # eps heuristic and OPTICS ordering
    kd = k_distance(X, k=4)
    print(f"\nk-distance (k=4) range: [{kd.min():.3f}, {kd.max():.3f}] — elbow ~ good eps")
    order, reach = optics_reachability(X[:120], min_samples=5)
    finite = reach[np.isfinite(reach)]
    print(f"OPTICS reachability over {len(order)} pts: "
          f"mean={finite.mean():.3f} (valleys=clusters)")


if __name__ == "__main__":
    demo()

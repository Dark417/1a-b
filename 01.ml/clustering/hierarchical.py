"""
Agglomerative Hierarchical Clustering
=====================================
Build a hierarchy of clusters bottom-up: start with every point in its own
cluster, then repeatedly merge the two *closest* clusters until one remains. The
sequence of merges forms a tree (dendrogram); cutting it at a chosen height (or
target cluster count) yields a flat clustering at any granularity — no `k`
needed up front.

The notion of "closest" is the **linkage** criterion:
    - single   : min pairwise distance  (chains; finds non-convex shapes)
    - complete : max pairwise distance  (compact, similar-diameter clusters)
    - average  : mean pairwise distance (UPGMA; a compromise)
    - ward     : merge that least increases within-cluster variance (spherical)

Variants implemented here:
    - All four linkages via a unified Lance-Williams update (NumPy, from scratch)
    - A PyTorch (torch.cdist) variant for the initial distance matrix + merges
    - A `linkage_` matrix in SciPy format so the notebook can draw a dendrogram

Training techniques demonstrated:
    - Lance-Williams recurrence: update cluster distances in O(1) per merge
    - Choosing the cut height / number of clusters from the merge distances

References:
    - Lance & Williams (1967), "A general theory of classificatory sorting strategies"
    - Ward (1963), "Hierarchical Grouping to Optimize an Objective Function"
    - Murtagh & Contreras (2012), survey of agglomerative methods
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch, Lance-Williams update)
# ---------------------------------------------------------------------------
class AgglomerativeNumPy:
    r"""
    Lance-Williams recurrence. When clusters i and j merge into (ij), the
    distance from the new cluster to any other cluster k is

        d(ij, k) = a_i d(i,k) + a_j d(j,k) + b d(i,j) + g |d(i,k) - d(j,k)|,

    with coefficients depending on the linkage (n_x = cluster sizes):

      single:   a_i=a_j=1/2, b=0,  g=-1/2     -> d = min(d_ik, d_jk)
      complete: a_i=a_j=1/2, b=0,  g=+1/2     -> d = max(d_ik, d_jk)
      average:  a_i=n_i/(n_i+n_j), a_j=n_j/(n_i+n_j), b=0, g=0
      ward:     a_i=(n_i+n_k)/T, a_j=(n_j+n_k)/T, b=-n_k/T, g=0, T=n_i+n_j+n_k
                (operates on *squared* Euclidean distances)
    """

    def __init__(self, n_clusters=2, linkage="ward"):
        assert linkage in ("single", "complete", "average", "ward")
        self.n_clusters, self.linkage = n_clusters, linkage

    def _lw_update(self, d_ik, d_jk, d_ij, ni, nj, nk):
        L = self.linkage
        if L == "single":
            return np.minimum(d_ik, d_jk)
        if L == "complete":
            return np.maximum(d_ik, d_jk)
        if L == "average":
            return (ni * d_ik + nj * d_jk) / (ni + nj)
        # ward (on squared distances)
        T = ni + nj + nk
        return ((ni + nk) * d_ik + (nj + nk) * d_jk - nk * d_ij) / T

    def fit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        # initial pairwise distances; Ward works on squared Euclidean distances
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(2)
        D = d2 if self.linkage == "ward" else np.sqrt(np.maximum(d2, 0))
        np.fill_diagonal(D, np.inf)

        sizes = np.ones(n)
        active = list(range(n))
        # SciPy-style linkage rows: [idx_a, idx_b, distance, new_size]
        cluster_id = list(range(n))      # current external id of each active slot
        next_id = n
        linkage_rows = []

        while len(active) > 1:
            # find the closest pair among active clusters
            best = np.inf; bi = bj = -1
            for a in range(len(active)):
                for b in range(a + 1, len(active)):
                    ia, ib = active[a], active[b]
                    if D[ia, ib] < best:
                        best, bi, bj = D[ia, ib], a, b
            ia, ib = active[bi], active[bj]
            ni, nj = sizes[ia], sizes[ib]
            # record merge (report Euclidean distance even for Ward's squared metric)
            dist = np.sqrt(best) if self.linkage in ("ward",) else best
            linkage_rows.append([cluster_id[ia], cluster_id[ib], dist, ni + nj])

            # update distances from the merged cluster (store into slot ia)
            for ic in active:
                if ic == ia or ic == ib:
                    continue
                D[ia, ic] = D[ic, ia] = self._lw_update(
                    D[ia, ic], D[ib, ic], D[ia, ib], ni, nj, sizes[ic])
            sizes[ia] = ni + nj
            cluster_id[ia] = next_id; next_id += 1
            active.remove(ib)                     # ib absorbed into ia

        self.linkage_ = np.array(linkage_rows, float)
        self.labels_ = self._cut(n, self.n_clusters)
        return self

    def _cut(self, n, k):
        """Cut the dendrogram into k flat clusters (undo the last k-1 merges)."""
        # parent map from the merge order
        parent = {}
        next_id = n
        for a, b, _, _ in self.linkage_:
            parent[int(a)] = next_id
            parent[int(b)] = next_id
            next_id += 1
        # the first (n - k) merges are "kept"; later merges are cut
        n_merges = len(self.linkage_) - (k - 1)
        n_merges = max(0, n_merges)
        keep = {}
        nid = n
        for m, (a, b, _, _) in enumerate(self.linkage_):
            if m < n_merges:
                keep[int(a)] = nid; keep[int(b)] = nid
            nid += 1

        # union-find over kept merges to assign root labels
        def root(x):
            while x in keep:
                x = keep[x]
            return x

        roots = [root(i) for i in range(n)]
        uniq = {r: c for c, r in enumerate(sorted(set(roots)))}
        return np.array([uniq[r] for r in roots])

    def fit_predict(self, X):
        return self.fit(X).labels_


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (GPU-friendly distance matrix; same LW merges)
# ---------------------------------------------------------------------------
import torch


def get_device():
    """Pick the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def agglomerative_torch(X, n_clusters=2, linkage="ward"):
    """Compute the initial distance matrix with torch.cdist, then merge.

    The merge loop is inherently sequential; we keep the heavy O(n^2 d) distance
    computation on the (GPU) tensor and hand a NumPy matrix to the proven NumPy
    merger so both paths give identical trees.
    """
    dev = get_device()
    Xt = torch.as_tensor(np.asarray(X, np.float32), device=dev)
    D = torch.cdist(Xt, Xt)                      # Euclidean distances on device
    if linkage == "ward":
        D = D ** 2                               # Ward operates on squared distances
    model = AgglomerativeNumPy(n_clusters=n_clusters, linkage=linkage)
    # inject the precomputed matrix by running the same merge logic
    Dn = D.cpu().numpy().astype(float)
    np.fill_diagonal(Dn, np.inf)
    model._merge_from_matrix(Dn, len(Xt))
    return model.labels_, model.linkage_


def _merge_from_matrix(self, D, n):
    """Run the Lance-Williams merge loop on a precomputed distance matrix D."""
    sizes = np.ones(n)
    active = list(range(n))
    cluster_id = list(range(n))
    next_id = n
    linkage_rows = []
    while len(active) > 1:
        best = np.inf; bi = bj = -1
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                ia, ib = active[a], active[b]
                if D[ia, ib] < best:
                    best, bi, bj = D[ia, ib], a, b
        ia, ib = active[bi], active[bj]
        ni, nj = sizes[ia], sizes[ib]
        dist = np.sqrt(best) if self.linkage == "ward" else best
        linkage_rows.append([cluster_id[ia], cluster_id[ib], dist, ni + nj])
        for ic in active:
            if ic == ia or ic == ib:
                continue
            D[ia, ic] = D[ic, ia] = self._lw_update(
                D[ia, ic], D[ib, ic], D[ia, ib], ni, nj, sizes[ic])
        sizes[ia] = ni + nj
        cluster_id[ia] = next_id; next_id += 1
        active.remove(ib)
    self.linkage_ = np.array(linkage_rows, float)
    self.labels_ = self._cut(n, self.n_clusters)
    return self


AgglomerativeNumPy._merge_from_matrix = _merge_from_matrix


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.metrics import adjusted_rand_score

    X, ytrue = make_blobs(n_samples=200, centers=3, cluster_std=0.7, random_state=SEED)

    print("Blobs (k=3) — ARI by linkage:")
    for L in ("single", "complete", "average", "ward"):
        m = AgglomerativeNumPy(n_clusters=3, linkage=L).fit(X)
        print(f"  {L:8s}: ARI={adjusted_rand_score(ytrue, m.labels_):.3f}  "
              f"merges={len(m.linkage_)}")

    # single linkage shines on non-convex shapes (moons) where Ward fails
    Xm, ym = make_moons(n_samples=200, noise=0.05, random_state=SEED)
    s = AgglomerativeNumPy(n_clusters=2, linkage="single").fit(Xm)
    w = AgglomerativeNumPy(n_clusters=2, linkage="ward").fit(Xm)
    print(f"\nMoons: single ARI={adjusted_rand_score(ym, s.labels_):.3f}  "
          f"ward ARI={adjusted_rand_score(ym, w.labels_):.3f}")

    lab, link = agglomerative_torch(X, n_clusters=3, linkage="ward")
    print(f"\nTorch (ward) ARI={adjusted_rand_score(ytrue, lab):.3f}  "
          f"linkage rows={len(link)}  device={get_device()}")

    print(f"\nLast 3 merge distances (ward): "
          f"{np.round(AgglomerativeNumPy(3, 'ward').fit(X).linkage_[-3:, 2], 2)}  "
          f"(big jump => natural #clusters)")


if __name__ == "__main__":
    demo()

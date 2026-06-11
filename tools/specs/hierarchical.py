from tools.nbreg import register, md, code, show, run_demo

MOD = "hierarchical"


@register("hierarchical", "01.ml/clustering/hierarchical.ipynb")
def build():
    return [
        md(r"""
# Agglomerative Hierarchical Clustering — a tree of merges

> Tutorial pair for [`hierarchical.py`](hierarchical.py).

## 1. Intuition
Start with every point as its own cluster. Repeatedly glue together the two
**closest** clusters. After $n-1$ merges everything is one cluster, and the
record of merges is a **dendrogram** — a tree you can slice at any height to get
a flat clustering at any granularity. You never have to commit to $k$ up front;
you read it off the tree (look for a big jump in merge distance).
"""),
        md(r"""
## 2. Concept (the slide)
The only design choice is what "distance between two *clusters*" means — the
**linkage**:
- **single** $=\min$ pairwise distance — chains points together, follows
  non-convex shapes, but suffers *chaining* (bridges between blobs).
- **complete** $=\max$ pairwise distance — compact, equal-diameter clusters.
- **average** (UPGMA) $=$ mean pairwise distance — a robust compromise.
- **Ward** — merge the pair that *least increases* total within-cluster variance;
  tends to give balanced, spherical clusters (the agglomerative analogue of
  k-means' objective).

All four share one efficient update rule: **Lance-Williams**.
"""),
        md(r"""
## 3. Math derivation

**Setup.** Clusters $C_1,\dots,C_t$; at each step merge the pair minimizing the
linkage distance $d(C_i,C_j)$, giving $t-1$ clusters, until one remains.

**Linkage criteria.** With pointwise distance $d(\cdot,\cdot)$,
$$
d_{\text{single}}(A,B)=\min_{a\in A,b\in B} d(a,b),\quad
d_{\text{complete}}(A,B)=\max_{a\in A,b\in B} d(a,b),
$$
$$
d_{\text{average}}(A,B)=\frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B} d(a,b).
$$
**Ward** uses the increase in error sum of squares (ESS). For a cluster
$C$ with centroid $\bar{\mathbf x}_C$, $\mathrm{ESS}(C)=\sum_{x\in C}\lVert x-\bar{\mathbf x}_C\rVert^2$.
Merging $A,B$ raises the total ESS by
$$\Delta(A,B)=\mathrm{ESS}(A\cup B)-\mathrm{ESS}(A)-\mathrm{ESS}(B)
=\frac{|A|\,|B|}{|A|+|B|}\,\lVert\bar{\mathbf x}_A-\bar{\mathbf x}_B\rVert^2,$$
and Ward merges the pair with the smallest $\Delta$.

**Lance-Williams recurrence.** Recomputing $d$ from scratch after each merge is
wasteful. When $C_i,C_j$ merge into $C_{ij}$, the distance to any other cluster
$C_k$ updates in $O(1)$:
$$
d(C_{ij},C_k)=\alpha_i\,d(C_i,C_k)+\alpha_j\,d(C_j,C_k)+\beta\,d(C_i,C_j)+\gamma\,\bigl|d(C_i,C_k)-d(C_j,C_k)\bigr|.
$$
With $n_x=|C_x|$ and $T=n_i+n_j+n_k$, the coefficients recover each linkage:

| linkage | $\alpha_i$ | $\alpha_j$ | $\beta$ | $\gamma$ |
|---|---|---|---|---|
| single   | $1/2$ | $1/2$ | $0$ | $-1/2$ |
| complete | $1/2$ | $1/2$ | $0$ | $+1/2$ |
| average  | $\frac{n_i}{n_i+n_j}$ | $\frac{n_j}{n_i+n_j}$ | $0$ | $0$ |
| Ward     | $\frac{n_i+n_k}{T}$ | $\frac{n_j+n_k}{T}$ | $-\frac{n_k}{T}$ | $0$ |

(Single/complete fall straight out: $\tfrac12 d_{ik}+\tfrac12 d_{jk}\mp\tfrac12|d_{ik}-d_{jk}|=\min/\max$.)
Ward operates on **squared** Euclidean distances, then merge heights are reported
as $\sqrt{\cdot}$.

**Complexity & cutting.** The naive loop is $O(n^3)$ ($O(n^2)$ merges, each an
$O(n)$ update plus an $O(n^2)$ argmin here for clarity; priority queues give
$O(n^2\log n)$). To get $k$ flat clusters, **cut** the dendrogram below its top
$k-1$ merges — i.e. keep the first $n-k$ merges and read off the connected
components.
"""),
        md("## 4. NumPy implementation (all linkages via Lance-Williams + dendrogram cut)"),
        show(MOD, "AgglomerativeNumPy"),
        md("## 5. PyTorch implementation (torch.cdist distance matrix, same merges)"),
        show(MOD, "agglomerative_torch"),
        md("## 6. Train / run — ARI per linkage, moons, merge distances"),
        run_demo(MOD),
        md("## 7. Visualization — clusters + a dendrogram from the linkage matrix"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs
import hierarchical as M

X, y = make_blobs(n_samples=60, centers=3, cluster_std=0.7, random_state=0)
m = M.AgglomerativeNumPy(n_clusters=3, linkage="ward").fit(X)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=m.labels_, s=18, cmap="tab10")
ax[0].set_title("Ward agglomerative (cut at k=3)")
# our linkage_ rows are in SciPy format: [idx_a, idx_b, distance, size]
dendrogram(m.linkage_, ax=ax[1], no_labels=True, color_threshold=0)
ax[1].set_title("Dendrogram (height = merge distance)")
ax[1].set_ylabel("merge distance")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- A **single fit yields all granularities** — slice the dendrogram at any height;
  read $k$ off a large vertical gap between successive merges.
- **Linkage matters a lot:** single linkage handles non-convex shapes but chains;
  Ward/complete give compact balls but mis-handle elongated structure.
- **Cost** is roughly $O(n^2)$–$O(n^3)$ and memory $O(n^2)$ — fine for thousands
  of points, not millions.
- Deterministic (no random init), but sensitive to distance scaling —
  standardize features first.

**Next:** seek the *modes* of the density directly with gradient ascent →
mean shift.
"""),
    ]

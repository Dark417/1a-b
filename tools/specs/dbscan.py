from tools.nbreg import register, md, code, show, run_demo

MOD = "dbscan"


@register("dbscan", "ml/clustering/dbscan.ipynb")
def build():
    return [
        md(r"""
# DBSCAN — clustering by density, not by centroids

> Tutorial pair for [`dbscan.py`](dbscan.py).

## 1. Intuition
Forget centroids and a preset $k$. A cluster is just a region where points are
*packed densely*, separated from other clusters by *sparse* regions. DBSCAN walks
through dense neighborhoods, growing a cluster as long as points stay crowded,
and labels the leftovers in sparse regions as **noise**. It finds arbitrarily
shaped clusters (moons, rings) and is robust to outliers.
"""),
        md(r"""
## 2. Concept (the slide)
Two knobs: a radius $\varepsilon$ and a count $m=$ `min_samples`.
- **Core point:** has $\ge m$ points within $\varepsilon$ (a dense interior).
- **Border point:** within $\varepsilon$ of a core point but not itself core.
- **Noise point:** neither — labeled $-1$.

A cluster = a maximal set of points reachable from a core point by hopping
through other core points' $\varepsilon$-balls. No $k$ to choose; instead pick
$\varepsilon$ (the **k-distance elbow**) and $m\,(\approx 2\cdot\dim)$.
**OPTICS** generalizes this: instead of one global $\varepsilon$ it produces a
*reachability plot* you can cut at many scales.
"""),
        md(r"""
## 3. Math derivation

Fix a metric $d$, radius $\varepsilon$, threshold $m$.

**$\varepsilon$-neighborhood.** $N_\varepsilon(p)=\{q: d(p,q)\le\varepsilon\}$.
$p$ is a **core point** iff $|N_\varepsilon(p)|\ge m$.

**Directly density-reachable.** $q$ is directly density-reachable from $p$ if
$p$ is core and $q\in N_\varepsilon(p)$. (Asymmetric: $q$ need not be core.)

**Density-reachable.** $q$ is density-reachable from $p$ if there is a chain
$p=o_1,o_2,\dots,o_\ell=q$ where each $o_{t+1}$ is directly density-reachable
from $o_t$ (so $o_1,\dots,o_{\ell-1}$ are all core). This is the transitive
closure through core points; it is *not* symmetric (a border endpoint can be
reached but cannot reach back).

**Density-connected.** $p,q$ are density-connected if some core $o$ density-reaches
*both*. This **is** symmetric, and it is the equivalence-like relation that
defines clusters:
$$\text{cluster } C = \{\, q : q \text{ is density-connected to some fixed core } p \,\}.$$

**Two cluster properties** (Ester et al. 1996):
- *Maximality:* if $p\in C$ is core and $q$ is density-reachable from $p$, then $q\in C$.
- *Connectivity:* any two points in $C$ are density-connected.

**Algorithm = realize these definitions.** Scan points; from each unvisited
**core** point start a BFS over $N_\varepsilon$, expanding the frontier **only
through core points** (so density-reachability is respected) and absorbing border
points without letting them seed further growth. Non-core points reached by no
core stay noise. Each point is visited $O(1)$ times; the cost is dominated by the
neighborhood queries, $O(n^2)$ naively (or $O(n\log n)$ with a spatial index).

**Choosing $\varepsilon$ (k-distance plot).** Sort every point's distance to its
$m$-th nearest neighbor in descending order. Interior points have small
$k$-distance; noise points have large. The **elbow** of that curve is a good
$\varepsilon$.

**OPTICS.** Define $\text{core-dist}(p)$ = distance to the $m$-th neighbor and
$\text{reach-dist}(o,p)=\max(\text{core-dist}(p), d(p,o))$. Visiting points in
order of smallest reachability yields a **reachability plot** whose *valleys* are
clusters at a given depth — a DBSCAN clustering for *every* $\varepsilon$ at once.
"""),
        md("## 4. NumPy implementation (BFS over the eps-graph + k-distance + OPTICS)"),
        show(MOD, "DBSCANNumPy", "k_distance", "optics_reachability"),
        md("## 5. PyTorch implementation (torch.cdist neighborhood graph)"),
        show(MOD, "DBSCANTorch"),
        md("## 6. Train / run — moons, blobs+noise, eps heuristic, OPTICS"),
        run_demo(MOD),
        md("## 7. Visualization — clusters & noise, point types, k-distance elbow"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import dbscan as M

X, y = make_moons(n_samples=400, noise=0.06, random_state=0)
db = M.DBSCANNumPy(eps=0.2, min_samples=5).fit(X)
kd = M.k_distance(X, k=4)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# noise drawn black with an 'x'
mask = db.labels_ == M.NOISE
ax[0].scatter(X[~mask, 0], X[~mask, 1], c=db.labels_[~mask], s=12, cmap="tab10")
ax[0].scatter(X[mask, 0], X[mask, 1], c="k", marker="x", s=30, label="noise")
ax[0].set_title(f"DBSCAN: {db.n_clusters_} clusters, {int(mask.sum())} noise")
ax[0].legend(loc="upper right")
ax[1].plot(kd)
ax[1].set_xlabel("points sorted"); ax[1].set_ylabel("4-distance")
ax[1].set_title("k-distance plot (pick eps at the elbow)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **No $k$ needed** and clusters can be any shape — great for moons/rings and for
  flagging outliers as noise ($-1$).
- **Very sensitive to $\varepsilon$ and $m$**: too small $\Rightarrow$ everything
  is noise; too large $\Rightarrow$ clusters merge. Use the k-distance elbow.
- **Varying densities break a single $\varepsilon$** — different clusters need
  different radii. This is exactly what **OPTICS** (multi-scale reachability)
  fixes.
- Naive implementation is $O(n^2)$; production code uses KD-/ball-trees.

**Next:** build a *tree* of merges instead of a flat partition → agglomerative
hierarchical clustering.
"""),
    ]

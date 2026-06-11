from tools.nbreg import register, md, code, show, run_demo

MOD = "kmeans"


@register("kmeans", "01.ml/clustering/kmeans.ipynb")
def build():
    return [
        md(r"""
# k-Means — clustering as coordinate descent

> Tutorial pair for [`kmeans.py`](kmeans.py).

## 1. Intuition
Find $k$ "prototypes" (centroids) so that every point is close to its nearest
prototype. Repeat two cheap steps until nothing moves: **assign** points to the
nearest centroid, then **recompute** each centroid as the mean of its points.
"""),
        md(r"""
## 2. Concept (the slide)
- **Objective (inertia):** minimize within-cluster squared distance
  $J=\sum_{i}\lVert \mathbf x_i-\boldsymbol\mu_{c_i}\rVert^2$.
- **Lloyd's algorithm:** alternate assignment (E) and centroid update (M).
- **Caveats:** finds a *local* optimum (init-sensitive), assumes roughly
  spherical, equally-sized clusters, and you must choose $k$.
"""),
        md(r"""
## 3. Math derivation

Minimize over both assignments $c_i\in\{1..k\}$ and centroids $\boldsymbol\mu_j$:

$$J(\{c_i\},\{\boldsymbol\mu_j\})=\sum_{i=1}^{n}\lVert \mathbf x_i-\boldsymbol\mu_{c_i}\rVert^2 .$$

This is **coordinate descent**, decreasing $J$ on each half-step:

- **Assignment (fix $\mu$, optimize $c$):** each point independently picks
  $c_i=\arg\min_j\lVert\mathbf x_i-\boldsymbol\mu_j\rVert^2$ — clearly minimizes its term.
- **Update (fix $c$, optimize $\mu$):** set $\partial J/\partial\boldsymbol\mu_j=0$:
  $$\sum_{i:c_i=j}2(\boldsymbol\mu_j-\mathbf x_i)=0
   \;\Rightarrow\;
   \boxed{\;\boldsymbol\mu_j=\frac{1}{|C_j|}\sum_{i\in C_j}\mathbf x_i\;}$$
  the centroid is the **mean** (hence the name).

Both steps never increase $J$, and there are finitely many assignments, so the
algorithm **converges** — but only to a local minimum.

**k-means++ initialization.** Pick the first centroid at random, then pick each
next centroid with probability $\propto D(\mathbf x)^2$ (squared distance to the
nearest chosen centroid). This spreads seeds out and gives an
$O(\log k)$-competitive expected cost — far better than random seeds.

**Choosing $k$.** *Elbow*: plot $J$ vs $k$, look for the kink. *Silhouette*:
$s_i=\dfrac{b_i-a_i}{\max(a_i,b_i)}$ where $a_i$ = mean intra-cluster distance,
$b_i$ = mean distance to the nearest *other* cluster; average $s_i$ near 1 is good.
"""),
        md("## 4. NumPy implementation (Lloyd + k-means++ + mini-batch + silhouette)"),
        show(MOD, "KMeansNumPy", "MiniBatchKMeansNumPy"),
        md("## 5. PyTorch implementation (GPU-friendly)"),
        show(MOD, "kmeans_torch"),
        md("## 6. Train — compare variants and the elbow"),
        run_demo(MOD),
        md("## 7. Visualization — clusters, centroids, and the elbow curve"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import kmeans as M

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=0)
km = M.KMeansNumPy(k=4).fit(X)
ks = range(2, 9)
inertias = [M.KMeansNumPy(k=k, n_init=3).fit(X).inertia_ for k in ks]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X[:,0], X[:,1], c=km.labels_, s=10, cmap="tab10")
ax[0].scatter(km.centroids[:,0], km.centroids[:,1], c="k", marker="X", s=160)
ax[0].set_title("k-means++ (k=4)")
ax[1].plot(list(ks), inertias, "o-"); ax[1].axvline(4, ls="--", c="r")
ax[1].set_xlabel("k"); ax[1].set_ylabel("inertia"); ax[1].set_title("Elbow")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Always use **k-means++** + several restarts (we take the best inertia).
- It minimizes *Euclidean* inertia → spherical bias; for elongated/varied
  clusters use GMM (soft, elliptical) or DBSCAN (density, arbitrary shapes).
- **Mini-batch** trades a little accuracy for big speed at scale.

**Next:** soft, probabilistic clusters with full covariances → Gaussian Mixture
Models (EM).
"""),
    ]

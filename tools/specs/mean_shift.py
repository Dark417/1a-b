from tools.nbreg import register, md, code, show, run_demo

MOD = "mean_shift"


@register("mean_shift", "01.ml/clustering/mean_shift.ipynb")
def build():
    return [
        md(r"""
# Mean Shift — climbing the density to its modes

> Tutorial pair for [`mean_shift.py`](mean_shift.py).

## 1. Intuition
Put a smooth "bump" (kernel) on every data point and add them up: that is a
**kernel density estimate** of where the data is dense. Now let each point roll
*uphill* on this density surface. Points sliding into the same peak (mode) belong
to the same cluster. You never set $k$ — the number of clusters is whatever the
density's peaks decide, controlled by a single **bandwidth** $h$.
"""),
        md(r"""
## 2. Concept (the slide)
- **KDE:** $\hat f(x)=\frac{1}{n h^d}\sum_i K\!\big(\tfrac{x-x_i}{h}\big)$.
- **Mean-shift step:** replace $x$ by the **kernel-weighted mean** of nearby
  points. This is gradient ascent on $\hat f$ with an automatic step size.
- **Kernels:** *flat* (average points within radius $h$) or *Gaussian* (smooth
  RBF weights, $h$ acts as a std).
- **Clusters:** seed a trajectory at every point, run to convergence, then merge
  points whose modes nearly coincide. Bandwidth $h$ is the whole story: small $h$
  $\Rightarrow$ many modes, large $h$ $\Rightarrow$ few.
"""),
        md(r"""
## 3. Math derivation

Write the kernel via a radial **profile** $k$: $K(u)=c_k\,k(\lVert u\rVert^2)$.
The density estimate is
$$\hat f(x)=\frac{c_k}{n h^d}\sum_{i=1}^n k\!\Big(\Big\lVert\frac{x-x_i}{h}\Big\rVert^2\Big).$$

**Gradient.** Differentiate w.r.t. $x$ (let $g=-k'$):
$$\nabla\hat f(x)=\frac{2c_k}{n h^{d+2}}\sum_i (x_i-x)\,k'\!\big(\cdot\big)
=\frac{2c_k}{n h^{d+2}}\sum_i (x_i-x)\,\big(-g(\cdot)\big)
=\frac{2c_k}{n h^{d+2}}\Big[\sum_i (x_i-x)\,g_i\Big],$$
with $g_i=g\big(\lVert(x-x_i)/h\rVert^2\big)$. Factor out $\sum_i g_i$:
$$\nabla\hat f(x)=\underbrace{\frac{2c_k}{n h^{d+2}}\sum_i g_i}_{\ge 0}\;
\Big[\underbrace{\frac{\sum_i x_i\,g_i}{\sum_i g_i}-x}_{\textstyle\equiv\ m(x)}\Big].$$

The bracket is the **mean-shift vector**
$$\boxed{\,m(x)=\frac{\sum_i x_i\,g_i}{\sum_i g_i}-x\,}$$
— it points in the **same direction as the density gradient** but is already
normalized by the local density. Hence the fixed-point update
$$x \leftarrow x + m(x) = \frac{\sum_i x_i\,g_i}{\sum_i g_i}$$
is **adaptive gradient ascent**: a big step in flat regions, a small one near a
peak. Comaniciu & Meer prove the trajectory is smooth and converges to a
stationary point (a mode) of $\hat f$.

**Kernels.**
- *Gaussian:* $k(t)=e^{-t/2}\Rightarrow g(t)=e^{-t/2}$, so weights are
  $g_i=\exp(-\lVert x-x_i\rVert^2/2h^2)$ — a soft RBF average.
- *Flat (Epanechnikov-like step):* $k(t)=\mathbf 1[t\le 1]\Rightarrow g$ is the
  indicator, so the update is simply the **mean of points within radius $h$**.

**Clustering.** Run the iteration from each data point; points whose converged
modes lie within a tolerance are merged into one cluster. **Bandwidth** is chosen
by a heuristic, e.g. the mean of the $q$-quantile nearest-neighbor distances.
"""),
        md("## 4. NumPy implementation (flat + Gaussian kernels, bandwidth heuristic)"),
        show(MOD, "MeanShiftNumPy", "estimate_bandwidth"),
        md("## 5. PyTorch implementation (all seeds shifted in parallel via torch.cdist)"),
        show(MOD, "MeanShiftTorch"),
        md("## 6. Train / run — ARI, kernel comparison, bandwidth vs #clusters"),
        run_demo(MOD),
        md("## 7. Visualization — clusters, mode centers, and convergence trajectories"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import mean_shift as M

X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
h = M.estimate_bandwidth(X, quantile=0.3) / 2.5
ms = M.MeanShiftNumPy(bandwidth=h, kernel="gaussian").fit(X)

# trace a few trajectories rolling uphill to their modes
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=ms.labels_, s=12, cmap="tab10")
ax[0].scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
              c="k", marker="X", s=180)
ax[0].set_title(f"Mean shift: {ms.n_clusters_} modes (h={h:.2f})")
for i in range(0, len(X), 30):
    y0 = X[i].copy(); path = [y0.copy()]
    for _ in range(60):
        y1 = ms._shift(y0, X)
        path.append(y1.copy())
        if np.linalg.norm(y1 - y0) < 1e-4:
            break
        y0 = y1
    path = np.array(path)
    ax[1].plot(path[:, 0], path[:, 1], "-o", ms=2, lw=1)
ax[1].scatter(X[:, 0], X[:, 1], c="lightgray", s=6, zorder=0)
ax[1].set_title("Trajectories ascending the KDE to modes")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **No $k$**: clusters emerge from the density; **bandwidth $h$ is the only knob**
  and it sets everything (small $h\Rightarrow$ over-segment, large $h\Rightarrow$
  one blob). Choose it with a nearest-neighbor heuristic.
- A flat kernel uses $h$ as a hard radius; a Gaussian kernel uses it as a std, so
  the *same* clusters need a smaller $h$ for the Gaussian.
- Robust to cluster shape and outliers (modes ignore sparse points), but the
  naive version is $O(\text{iters}\cdot n^2)$ — seed from a subset / bin for speed.
- Mode-merging tolerance matters: too large fuses real clusters.

**Next:** cluster by the *geometry of a similarity graph* → spectral clustering.
"""),
    ]

from tools.nbreg import register, md, code, show, run_demo

MOD = "knn"


@register("knn", "ml/knn/knn.ipynb")
def build():
    return [
        md(r"""
# k-Nearest Neighbours — learning by memorizing

> Tutorial pair for [`knn.py`](knn.py).

## 1. Intuition
"You are the average of your k closest friends." To label a new point, find the
k training points nearest to it and let them vote. There is **no training** — we
just store the data and do all the work at query time (*lazy learning*).
"""),
        md(r"""
## 2. Concept (the slide)
- **Classification:** majority (optionally distance-weighted) vote of the k nearest.
- **Regression:** (weighted) mean of the k nearest targets.
- **Hyperparameters:** $k$, the distance metric, and the weighting.
- **Geometry matters:** features must be on comparable scales (standardize!), and
  it degrades in high dimensions (curse of dimensionality).
"""),
        md(r"""
## 3. Math derivation

**Distance.** Minkowski-$p$: $\;d_p(\mathbf a,\mathbf b)=\big(\sum_j|a_j-b_j|^p\big)^{1/p}$
($p{=}2$ Euclidean, $p{=}1$ Manhattan). For Euclidean we use the identity
$\lVert\mathbf a-\mathbf b\rVert^2=\lVert\mathbf a\rVert^2+\lVert\mathbf b\rVert^2-2\,\mathbf a^\top\mathbf b$
to compute all pairwise distances with one matmul.

**Prediction.** Let $\mathcal N_k(\mathbf x)$ be the indices of the k nearest
neighbours and $w_i$ a weight ($w_i{=}1$ uniform, or $w_i{=}1/d_i$ weighted).
$$\hat y_{\text{clf}}=\arg\max_{c}\sum_{i\in\mathcal N_k}w_i\,\mathbb 1[y_i=c],
  \qquad
  \hat y_{\text{reg}}=\frac{\sum_{i\in\mathcal N_k}w_i\,y_i}{\sum_{i\in\mathcal N_k}w_i}.$$

**Bias–variance via $k$.** Small $k$ → flexible, low bias, **high variance**
(noisy boundary). Large $k$ → smooth, higher bias, low variance. The
1-NN error is famously $\le 2\times$ the Bayes error as $n\to\infty$
(Cover & Hart).

**Speed.** Brute force is $O(nd)$ per query. A **KD-tree** partitions space so
the average query is $O(\log n)$ in low dimensions (it prunes whole subtrees
whose bounding region is farther than the current k-th best).
"""),
        md("## 4. NumPy implementation (brute force + KD-tree)"),
        show(MOD, "KNNNumPy", "KDTree"),
        md("## 5. PyTorch implementation (`torch.cdist`, GPU-ready)"),
        show(MOD, "knn_torch"),
        md("## 6. Train — vary k, weighting, and check KD-tree == brute force"),
        run_demo(MOD),
        md("## 7. Visualization — how k smooths the decision boundary"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import knn as M

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
X = (X - X.mean(0)) / X.std(0)
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, k in zip(axes, (1, 5, 25)):
    zz = M.KNNNumPy(k=k).fit(X, y).predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.4, cmap="coolwarm")
    ax.scatter(X[:,0], X[:,1], c=y, s=12, edgecolor="k", cmap="coolwarm")
    ax.set_title(f"k = {k}")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Standardize features** — kNN is pure geometry.
- Choose $k$ by cross-validation; odd $k$ avoids ties in binary problems.
- Memory- and query-heavy at scale; use KD/ball-trees or approximate NN.
- Curse of dimensionality: in high-$d$ all points become equidistant.
"""),
    ]

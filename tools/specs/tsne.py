from tools.nbreg import register, md, code, show, run_demo

MOD = "tsne"


@register("tsne", "01.ml/dimensionality-reduction/tsne.ipynb")
def build():
    return [
        md(r"""
# t-SNE — neighbor embeddings that reveal clusters

> Tutorial pair for [`tsne.py`](tsne.py).

## 1. Intuition
t-SNE turns *distances* into *probabilities of being neighbors*, then arranges points
in 2D so the neighbor probabilities match. A point's close friends in high-D stay
close in the map; far-away points are free to drift apart. The result is the famous
picture of well-separated clusters.
"""),
        md(r"""
## 2. Concept (the slide)
- **High-D affinities $P$:** for each point, a Gaussian over neighbors. Its bandwidth
  is set *per point* so the effective number of neighbors equals a target **perplexity**.
- **Low-D affinities $Q$:** a heavy-tailed **Student-t** kernel. The fat tail lets
  moderately-distant points be modeled with small $q$ without huge penalty — this
  cures the **crowding problem**.
- **Objective:** make $Q$ look like $P$ by minimizing $\mathrm{KL}(P\,\|\,Q)$ with
  gradient descent.
- **Tricks:** *early exaggeration* (scale up $P$ early to form tight clusters) and
  *momentum*.
"""),
        md(r"""
## 3. Math derivation

**High-dimensional conditional affinities.** Asymmetric Gaussian neighbor probability

$$p_{j|i}=\frac{\exp(-\lVert\mathbf x_i-\mathbf x_j\rVert^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\lVert\mathbf x_i-\mathbf x_k\rVert^2/2\sigma_i^2)},\qquad p_{i|i}=0.$$

**Perplexity calibration.** Choose each $\sigma_i$ so the row's entropy matches a target:

$$\mathrm{Perp}(P_i)=2^{H(P_i)},\qquad H(P_i)=-\sum_j p_{j|i}\log_2 p_{j|i}.$$

Perplexity $\approx$ "how many neighbors each point feels". We **binary-search**
$\beta_i=1/2\sigma_i^2$ until $H(P_i)=\log_2(\text{perplexity})$.

**Symmetrize** into a single joint distribution:

$$p_{ij}=\frac{p_{j|i}+p_{i|j}}{2n},\qquad \sum_{ij}p_{ij}=1.$$

**Low-dimensional affinities (Student-t, 1 dof):**

$$q_{ij}=\frac{(1+\lVert\mathbf y_i-\mathbf y_j\rVert^2)^{-1}}{\sum_{k\ne l}(1+\lVert\mathbf y_k-\mathbf y_l\rVert^2)^{-1}}.$$

**Objective — KL divergence:**

$$C=\mathrm{KL}(P\,\|\,Q)=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}}.$$

**Gradient.** Writing $d_{ij}=\lVert\mathbf y_i-\mathbf y_j\rVert$ and using
$\sum q=1$, the gradient telescopes to the famous compact form

$$\boxed{\;\frac{\partial C}{\partial \mathbf y_i}=4\sum_{j}(p_{ij}-q_{ij})\,(\mathbf y_i-\mathbf y_j)\,(1+d_{ij}^2)^{-1}\;}$$

Each term is a **spring**: attractive when $p>q$ (should be closer), repulsive when
$p<q$. The factor $(1+d^2)^{-1}$ is exactly the Student-t tail that prevents crowding.
We descend with momentum and **early exaggeration** (multiply $P$ by ~12 early on).
"""),
        md("## 4. NumPy implementation (perplexity search, P/Q, KL gradient, early exaggeration)"),
        show(MOD, "TSNENumPy"),
        md("## 5. PyTorch implementation (same objective, KL gradient via autograd)"),
        show(MOD, "tsne_torch"),
        md("## 6. Train / run — KL, trustworthiness, cluster separation"),
        run_demo(MOD),
        md("## 7. Visualization — digits embedded in 2D"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import tsne as M

from sklearn.datasets import load_digits
digits = load_digits()
rng = np.random.default_rng(0)
idx = rng.choice(len(digits.data), 300, replace=False)
X, y = digits.data[idx], digits.target[idx]
X = (X - X.mean(0)) / (X.std(0) + 1e-8)

Y = M.TSNENumPy(n_components=2, perplexity=30, n_iter=300).fit_transform(X)

plt.figure(figsize=(6, 5))
sc = plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=18)
plt.colorbar(sc, label="digit"); plt.title("t-SNE of handwritten digits")
plt.xlabel("dim 1"); plt.ylabel("dim 2"); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- t-SNE is for **visualization**, not a general feature transform: there is no
  `transform()` for new points and **distances between clusters are not meaningful**.
- **Perplexity matters** (typical 5–50); too small fragments, too large blurs.
- It is **non-convex** — different seeds give different maps; use early exaggeration
  + momentum and don't over-interpret cluster sizes/gaps.
- $O(n^2)$ per step (this from-scratch version); real implementations use
  Barnes–Hut / FFT approximations. For new-point mapping and global structure,
  prefer **UMAP**.
"""),
    ]

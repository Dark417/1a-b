from tools.nbreg import register, md, code, show, run_demo

MOD = "umap"


@register("umap", "ml/dimensionality-reduction/umap.ipynb")
def build():
    return [
        md(r"""
# UMAP — manifold layout via a fuzzy neighbor graph

> Tutorial pair for [`umap.py`](umap.py).

## 1. Intuition
UMAP assumes the data lies on a manifold and builds a weighted nearest-neighbor
**graph** whose edge weights say "how surely are these two points neighbors?". It then
drops a low-dimensional graph and tugs it into shape: connected points **attract**,
random non-neighbors **repel**. Like t-SNE it shows clusters, but it tends to keep more
**global structure** and is fast.
"""),
        md(r"""
## 2. Concept (the slide)
- **Fuzzy simplicial set:** a kNN graph where each edge carries a *membership
  strength* in $[0,1]$. Distances are measured from each point's nearest neighbor
  ($\rho_i$, local connectivity) and scaled by a per-point bandwidth $\sigma_i$.
- **Symmetrize** the directed memberships with a fuzzy **union** (probabilistic
  t-conorm).
- **Low-D layout:** edges get a smooth membership $\phi(d)=1/(1+a\,d^{2b})$.
  Minimize the fuzzy-set **cross-entropy** between high-D and low-D memberships —
  an **attractive** term on edges and a **repulsive** term on non-edges, optimized by
  SGD with **negative sampling**.
"""),
        md(r"""
## 3. Math derivation

**Local fuzzy memberships.** For point $i$ with $k$ nearest neighbors, let
$\rho_i=\min_{j}d(\mathbf x_i,\mathbf x_j)$ (nearest-neighbor distance). Choose a
bandwidth $\sigma_i$ by binary search so the row's total membership is constant:

$$\sum_{j\in \mathrm{kNN}(i)}\exp\!\Big(-\frac{\max(0,\,d_{ij}-\rho_i)}{\sigma_i}\Big)=\log_2 k.$$

The directed membership strength is
$\;w_{j|i}=\exp\!\big(-\max(0,d_{ij}-\rho_i)/\sigma_i\big)$. Subtracting $\rho_i$
guarantees each point connects to its nearest neighbor with weight 1 (local
connectivity assumption).

**Symmetrization (fuzzy union / probabilistic t-conorm):**

$$w_{ij}=w_{j|i}+w_{i|j}-w_{j|i}\,w_{i|j}.$$

**Low-dimensional membership.** UMAP models the membership of a low-D edge of length
$d$ by the smooth curve

$$\phi(d)=\frac{1}{1+a\,d^{2b}},$$

with $(a,b)$ **fit by least squares** to the desired shape (controlled by `min_dist`
and `spread`): flat at 1 for $d\le\texttt{min\_dist}$, exponential decay beyond.

**Objective — fuzzy cross-entropy.** Treat each pair as a Bernoulli with high-D
"truth" $w_{ij}$ and low-D "prediction" $q_{ij}=\phi(\lVert\mathbf y_i-\mathbf y_j\rVert)$:

$$C=\sum_{i\ne j}\Big[\,w_{ij}\log\frac{w_{ij}}{q_{ij}}+(1-w_{ij})\log\frac{1-w_{ij}}{1-q_{ij}}\Big].$$

The first part is **attractive** (pull neighbors together where $w$ is large), the
second is **repulsive** (push non-neighbors apart where $w\approx0$).

**Gradients / forces.** Differentiating $C$ w.r.t. $\mathbf y_i$, with
$d^2=\lVert\mathbf y_i-\mathbf y_j\rVert^2$:

$$\text{attractive: } \frac{-2ab\,d^{2(b-1)}}{1+a\,d^{2b}}(\mathbf y_i-\mathbf y_j)\,w_{ij},
\qquad
\text{repulsive: } \frac{2b}{(\epsilon+d^2)(1+a\,d^{2b})}(\mathbf y_i-\mathbf y_j).$$

Evaluating the full repulsive sum is $O(n^2)$, so UMAP uses **negative sampling**:
for each edge, repel against a few random points. We also **decay the learning rate**
linearly over epochs. (Our NumPy version applies these same forces vectorized per
epoch for speed.)
"""),
        md("## 4. NumPy implementation (fuzzy graph + attractive/repulsive SGD)"),
        show(MOD, "UMAPNumPy"),
        md("## 5. PyTorch implementation (fuzzy cross-entropy via autograd)"),
        show(MOD, "umap_torch"),
        md("## 6. Train / run — fitted (a,b), trustworthiness, cluster ratio"),
        run_demo(MOD),
        md("## 7. Visualization — digits laid out by UMAP"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import umap as M

from sklearn.datasets import load_digits
digits = load_digits()
rng = np.random.default_rng(0)
idx = rng.choice(len(digits.data), 300, replace=False)
X, y = digits.data[idx], digits.target[idx]
X = (X - X.mean(0)) / (X.std(0) + 1e-8)

Y = M.UMAPNumPy(n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=120).fit_transform(X)

plt.figure(figsize=(6, 5))
sc = plt.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=18)
plt.colorbar(sc, label="digit"); plt.title("UMAP of handwritten digits")
plt.xlabel("dim 1"); plt.ylabel("dim 2"); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **`n_neighbors`** trades local vs global structure (small → local detail, large →
  global shape); **`min_dist`** controls how tightly points clump.
- UMAP usually preserves **global structure** better than t-SNE and is faster
  (negative sampling), and unlike t-SNE it can **embed new points**.
- Still a **visualization** tool: inter-cluster distances and densities are not
  literal — don't read absolute geometry into the map.
- The layout is **stochastic / non-convex**; fix the seed for reproducibility.
- This is a **simplified** UMAP (faithful fuzzy graph + force layout); the real library
  adds approximate kNN, a spectral initialization, and a sampled edge schedule.
"""),
    ]

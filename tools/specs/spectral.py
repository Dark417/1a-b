from tools.nbreg import register, md, code, show, run_demo

MOD = "spectral"


@register("spectral", "ml/clustering/spectral.ipynb")
def build():
    return [
        md(r"""
# Spectral Clustering — clustering the graph, not the coordinates

> Tutorial pair for [`spectral.py`](spectral.py).

## 1. Intuition
Two points on opposite ends of a curved "moon" are far apart in space but
*connected* through a chain of near neighbors. Spectral clustering turns the data
into a **similarity graph**, then uses the eigenvectors of its **Laplacian** to
find a low-dimensional embedding where those connected points sit together.
Ordinary k-means in that embedding then separates shapes that k-means could never
separate in the original space.
"""),
        md(r"""
## 2. Concept (the slide)
1. **Affinity** $W$: edge weight $w_{ij}=\exp(-\gamma\lVert x_i-x_j\rVert^2)$
   (optionally sparsified to a $k$-NN graph).
2. **Laplacian** $L=D-W$ with degree $D=\mathrm{diag}(\sum_j w_{ij})$ — or a
   normalized variant.
3. **Embed**: take the eigenvectors of the $k$ *smallest* eigenvalues $\to$
   $U\in\mathbb R^{n\times k}$.
4. **k-means** on the rows of $U$ gives the clusters.

The smallest eigenvectors are the *smoothest* functions on the graph — nearly
constant within well-connected components — which is exactly what makes the
clusters fall out.
"""),
        md(r"""
## 3. Math derivation

**The graph Laplacian.** For affinity $W\succeq 0$ and degrees $d_i=\sum_j w_{ij}$,
$L=D-W$. For any vector $f\in\mathbb R^n$,
$$f^\top L f=\tfrac12\sum_{i,j} w_{ij}\,(f_i-f_j)^2\ \ge 0,$$
so $L$ is positive semidefinite. The all-ones vector gives $L\mathbf 1=0$, so
$\lambda_1=0$; the multiplicity of eigenvalue $0$ equals the number of connected
components, and those eigenvectors are constant on each component — the ideal
cluster indicators.

**Graph cut $\to$ eigenproblem.** Partition $V$ into $A,\bar A$. The cut is
$\mathrm{cut}(A,\bar A)=\sum_{i\in A,j\in\bar A} w_{ij}$. Minimizing it alone
peels off single vertices, so we *balance* it — the **normalized cut**
$$\mathrm{Ncut}(A,\bar A)=\mathrm{cut}(A,\bar A)\Big(\tfrac{1}{\mathrm{vol}(A)}+\tfrac{1}{\mathrm{vol}(\bar A)}\Big),
\quad \mathrm{vol}(A)=\sum_{i\in A} d_i.$$
With a $\{+,-\}$ indicator $f$ encoding the partition, one shows
$\mathrm{Ncut}\propto \dfrac{f^\top L f}{f^\top D f}$ subject to $f\perp_D \mathbf 1$.
Exact minimization over discrete $f$ is **NP-hard**; **relax** $f$ to real values:
$$\min_{f}\ \frac{f^\top L f}{f^\top D f}\quad\text{s.t. } f^\top D\mathbf 1=0.$$
This Rayleigh quotient is minimized by the **second-smallest generalized
eigenvector** of $L f=\lambda D f$ — the *Fiedler vector*. For $k>2$ clusters,
take the $k$ smallest eigenvectors and cluster their rows.

**Normalized Laplacians.** Solving $Lf=\lambda Df$ corresponds to
- random-walk: $L_{rw}=I-D^{-1}W$ (Shi-Malik), eigenvectors of $D^{-1}W$;
- symmetric: $L_{sym}=I-D^{-1/2}WD^{-1/2}=D^{1/2}L_{rw}D^{-1/2}$
  (Ng-Jordan-Weiss). For $L_{sym}$ the embedding rows are **renormalized to unit
  length** before k-means.

**Laplacian eigenmaps view.** The embedding $x_i\mapsto U_{i,:}$ minimizes
$\sum_{ij} w_{ij}\lVert U_{i,:}-U_{j,:}\rVert^2 = \mathrm{tr}(U^\top L U)$ subject
to orthonormality — strongly connected points are pulled together, so the
embedding makes clusters linearly separable for k-means.

**Eigengap heuristic.** Sort the eigenvalues $0=\lambda_1\le\lambda_2\le\cdots$;
a large gap $\lambda_{k+1}-\lambda_k$ suggests $k$ clusters.
"""),
        md("## 4. NumPy implementation (RBF/kNN affinity, 3 Laplacians, eigen-embed + k-means)"),
        show(MOD, "SpectralClusteringNumPy"),
        md("## 5. PyTorch implementation (torch.cdist affinity + torch.linalg.eigh)"),
        show(MOD, "spectral_torch"),
        md("## 6. Train / run — moons & circles vs k-means, Laplacian variants, eigengap"),
        run_demo(MOD),
        md("## 7. Visualization — input clusters and the spectral embedding"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import spectral as M

X, y = make_moons(n_samples=300, noise=0.06, random_state=0)
sc = M.SpectralClusteringNumPy(2, affinity="rbf", gamma=15, laplacian="sym").fit(X)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=sc.labels_, s=14, cmap="coolwarm")
ax[0].set_title("Spectral clustering (two moons)")
# the 2-D embedding: clusters become two tight, linearly separable blobs
U = sc.embedding_
ax[1].scatter(U[:, 0], U[:, 1], c=sc.labels_, s=14, cmap="coolwarm")
ax[1].set_xlabel("eigvec 1"); ax[1].set_ylabel("eigvec 2")
ax[1].set_title("Laplacian eigen-embedding (k-means lives here)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Handles non-convex clusters** (moons, rings) by clustering graph connectivity
  rather than Euclidean proximity — where plain k-means fails.
- **Affinity scale $\gamma$ is critical**: too large disconnects the graph, too
  small connects everything. The $k$-NN graph is often more robust than full RBF.
- **Normalized** Laplacians ($L_{sym}$, $L_{rw}$) usually beat the unnormalized
  one when degrees vary; remember to **row-normalize** the $L_{sym}$ embedding.
- Cost is $O(n^2)$ memory and an $O(n^3)$ dense eigensolve — use sparse $k$-NN
  graphs + Lanczos for large $n$. Read $k$ from the **eigengap**.

**Next:** leave clustering for *dimensionality reduction* — find the directions
of maximum variance with PCA.
"""),
    ]

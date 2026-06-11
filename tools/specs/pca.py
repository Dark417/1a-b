from tools.nbreg import register, md, code, show, run_demo

MOD = "pca"


@register("pca", "01.ml/dimensionality-reduction/pca.ipynb")
def build():
    return [
        md(r"""
# PCA — the directions of maximum variance

> Tutorial pair for [`pca.py`](pca.py).

## 1. Intuition
High-dimensional data often lives near a low-dimensional subspace. PCA finds the
orthogonal axes along which the data varies most and keeps the top few — the
best linear compression in a least-squares sense.
"""),
        md(r"""
## 2. Concept (the slide)
- **Center** the data (subtract the mean).
- The **first principal component** is the unit direction of greatest variance;
  each next one is orthogonal to all previous and maximizes remaining variance.
- These directions are the **eigenvectors of the covariance matrix** (= right
  singular vectors of the centered data). Eigenvalues = variance captured.
"""),
        md(r"""
## 3. Math derivation

Center: $\tilde X = X - \bar{\mathbf x}$. Covariance $C=\frac1n\tilde X^\top\tilde X$.

**Maximize variance.** For a unit direction $\mathbf w$, the projected variance is
$\operatorname{Var}(\tilde X\mathbf w)=\mathbf w^\top C\,\mathbf w$. Solve

$$\max_{\mathbf w}\ \mathbf w^\top C\mathbf w \quad\text{s.t.}\quad \mathbf w^\top\mathbf w=1.$$

Lagrangian $\mathcal L=\mathbf w^\top C\mathbf w-\lambda(\mathbf w^\top\mathbf w-1)$,
set $\nabla_{\mathbf w}=0$:

$$2C\mathbf w-2\lambda\mathbf w=0\;\Longrightarrow\;\boxed{C\mathbf w=\lambda\mathbf w}.$$

So $\mathbf w$ is an **eigenvector** of $C$ and the variance it captures is the
**eigenvalue** $\lambda=\mathbf w^\top C\mathbf w$. The top-$k$ eigenvectors (largest
$\lambda$) are the principal components.

**SVD route (preferred).** $\tilde X=U\Sigma V^\top$. Then
$C=\frac1n V\Sigma^2V^\top$, so columns of $V$ are the components and
$\sigma_i^2/n$ the variances — computed without forming $C$ (better conditioned).

**Reconstruction view.** PCA equivalently minimizes reconstruction error
$\sum_i\lVert\mathbf x_i-(\text{projection onto }k\text{-dim subspace})\rVert^2$ —
the same subspace. **Explained-variance ratio** $\lambda_i/\sum_j\lambda_j$ tells
you how many components to keep.

**Kernel PCA.** Replace inner products $\mathbf x^\top\mathbf x'$ with a kernel
$k(\mathbf x,\mathbf x')$ (e.g. RBF) and eigendecompose the centered kernel
(Gram) matrix — PCA in a nonlinear feature space, untangling curved manifolds.
"""),
        md("## 4. NumPy implementation (eig, SVD, whitening, kernel PCA)"),
        show(MOD, "PCANumPy", "KernelPCANumPy"),
        md("## 5. PyTorch implementation"),
        show(MOD, "pca_torch"),
        md("## 6. Run — eig vs SVD agreement, reconstruction, kernel PCA"),
        run_demo(MOD),
        md("## 7. Visualization — 4D Iris projected to 2D + scree plot"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pca as M

X, y = load_iris(return_X_y=True); X = (X - X.mean(0)) / X.std(0)
p = M.PCANumPy(n_components=4).fit(X); Z = p.transform(X)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(Z[:,0], Z[:,1], c=y, cmap="viridis", edgecolor="k", s=18)
ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].set_title("Iris in 2D")
ax[1].bar(range(1,5), p.explained_variance_ratio_)
ax[1].set_xlabel("component"); ax[1].set_ylabel("explained variance ratio")
ax[1].set_title("Scree plot")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Centering is mandatory**; **standardize** when features have different units.
- PCA is unsupervised (ignores labels) — for class-separating directions use LDA.
- It is **linear**; curved manifolds need kernel PCA, t-SNE, or UMAP.
- Keep enough components to explain ~95% of variance (read the scree plot).
"""),
    ]

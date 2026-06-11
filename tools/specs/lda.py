from tools.nbreg import register, md, code, show, run_demo

MOD = "lda"


@register("lda", "01.ml/dimensionality-reduction/lda.ipynb")
def build():
    return [
        md(r"""
# LDA — supervised directions that separate classes

> Tutorial pair for [`lda.py`](lda.py).

## 1. Intuition
PCA asks "where does the data vary most?" — it never looks at the labels. **Fisher's
Linear Discriminant Analysis** asks a sharper question: *which directions push the
classes apart while keeping each class tight?* Project onto those directions and the
classes line up neatly, ready for a simple linear classifier.
"""),
        md(r"""
## 2. Concept (the slide)
- Summarize each class by its **mean**; summarize spread with two scatter matrices:
  - **within-class** $S_W$ — how fuzzy each class is around its own mean,
  - **between-class** $S_B$ — how far the class means sit from the global mean.
- A good projection $\mathbf w$ makes $S_B$ large and $S_W$ small. Maximize their
  ratio (the **Fisher / Rayleigh quotient**).
- The optimum solves a **generalized eigenvalue problem**. At most $C-1$ useful
  axes exist for $C$ classes (because $S_B$ has rank $\le C-1$).
- With a shared-covariance Gaussian assumption, LDA is also a **linear classifier**.
"""),
        md(r"""
## 3. Math derivation

**Scatter matrices.** With class means $\mathbf m_c$, global mean $\mathbf m$, counts $n_c$:

$$S_W=\sum_{c}\sum_{i\in c}(\mathbf x_i-\mathbf m_c)(\mathbf x_i-\mathbf m_c)^\top,
\qquad
S_B=\sum_{c} n_c(\mathbf m_c-\mathbf m)(\mathbf m_c-\mathbf m)^\top.$$

**Fisher criterion.** For a projection direction $\mathbf w$, the between- and
within-class variances of the projected data are $\mathbf w^\top S_B\mathbf w$ and
$\mathbf w^\top S_W\mathbf w$. Maximize their ratio:

$$J(\mathbf w)=\frac{\mathbf w^\top S_B\mathbf w}{\mathbf w^\top S_W\mathbf w}.$$

$J$ is scale-invariant, so fix $\mathbf w^\top S_W\mathbf w=1$ and use a Lagrangian
$\mathcal L=\mathbf w^\top S_B\mathbf w-\lambda(\mathbf w^\top S_W\mathbf w-1)$. Setting
$\nabla_{\mathbf w}\mathcal L=0$:

$$2S_B\mathbf w-2\lambda S_W\mathbf w=0\;\Longrightarrow\;\boxed{S_B\mathbf w=\lambda S_W\mathbf w}.$$

This **generalized eigenproblem** is equivalent to $S_W^{-1}S_B\mathbf w=\lambda\mathbf w$,
and at the optimum $J(\mathbf w)=\lambda$ — the eigenvalue *is* the separability. Pick
the top-$k$ eigenvectors.

**Whitening trick (what the code does).** $S_B\mathbf w=\lambda S_W\mathbf w$ is awkward
because $S_W^{-1}S_B$ is generally non-symmetric. Factor $S_W=S_W^{1/2}S_W^{1/2}$ and
substitute $\mathbf u=S_W^{1/2}\mathbf w$:

$$\big(S_W^{-1/2}S_B\,S_W^{-1/2}\big)\,\mathbf u=\lambda\,\mathbf u,$$

a **symmetric** standard eigenproblem (use `eigh`, numerically stable); recover
$\mathbf w=S_W^{-1/2}\mathbf u$.

**Two-class closed form.** With $C=2$, $S_B$ has rank 1 and the single optimal
direction is

$$\boxed{\;\mathbf w\propto S_W^{-1}(\mathbf m_1-\mathbf m_0)\;}$$

**LDA as a Gaussian classifier.** Assume each class is Gaussian with a *shared*
covariance $\Sigma=S_W/(n-C)$ and prior $\pi_c$. The log-posterior's quadratic term
cancels, leaving a **linear discriminant**

$$\delta_c(\mathbf x)=\mathbf x^\top\Sigma^{-1}\mathbf m_c-\tfrac12\mathbf m_c^\top\Sigma^{-1}\mathbf m_c+\log\pi_c,$$

and we predict $\arg\max_c\delta_c(\mathbf x)$.

**Regularization.** When $S_W$ is singular ($n<d$ or collinear features),
**shrink** it toward a sphere: $S_W\leftarrow(1-\gamma)S_W+\gamma\frac{\operatorname{tr}S_W}{d}I$.
"""),
        md("## 4. NumPy implementation (generalized eigenproblem + two-class + Gaussian classifier)"),
        show(MOD, "LDANumPy", "fisher_two_class"),
        md("## 5. PyTorch implementation (whitening + symmetric eigensolver)"),
        show(MOD, "lda_torch"),
        md("## 6. Train / run — eigenvalues, accuracy, and comparison to sklearn & PCA"),
        run_demo(MOD),
        md("## 7. Visualization — Iris projected by LDA vs PCA"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import lda as M

from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True); X = (X - X.mean(0)) / X.std(0)

Z = M.LDANumPy(n_components=2).fit_transform(X, y)          # supervised
Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
Zp = Xc @ Vt[:2].T                                          # unsupervised PCA

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for c in np.unique(y):
    ax[0].scatter(Z[y == c, 0], Z[y == c, 1], s=18, label=f"class {c}")
    ax[1].scatter(Zp[y == c, 0], Zp[y == c, 1], s=18)
ax[0].set_title("LDA (supervised)"); ax[0].set_xlabel("LD1"); ax[0].set_ylabel("LD2")
ax[0].legend()
ax[1].set_title("PCA (unsupervised)"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- LDA is **supervised** — it uses labels to find class-separating axes; PCA does not.
- You get at most **$C-1$** discriminant directions (3 classes → 2 axes).
- $S_W$ must be invertible: **standardize**, and use **shrinkage** when $n<d$.
- The Gaussian-classifier view assumes **equal class covariances**; when they differ,
  use **QDA** (per-class covariance, quadratic boundary).
- LDA is **linear**; for curved class boundaries, combine with kernels or use
  nonlinear embeddings (t-SNE/UMAP) for *visualization* (not classification).
"""),
    ]

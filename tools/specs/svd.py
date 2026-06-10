from tools.nbreg import register, md, code, show, run_demo

MOD = "svd"


@register("svd", "ml/dimensionality-reduction/svd.ipynb")
def build():
    return [
        md(r"""
# Truncated SVD — the best low-rank view of any matrix

> Tutorial pair for [`svd.py`](svd.py).

## 1. Intuition
Every matrix can be written as a sum of rank-1 "layers", each with a weight (the
singular value) saying how important it is. Keep the few heaviest layers and you have
the **best possible** low-rank approximation — the basis of compression,
dimensionality reduction (truncated SVD), and Latent Semantic Analysis for text.
"""),
        md(r"""
## 2. Concept (the slide)
- **SVD:** $A=U\Sigma V^\top$ with orthonormal $U,V$ and singular values
  $\sigma_1\ge\sigma_2\ge\dots\ge0$ on $\Sigma$'s diagonal.
- Geometrically $A$ = rotate ($V^\top$) → scale ($\Sigma$) → rotate ($U$).
- **Truncate** to top $k$: $A_k=U_k\Sigma_k V_k^\top$. The row embedding is
  $Z=AV_k=U_k\Sigma_k$ (an $n\times k$ reduction). **No centering** needed — that's
  the difference from PCA and what lets it run on sparse term–document matrices (LSA).
- **Eckart–Young:** $A_k$ is the *optimal* rank-$k$ approximation in both Frobenius
  and spectral norm.
- **Randomized SVD** computes the top $k$ fast via a random projection sketch.
"""),
        md(r"""
## 3. Math derivation

**Existence.** For any $A\in\mathbb R^{n\times d}$ there exist orthonormal
$U\in\mathbb R^{n\times r}$, $V\in\mathbb R^{d\times r}$ ($r=\operatorname{rank}A$) and
$\sigma_1\ge\dots\ge\sigma_r>0$ with

$$A=U\Sigma V^\top=\sum_{i=1}^{r}\sigma_i\,\mathbf u_i\mathbf v_i^\top.$$

The $\mathbf v_i$ are eigenvectors of $A^\top A$ (eigenvalues $\sigma_i^2$), the
$\mathbf u_i$ eigenvectors of $AA^\top$, and $A\mathbf v_i=\sigma_i\mathbf u_i$.

**Relation to PCA.** If $A$ is column-centered, $\frac1nA^\top A$ is the covariance, so
$V$ are the principal directions and $\sigma_i^2/n$ the explained variances. Truncated
SVD = PCA *without the centering step*.

**Eckart–Young–Mirsky theorem.** Among all matrices $B$ of rank $\le k$,

$$\min_{\operatorname{rank}B\le k}\lVert A-B\rVert_F=\lVert A-A_k\rVert_F=\sqrt{\sum_{i>k}\sigma_i^2},
\qquad
\min_{\operatorname{rank}B\le k}\lVert A-B\rVert_2=\sigma_{k+1},$$

both achieved by $A_k=\sum_{i\le k}\sigma_i\mathbf u_i\mathbf v_i^\top$.

*Sketch (spectral norm):* for any rank-$k$ $B$, its null space (dim $\ge d-k$) meets
$\operatorname{span}\{\mathbf v_1,\dots,\mathbf v_{k+1}\}$ (dim $k+1$) in a nonzero
vector $\mathbf z$; on it $\lVert(A-B)\mathbf z\rVert=\lVert A\mathbf z\rVert\ge\sigma_{k+1}\lVert\mathbf z\rVert$,
so $\lVert A-B\rVert_2\ge\sigma_{k+1}$, with equality at $A_k$.

**Randomized SVD** (Halko–Martinsson–Tropp). Draw Gaussian $\Omega\in\mathbb R^{d\times(k+p)}$,
form the sketch $Y=A\Omega$, orthonormalize $Q=\operatorname{qr}(Y)$ (its columns
capture the dominant range of $A$), then SVD the small $B=Q^\top A$ and lift back
$U=QU_B$. **Power iterations** $Y\leftarrow A(A^\top Y)$ sharpen the spectrum so the
top directions dominate the sketch.

**LSA.** With $A$ a term–document matrix (TF-IDF weights), truncated SVD yields latent
"topics": columns of $V_k$ are document factors, rows of $U_k$ are term factors, and
synonyms/co-occurring terms collapse into the same latent dimensions.
"""),
        md("## 4. NumPy implementation (truncated SVD, low-rank approx, randomized SVD)"),
        show(MOD, "TruncatedSVDNumPy", "low_rank_approx", "randomized_svd"),
        md("## 5. PyTorch implementation (truncated SVD + SGD matrix factorization)"),
        show(MOD, "truncated_svd_torch", "low_rank_sgd_torch"),
        md("## 6. Train / run — spectrum, Eckart–Young error, randomized vs exact, LSA"),
        run_demo(MOD),
        md("## 7. Visualization — singular-value spectrum, rank-k error, 2D embedding"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import svd as M

from sklearn.datasets import load_digits
digits = load_digits()
rng = np.random.default_rng(0)
idx = rng.choice(len(digits.data), 300, replace=False)
A, y = digits.data[idx], digits.target[idx]

_, s, _ = np.linalg.svd(A, full_matrices=False)
ranks = range(1, 31)
errs = [np.sqrt((s[k:] ** 2).sum()) for k in ranks]      # Eckart-Young Frobenius
Z = M.TruncatedSVDNumPy(n_components=2).fit_transform(A)  # 2D embedding

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(range(1, len(s) + 1), s, "o-", ms=3)
ax[0].set_title("singular value spectrum"); ax[0].set_xlabel("i"); ax[0].set_ylabel(r"$\sigma_i$")
ax[1].plot(list(ranks), errs, "o-", ms=3)
ax[1].set_title(r"rank-$k$ Frobenius error"); ax[1].set_xlabel("k"); ax[1].set_ylabel(r"$\|A-A_k\|_F$")
sc = ax[2].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=14)
ax[2].set_title("digits via top-2 SVD"); ax[2].set_xlabel("SV1"); ax[2].set_ylabel("SV2")
plt.colorbar(sc, ax=ax[2], label="digit")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Truncated SVD is **PCA without centering** — ideal for **sparse** data (text/LSA)
  where centering would destroy sparsity.
- **Eckart–Young** guarantees the top-$k$ truncation is *optimal* — no other rank-$k$
  matrix is closer. Read the **singular-value spectrum** to pick $k$.
- The decomposition is unique up to **sign/rotation of degenerate singular values**;
  don't over-interpret individual vectors when $\sigma_i\approx\sigma_{i+1}$.
- For huge matrices use **randomized SVD** (or `scipy.sparse.linalg.svds`), not the
  full dense SVD.
- SGD matrix factorization reaches the **same optimum subspace** — and generalizes to
  missing entries (recommender systems) where plain SVD can't.
"""),
    ]

from tools.nbreg import register, md, code, show, run_demo

MOD = "ica"


@register("ica", "ml/dimensionality-reduction/ica.ipynb")
def build():
    return [
        md(r"""
# ICA — unmixing independent signals (FastICA)

> Tutorial pair for [`ica.py`](ica.py).

## 1. Intuition
Several microphones each record a different *mixture* of the same voices (the cocktail
party). ICA recovers the individual voices knowing **only** the mixtures — no info about
who sat where. The key insight: real signals are **non-Gaussian**, and any mixture of
them looks **more Gaussian** than the originals. So "un-mixing" = "making the outputs as
non-Gaussian (and independent) as possible".
"""),
        md(r"""
## 2. Concept (the slide)
- Model: observed $\mathbf x = A\mathbf s$, with unknown mixing matrix $A$ and
  statistically **independent** sources $\mathbf s$. Find $W\approx A^{-1}$ so
  $\hat{\mathbf s}=W\mathbf x$ are independent.
- **Why non-Gaussianity?** Central Limit Theorem: sums of independents drift toward
  Gaussian. Maximizing non-Gaussianity of projections recovers the sources.
- Measure non-Gaussianity by **negentropy**, approximated with nonlinear contrasts
  (logcosh, exp, kurtosis).
- **Whiten** first (decorrelate + unit variance) → the unmixing reduces to finding an
  **orthogonal** matrix, solved by a **fixed-point** iteration (FastICA).
- **Ambiguities:** sign and permutation of the recovered sources are arbitrary.
"""),
        md(r"""
## 3. Math derivation

**Model & goal.** $\mathbf x=A\mathbf s$; we seek $\mathbf y=W\mathbf x$ with components
as independent as possible. Independence $\Rightarrow$ each $y_i$ should be maximally
non-Gaussian (CLT argument).

**Negentropy.** For unit-variance $y$, negentropy
$J(y)=H(y_\text{gauss})-H(y)\ge0$ is zero **iff** $y$ is Gaussian — a principled
non-Gaussianity measure. It's hard to compute, so approximate with a smooth contrast $G$:

$$J(y)\;\approx\;\big[\,\mathbb E\{G(y)\}-\mathbb E\{G(\nu)\}\,\big]^2,\qquad \nu\sim\mathcal N(0,1).$$

Common choices ($g=G'$): $\,G=\frac1a\log\cosh(a u)\Rightarrow g=\tanh(au)$;
$\,G=-e^{-u^2/2}\Rightarrow g=u\,e^{-u^2/2}$; kurtosis $G=u^4/4\Rightarrow g=u^3$.

**Whitening (mandatory pre-step).** Center, then transform $\mathbf x_w=K\mathbf x$ so
$\mathbb E\{\mathbf x_w\mathbf x_w^\top\}=I$. From the covariance eigendecomposition
$C=E D E^\top$, take $K=D^{-1/2}E^\top$. After whitening the unmixing matrix is
**orthogonal**, shrinking the search space.

**FastICA fixed point.** Maximize $\mathbb E\{G(\mathbf w^\top\mathbf x_w)\}$ subject to
$\lVert\mathbf w\rVert=1$. The Lagrange/Newton step yields the celebrated update

$$\boxed{\;\mathbf w^+=\mathbb E\{\mathbf x_w\,g(\mathbf w^\top\mathbf x_w)\}-\mathbb E\{g'(\mathbf w^\top\mathbf x_w)\}\,\mathbf w,\qquad \mathbf w\leftarrow\mathbf w^+/\lVert\mathbf w^+\rVert\;}$$

It converges **cubically** for the kurtosis contrast — far faster than gradient ascent
(hence *Fast*ICA).

**Getting several components.**
- *Deflation:* find $\mathbf w$'s one at a time, **Gram–Schmidt**-orthogonalizing each
  new one against those already found:
  $\mathbf w\leftarrow\mathbf w-\sum_{j<i}(\mathbf w^\top\mathbf w_j)\mathbf w_j$.
- *Symmetric:* update all rows together, then **symmetric decorrelation**
  $W\leftarrow(WW^\top)^{-1/2}W$ (no component is privileged).

**Recover sources.** $\hat{\mathbf s}=W K(\mathbf x-\bar{\mathbf x})$. Note ICA fixes
neither the **sign** nor the **order** nor the **scale** of the sources.
"""),
        md("## 4. NumPy implementation (whitening + fixed-point, deflation & symmetric)"),
        show(MOD, "FastICANumPy"),
        md("## 5. PyTorch implementation (symmetric FastICA with torch.linalg)"),
        show(MOD, "fastica_torch"),
        md("## 6. Train / run — blind source separation vs PCA"),
        run_demo(MOD),
        md("## 7. Visualization — sources, mixtures, and recovered signals"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import ica as M

np.random.seed(0)
n = 2000; t = np.linspace(0, 8, n)
s1 = np.sin(2 * t); s2 = np.sign(np.sin(3 * t))
s3 = np.random.default_rng(0).uniform(-1, 1, n)
S = np.c_[s1, s2, s3]; S /= S.std(0)
A = np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])
X = S @ A.T
S_hat = M.FastICANumPy(n_components=3, fun="logcosh").fit_transform(X)

fig, ax = plt.subplots(3, 3, figsize=(13, 6), sharex=True)
for r, (data, title) in enumerate([(S, "sources"), (X, "mixtures"), (S_hat, "ICA recovered")]):
    for c in range(3):
        ax[r, c].plot(t[:400], data[:400, c], lw=0.8)
        if c == 0:
            ax[r, c].set_ylabel(title)
ax[0, 0].set_title("signal 1"); ax[0, 1].set_title("signal 2"); ax[0, 2].set_title("signal 3")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Whitening is mandatory** and turns ICA into a search over orthogonal matrices.
- ICA needs **non-Gaussian** sources — it **cannot** separate two Gaussians (their
  mixture is rotationally symmetric, so the directions are unidentifiable).
- **PCA ≠ ICA:** PCA only *decorrelates* (2nd-order); ICA enforces full statistical
  *independence* (higher-order) — the demo shows PCA failing to unmix.
- Outputs have **arbitrary sign, scale, and order** — match to ground truth by
  correlation when evaluating.
- Choose the contrast for robustness: **logcosh** is a good general default; pure
  **kurtosis** ($u^3$) is fast but sensitive to outliers.
"""),
    ]

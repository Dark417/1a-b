from tools.nbreg import register, md, code, show, run_demo

MOD = "glow"


@register("glow", "generative-models/normalizing-flows/glow.ipynb")
def build():
    return [
        md(r"""
# Glow — actnorm, invertible 1x1 convolutions, and affine coupling

> Tutorial pair for [`glow.py`](glow.py).

## 1. Intuition
RealNVP shuffles coordinates between coupling layers with a *fixed* mask. Glow
asks: why fix the permutation? Replace it with a **learned, invertible linear
mixing** of the channels -- an invertible $1\times1$ convolution -- so the model
itself decides how to route information. Glow also swaps batchnorm for
**actnorm**, a per-channel affine layer initialized from the first batch. Each
Glow step is *actnorm -> invertible 1x1 conv -> affine coupling*, and the whole
flow still has an exact, cheap log-likelihood.
"""),
        md(r"""
## 2. Concept (the slide)
- A Glow step = **actnorm** + **invertible 1x1 conv** + **affine coupling**.
- **ActNorm:** $y=(x-\mu)e^{\log s}$, with $\mu,s$ initialized so the first batch
  is zero-mean/unit-variance (data-dependent init). $\log|\det|=\sum\log s$.
- **Invertible 1x1 conv:** $y=Wx$ with learnable $W$; generalizes a permutation.
  $\log|\det|=\#\text{pixels}\cdot\log|\det W|$.
- **Affine coupling:** same triangular-Jacobian trick as RealNVP.
- Exact likelihood via change of variables; sample by running steps backward.
"""),
        md(r"""
## 3. Math derivation — the three log-dets

A flow's log-likelihood is, by change of variables,
$$\log p_X(x)=\log\mathcal N\big(f(x);0,I\big)+\sum_{k}\log\Big|\det\frac{\partial f_k}{\partial h_{k-1}}\Big|.$$
Each Glow component contributes one term.

**(a) ActNorm.** Per-channel affine $y=(x-\mu)\odot e^{\log s}$. The Jacobian is
diagonal, so
$$\log|\det J_{\text{actnorm}}|=\sum_c \log s_c \quad(\times\text{ spatial size}).$$
$\mu,\log s$ are *initialized from the first minibatch* so the activations start
normalized -- a normalization that, unlike batchnorm, is exact and batch-size
independent at test time.

**(b) Invertible $1\times1$ convolution.** A $1\times1$ conv with weight matrix
$W\in\mathbb R^{C\times C}$ mixes channels at each of the $H\cdot W$ spatial
locations: $y_{ij}=W x_{ij}$. Its Jacobian is block-diagonal with $H\cdot W$
identical blocks $W$, hence
$$\boxed{\,\log\big|\det J_{1\times1}\big|=H\cdot W\cdot\log|\det W|\,}.$$
Initializing $W$ as a random rotation (orthogonal) makes $\det W=\pm1$
(volume-preserving) and guarantees invertibility; the inverse pass uses $W^{-1}$.
For 2-D data here, $H=W=1$, $C=2$, so this is just a learned $2\times2$ mixing with
$\log|\det W|$.

**(c) Affine coupling.** Split, keep half, scale-and-shift the rest:
$y_b=x_b\odot e^{s(x_a)}+t(x_a)$. As in RealNVP the Jacobian is triangular, so
$$\log|\det J_{\text{coupling}}|=\sum_j s(x_a)_j.$$

**Putting it together.** One Glow step adds the three terms; a stack adds across
steps. The training objective is the negative exact log-likelihood
$$\mathcal L=-\frac1N\sum_i\Big[\log\mathcal N\big(f(x_i);0,I\big)+\textstyle\sum_k \log|\det J_k(x_i)|\Big],$$
and sampling runs every step's inverse: coupling$^{-1}$, then $W^{-1}$, then
actnorm$^{-1}$.
"""),
        md("## 4. Model — actnorm, invertible 1x1 conv, coupling, and a Glow step"),
        show(MOD, "ActNorm", "Inv1x1", "GlowStep", "Glow"),
        md("## 5. Training / sampling — exact log_prob, NLL training, inverse sampler"),
        show(MOD, "Glow"),
        md("## 6. Train & sample on 2-D two-moons"),
        run_demo(MOD),
        md("## 7. Visualization — learned density and samples"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt, torch
from sklearn.datasets import make_moons
import glow as M

X, _ = make_moons(2000, noise=0.05, random_state=0)
X = ((X - X.mean(0)) / X.std(0)).astype("float32")
m = M.Glow(dim=2, n_steps=6).fit(X, epochs=400)

gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 120), np.linspace(-2.5, 2.5, 120))
grid = np.c_[gx.ravel(), gy.ravel()].astype("float32")
with torch.no_grad():
    logp = m.log_prob(torch.tensor(grid)).numpy().reshape(gx.shape)
s = m.sample(2000)

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].contourf(gx, gy, np.exp(logp), levels=30, cmap="magma")
ax[0].scatter(X[:, 0], X[:, 1], s=3, alpha=.2, color="cyan")
ax[0].set_title("Exact learned density (Glow)"); ax[0].set_aspect("equal")
ax[1].scatter(X[:, 0], X[:, 1], s=4, alpha=.3, label="data")
ax[1].scatter(s[:, 0], s[:, 1], s=4, alpha=.4, color="r", label="Glow samples")
ax[1].legend(); ax[1].set_title("Samples (z~N(0,I) run backward)"); ax[1].set_aspect("equal")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Glow = RealNVP + two upgrades: a **learned** invertible $1\times1$ conv instead
  of a fixed permutation, and **actnorm** instead of batchnorm.
- The $1\times1$ conv's log-det is the clean $H W\log|\det W|$; orthogonal init
  keeps it invertible and stable.
- ActNorm's **data-dependent initialization** must happen once on a real batch
  before training (the code triggers it on the first forward pass).
- Pitfalls: $W$ can drift toward singular (det -> 0); the original paper uses an
  LU-parameterization for cheaper, stable log-dets in high channel counts.
- Like RealNVP, dimensionality is preserved -- flows are exact bijections, the
  price for an exact likelihood.
"""),
    ]

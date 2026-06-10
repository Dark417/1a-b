from tools.nbreg import register, md, code, show, run_demo

MOD = "realnvp"


@register("realnvp", "generative-models/normalizing-flows/realnvp.ipynb")
def build():
    return [
        md(r"""
# RealNVP — exact likelihoods with affine coupling layers

> Tutorial pair for [`realnvp.py`](realnvp.py).

## 1. Intuition
A normalizing flow builds a complicated distribution by pushing a simple one (a
Gaussian) through an **invertible** transformation. Because the map is invertible
and we can compute how it stretches space (its Jacobian determinant), the change-
of-variables formula gives the *exact* density of the data -- no bound (VAE) and
no adversary (GAN). RealNVP's trick is the **affine coupling layer**: transform
half the variables using the other half, so inversion and the Jacobian are both
trivial.
"""),
        md(r"""
## 2. Concept (the slide)
- **Flow:** $z=f(x)$ invertible; base $p_Z=\mathcal N(0,I)$.
- **Density:** $\log p_X(x)=\log p_Z(f(x))+\log|\det \partial f/\partial x|$.
- **Affine coupling:** split $x=(x_a,x_b)$; keep $x_a$; map
  $x_b\mapsto x_b\odot e^{s(x_a)}+t(x_a)$.
- The Jacobian is **triangular** $\Rightarrow$ $\log|\det|=\sum s(x_a)$ -- cheap.
- Stack couplings with **alternating masks** so every coordinate gets
  transformed; sample by running the stack backward.
"""),
        md(r"""
## 3. Math derivation — change of variables & the coupling Jacobian

**Change of variables.** If $z=f(x)$ is a diffeomorphism and $p_Z$ is the base
density, conservation of probability mass $p_X(x)|dx|=p_Z(z)|dz|$ gives
$$\boxed{\,\log p_X(x)=\log p_Z\big(f(x)\big)+\log\Big|\det\frac{\partial f}{\partial x}\Big|\,}.$$
For a composition $f=f_L\circ\cdots\circ f_1$ the log-dets simply add:
$\log|\det \partial f/\partial x|=\sum_k \log|\det \partial f_k/\partial h_{k-1}|$.

**Affine coupling layer.** Pick a binary mask $b$. Split $x=(x_a,x_b)$ with
$x_a=b\odot x$. Define
$$y_a=x_a,\qquad y_b=x_b\odot \exp\!\big(s(x_a)\big)+t(x_a),$$
where $s,t$ are arbitrary neural nets that see *only* $x_a$.

**Invertibility (no need to invert the net).** Given $y$, recover $x$ by
$$x_a=y_a,\qquad x_b=\big(y_b-t(x_a)\big)\odot\exp\!\big(-s(x_a)\big).$$
Because $s,t$ are evaluated at $x_a=y_a$ in both directions, we never invert the
networks themselves -- they can be arbitrarily complex.

**The Jacobian is triangular.** Order coordinates $(x_a,x_b)$. Then
$$\frac{\partial(y_a,y_b)}{\partial(x_a,x_b)}=
\begin{pmatrix} I & 0\\[2pt] \dfrac{\partial y_b}{\partial x_a} & \mathrm{diag}(e^{s(x_a)})\end{pmatrix}.$$
The matrix is block lower-triangular, so its determinant is the product of the
diagonal blocks; the off-diagonal $\partial y_b/\partial x_a$ is irrelevant:
$$\boxed{\,\log\Big|\det\frac{\partial y}{\partial x}\Big|=\sum_j s(x_a)_j\,}.$$
This is the whole reason RealNVP scales: the log-det that would normally cost
$O(d^3)$ is just a sum.

**Alternating masks.** A single coupling leaves $x_a$ untouched, so we stack
several and flip the mask each time; after a few layers every coordinate has been
both "kept" and "transformed", letting the flow model arbitrary couplings.

**Training.** Maximize the exact log-likelihood (equivalently minimize NLL):
$$\mathcal L=-\frac1N\sum_i\Big[\log\mathcal N\big(f(x_i);0,I\big)+\sum_k\log|\det J_k(x_i)|\Big].$$
"""),
        md("## 4. Model — affine coupling layer + the RealNVP stack"),
        show(MOD, "AffineCoupling", "RealNVP"),
        md("## 5. Training / sampling — exact log_prob, NLL training, inverse sampler"),
        show(MOD, "RealNVP"),
        md("## 6. Train & sample on 2-D two-moons (exact likelihood)"),
        run_demo(MOD),
        md("## 7. Visualization — learned density and samples"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt, torch
from sklearn.datasets import make_moons
import realnvp as M

X, _ = make_moons(2000, noise=0.05, random_state=0)
X = ((X - X.mean(0)) / X.std(0)).astype("float32")
m = M.RealNVP(dim=2, n_couplings=6).fit(X, epochs=400)

# evaluate the exact learned density on a grid
gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 120), np.linspace(-2.5, 2.5, 120))
grid = np.c_[gx.ravel(), gy.ravel()].astype("float32")
with torch.no_grad():
    logp = m.log_prob(torch.tensor(grid)).numpy().reshape(gx.shape)
s = m.sample(2000)

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].contourf(gx, gy, np.exp(logp), levels=30, cmap="viridis")
ax[0].scatter(X[:, 0], X[:, 1], s=3, alpha=.2, color="white")
ax[0].set_title("Exact learned density p(x)"); ax[0].set_aspect("equal")
ax[1].scatter(X[:, 0], X[:, 1], s=4, alpha=.3, label="data")
ax[1].scatter(s[:, 0], s[:, 1], s=4, alpha=.4, color="r", label="flow samples")
ax[1].legend(); ax[1].set_title("Samples (z~N(0,I) run backward)"); ax[1].set_aspect("equal")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Flows give an **exact** tractable likelihood -- the change-of-variables formula
  is the entire engine.
- The coupling layer's **triangular Jacobian** makes the log-det a cheap sum;
  this is what lets flows scale beyond toy dimensions.
- Use **alternating masks** so every coordinate is eventually transformed.
- Pitfalls: unbounded log-scales $s$ blow up -- bound them (here with
  $\tanh$); the dimension must be preserved (flows are bijections, so no
  bottleneck/compression).
- **Glow** (next file) generalizes the fixed mask into a learned invertible
  $1\times1$ convolution and adds actnorm.
"""),
    ]

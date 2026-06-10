from tools.nbreg import register, md, code, show, run_demo

MOD = "score_based"


@register("score_based", "generative-models/diffusion/score_based.ipynb")
def build():
    return [
        md(r"""
# Score-based generative models — learn the gradient of log-density

> Tutorial pair for [`score_based.py`](score_based.py).

## 1. Intuition
Forget about modelling the probability density $p(x)$ directly (it needs an
intractable normalizing constant). Instead learn its **score**, the vector field
$\nabla_x\log p(x)$ that points "uphill" toward higher-probability regions. If you
know which way is uphill everywhere, you can generate samples by repeatedly
stepping uphill and adding a dash of noise -- **Langevin dynamics**. To make this
robust where data is sparse, blur the data at several noise levels and learn the
score of each.
"""),
        md(r"""
## 2. Concept (the slide)
- **Score** $s_\theta(x)\approx\nabla_x\log p(x)$ -- no normalizing constant
  needed.
- **Denoising score matching (DSM):** add Gaussian noise; the score of the
  noised conditional is known in closed form, so it becomes a simple regression.
- **Multiple noise scales** $\sigma_1>\dots>\sigma_L$: one network conditioned on
  $\sigma$ (NCSN).
- **Annealed Langevin** sampling: start at the largest $\sigma$, follow the score
  with noise, anneal $\sigma$ down to (near) zero.
"""),
        md(r"""
## 3. Math derivation -- DSM and Langevin

**Score matching goal.** We want $s_\theta(x)\approx\nabla_x\log p_{\text{data}}(x)$.
The explicit objective $\tfrac12\mathbb E_p\|s_\theta-\nabla\log p\|^2$ needs the
unknown true score. Hyvarinen's integration-by-parts gives an equivalent form
needing only $s_\theta$ and its divergence, but the trace of the Jacobian is
costly.

**Denoising score matching (the practical trick).** Perturb $x$ with
$q_\sigma(\tilde x\mid x)=\mathcal N(\tilde x;x,\sigma^2 I)$. Its score is *exact*:
$$\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)
 =\frac{x-\tilde x}{\sigma^2}=-\frac{\epsilon}{\sigma},
 \qquad \tilde x=x+\sigma\epsilon,\ \epsilon\sim\mathcal N(0,I).$$
Vincent (2011) proved that matching the score of the *joint*-perturbed density
$p_\sigma(\tilde x)=\int q_\sigma(\tilde x\mid x)p(x)\,dx$ reduces to
$$\boxed{\;\mathcal L_\sigma=\mathbb E_{x,\epsilon}\Big\|s_\theta(\tilde x,\sigma)+\tfrac{\epsilon}{\sigma}\Big\|^2\;}$$
i.e. regress the score onto $-\epsilon/\sigma$. Multiplying inside by $\sigma$
(a $\lambda(\sigma)=\sigma^2$ weighting that balances scales) gives the numerically
nicer $\big\|\sigma\, s_\theta+\epsilon\big\|^2$, which is what the code uses.

**Why multiple scales.** At small $\sigma$ the score is accurate near the data
manifold but garbage far from it (where samples start). Large $\sigma$ gives
useful gradients everywhere but a blurry target. Training across
$\sigma_1>\dots>\sigma_L$ and annealing bridges the two.

**Langevin dynamics.** The SDE $dx=\tfrac12\nabla\log p(x)\,dt+dW$ has $p$ as its
stationary distribution. Discretizing with step $\alpha$:
$$x\leftarrow x+\tfrac{\alpha}{2}\,s_\theta(x,\sigma)+\sqrt{\alpha}\,z,\qquad z\sim\mathcal N(0,I).$$
**Annealed** Langevin runs this inner loop for each $\sigma$ from large to small,
scaling the step as $\alpha_i\propto\sigma_i^2$.

**Connection to diffusion.** This is the continuous-time twin of DDPM: the
forward noising is an SDE, the DDPM/Langevin samplers are discretizations, and the
deterministic DDIM sampler is the corresponding *probability-flow ODE*.
"""),
        md("## 4. Model -- noise-conditional score network"),
        show(MOD, "ScoreNet", "ScoreModel"),
        md("## 5. Training / sampling -- DSM loss + annealed Langevin"),
        show(MOD, "ScoreModel"),
        md("## 6. Train & sample on 2-D two-moons"),
        run_demo(MOD),
        md("## 7. Visualization -- learned score field and Langevin samples"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt, torch
from sklearn.datasets import make_moons
import score_based as M

X, _ = make_moons(2000, noise=0.05, random_state=0)
X = ((X - X.mean(0)) / X.std(0)).astype("float32")
m = M.ScoreModel(data_dim=2, n_scales=10).fit(X, epochs=400)

# score vector field at the smallest noise scale
gx, gy = np.meshgrid(np.linspace(-2.5, 2.5, 20), np.linspace(-2.5, 2.5, 20))
grid = np.c_[gx.ravel(), gy.ravel()].astype("float32")
with torch.no_grad():
    sig = torch.full((len(grid),), float(m.sigmas[-1]))
    sc = m.net(torch.tensor(grid), sig).numpy()

s = m.sample(2000, n_steps=50, step=2e-4)
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].quiver(gx, gy, sc[:, 0].reshape(gx.shape), sc[:, 1].reshape(gx.shape),
             angles="xy", scale=60, alpha=.7)
ax[0].scatter(X[:, 0], X[:, 1], s=4, alpha=.2, color="C0")
ax[0].set_title("Learned score field (small sigma)"); ax[0].set_aspect("equal")
ax[1].scatter(X[:, 0], X[:, 1], s=4, alpha=.2, label="data")
ax[1].scatter(s[:, 0], s[:, 1], s=4, alpha=.4, color="r", label="Langevin samples")
ax[1].legend(); ax[1].set_title("Annealed Langevin samples"); ax[1].set_aspect("equal")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Learning the **score** sidesteps the normalizing constant entirely.
- DSM turns score matching into a trivial regression onto the known noise
  direction $-\epsilon/\sigma$.
- A single noise scale fails: scores are unreliable off the data manifold.
  Multiple scales + annealing fix coverage and mixing.
- Sampling pitfalls: step size too large diverges, too small mixes slowly; the
  $\alpha\propto\sigma^2$ schedule matters.
- Score-based SDEs, DDPM, and DDIM are three views of the same object.
"""),
    ]

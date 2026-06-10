from tools.nbreg import register, md, code, show, run_demo

MOD = "ddpm"


@register("ddpm", "generative-models/diffusion/ddpm.ipynb")
def build():
    return [
        md(r"""
# Denoising Diffusion Probabilistic Models (DDPM) — learn to undo noise

> Tutorial pair for [`ddpm.py`](ddpm.py).

## 1. Intuition
Take a data point and slowly stir in Gaussian noise until, after many steps, it
is indistinguishable from static. That destruction is *easy* and needs no
learning. The hard, useful direction is the reverse: a network learns to remove
a little noise at a time. Sampling then means starting from pure noise and
running the learned denoiser backwards until a clean sample emerges.
"""),
        md(r"""
## 2. Concept (the slide)
- **Forward process** $q$: fixed, gradually adds Gaussian noise over $T$ steps.
- **Reverse process** $p_\theta$: learned, removes noise step by step.
- Each $q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I)$.
- A magic identity gives $q(x_t\mid x_0)$ in **closed form**, so we can jump to
  any noise level in one shot during training.
- The variational bound simplifies to predicting the **noise** $\epsilon$ with a
  plain MSE loss.
"""),
        md(r"""
## 3. Math derivation — from the VLB to a simple MSE

**Forward marginal in closed form.** Let $\alpha_t=1-\beta_t$ and
$\bar\alpha_t=\prod_{s\le t}\alpha_s$. Composing Gaussians,
$$\boxed{\,q(x_t\mid x_0)=\mathcal N\!\big(\sqrt{\bar\alpha_t}\,x_0,\;(1-\bar\alpha_t)I\big)\,}
\quad\Longleftrightarrow\quad
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ \epsilon\sim\mathcal N(0,I).$$
This is the same reparameterization trick as the VAE: noise is an external input.

**Tractable posterior.** Bayes on the Gaussian chain gives the reverse
*conditional* in closed form too:
$$q(x_{t-1}\mid x_t,x_0)=\mathcal N\big(\tilde\mu_t(x_t,x_0),\tilde\beta_t I\big),
\quad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t .$$

**Variational bound.** Maximizing $\log p_\theta(x_0)$ uses the ELBO
$$\mathbb E_q\Big[\underbrace{\mathrm{KL}(q(x_T|x_0)\,\|\,p(x_T))}_{L_T}
 +\sum_{t>1}\underbrace{\mathrm{KL}(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t))}_{L_{t-1}}
 -\underbrace{\log p_\theta(x_0|x_1)}_{L_0}\Big].$$
Each $L_{t-1}$ is a KL between two Gaussians, i.e. a squared difference of means.

**The simplification.** Parameterize the reverse mean through a noise predictor
$\epsilon_\theta(x_t,t)$:
$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big).$$
Substituting $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ into the
$L_{t-1}$ means and dropping the $t$-dependent weights yields the famous
**simplified objective**:
$$\boxed{\,L_{\text{simple}}=\mathbb E_{x_0,\,t,\,\epsilon}\big\|\epsilon-\epsilon_\theta(x_t,t)\big\|^2\,}.$$
Just regress the network onto the noise you added.

**Reverse sampler (ancestral).** Sample $x_T\sim\mathcal N(0,I)$ and iterate
$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)+\sqrt{\beta_t}\,z,
\quad z\sim\mathcal N(0,I)\ (z=0\text{ at }t=0).$$
"""),
        md("## 4. Model — sinusoidal time embedding + eps-prediction MLP"),
        show(MOD, "EpsMLP", "DDPM"),
        md("## 5. Training / sampling — closed-form q_sample, MSE loss, ancestral sampler"),
        show(MOD, "DDPM"),
        md("## 6. Train & sample on 2-D two-moons"),
        run_demo(MOD),
        md("## 7. Visualization — the forward noising process and learned samples"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt, torch
from sklearn.datasets import make_moons
import ddpm as M

X, _ = make_moons(2000, noise=0.05, random_state=0)
X = ((X - X.mean(0)) / X.std(0)).astype("float32")
m = M.DDPM(data_dim=2, T=200).fit(X, epochs=400)

# forward process: x_0 -> x_t for increasing t (closed form)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
xt0 = torch.tensor(X)
for ax, t in zip(axes, [0, 25, 75, 150, 199]):
    tt = torch.full((len(X),), t, dtype=torch.long)
    eps = torch.randn_like(xt0)
    xt = m.q_sample(xt0, tt, eps).numpy()
    ax.scatter(xt[:, 0], xt[:, 1], s=4, alpha=.3)
    ax.set_title(f"q(x_t|x_0), t={t}"); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

# reverse: generated samples vs data
s = m.sample(2000)
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], s=5, alpha=.3, label="data")
plt.scatter(s[:, 0], s[:, 1], s=5, alpha=.4, color="r", label="DDPM samples")
plt.legend(); plt.title("Reverse process samples vs data"); plt.axis("equal")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- The closed-form $q(x_t\mid x_0)$ is what makes training cheap: pick a random
  $t$, noise the sample in one step, regress the noise.
- The simplified MSE loss is an (unweighted) variational bound — it just works.
- Many steps $T$ make *training* easy but *sampling* slow (one network call per
  step). **DDIM** (next file) keeps the same trained network but samples in far
  fewer, deterministic steps.
- Pitfalls: the data should be standardized; too few steps or a bad $\beta$
  schedule leaves residual noise in samples.
"""),
    ]

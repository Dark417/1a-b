from tools.nbreg import register, md, code, show, run_demo

MOD = "wgan"


@register("wgan", "generative-models/gan/wgan.ipynb")
def build():
    return [
        md(r"""
# WGAN / WGAN-GP — replace JS divergence with the Wasserstein distance

> Tutorial pair for [`wgan.py`](wgan.py).

## 1. Intuition
When the real and fake distributions barely overlap (always true early in
training), the Jensen–Shannon divergence the vanilla GAN minimizes is locally
*constant* — the generator gets no gradient. The **Wasserstein-1 (earth-mover)
distance** measures the cost of *moving mass* from one distribution to the other
and varies smoothly even for disjoint supports, so it always points the generator
somewhere useful.
"""),
        md(r"""
## 2. Concept (the slide)
- The discriminator becomes a **critic** $f$: an unbounded real-valued score, not
  a probability (no sigmoid).
- The critic must be **1-Lipschitz**. Two enforcements:
  - **Weight clipping** (original WGAN): box-clip all critic weights to
    $[-c, c]$. Crude but works; pair with RMSProp.
  - **Gradient penalty** (WGAN-GP): softly pin $\|\nabla f\|_2$ to 1 on points
    interpolated between real and fake. Use Adam, no BatchNorm in the critic.
- Train the critic `n_critic` times per generator step so it stays near optimal.
"""),
        md(r"""
## 3. Math derivation — Wasserstein distance and its dual

The Wasserstein-1 distance between $p_{\text{data}}$ and $p_g$ is
$$W(p_{\text{data}},p_g)=\inf_{\gamma\in\Pi(p_{\text{data}},p_g)}
 \mathbb E_{(x,y)\sim\gamma}\,\|x-y\|,$$
an infimum over couplings $\gamma$ with the right marginals. This primal is
intractable, but **Kantorovich–Rubinstein duality** rewrites it as a maximization
over 1-Lipschitz functions:
$$W(p_{\text{data}},p_g)=\sup_{\|f\|_L\le 1}
 \mathbb E_{x\sim p_{\text{data}}}[f(x)]-\mathbb E_{x\sim p_g}[f(x)].$$

The critic $f$ approximates the optimal witness function. This gives the WGAN
objective (a min–max, like the GAN, but with a *different* value function):
$$\min_G\max_{\|f\|_L\le 1}\;
 \mathbb E_{x\sim p_{\text{data}}}[f(x)]-\mathbb E_{z\sim p_z}[f(G(z))].$$

**Why this beats JS.** For disjoint supports the JS divergence is the constant
$\log 2$, so $\nabla_G\,\mathrm{JSD}=0$; but $W$ scales with the *distance* between
supports, giving a non-vanishing $\nabla_G W$ everywhere.

**Enforcing $\|f\|_L\le 1$.**
- *Weight clipping:* constrain weights to $[-c,c]$ so $f$ is Lipschitz — but it
  biases the critic toward simple functions and can cause exploding/vanishing
  gradients.
- *Gradient penalty (WGAN-GP):* a 1-Lipschitz $f$ has $\|\nabla_x f(x)\|_2\le 1$,
  and the optimal critic has unit-norm gradient almost everywhere on the optimal
  coupling's support. So add
  $$\lambda\,\mathbb E_{\hat x}\big[(\|\nabla_{\hat x} f(\hat x)\|_2-1)^2\big],
  \qquad \hat x=\epsilon\,x+(1-\epsilon)\,G(z),\ \epsilon\sim U[0,1],$$
  penalizing deviation of the gradient norm from 1 on the lines between real and
  fake points. The reported $\mathbb E[f(\text{real})]-\mathbb E[f(\text{fake})]$
  is a running **estimate of $W$**.
"""),
        md("## 4. Generator / key component"),
        show(MOD, "Generator", "Critic"),
        md("## 5. Trainer / losses"),
        show(MOD, "WGANTorch"),
        md("## 6. Train"),
        run_demo(MOD),
        md("## 7. Visualization"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import wgan as M

real = M.make_ring(2000)
gan = M.WGANTorch(clip="gp").fit(real, steps=600, batch=128)
fake = gan.generate(1000)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(real[:, 0], real[:, 1], s=6, alpha=.3, label="real")
ax[0].scatter(fake[:, 0], fake[:, 1], s=6, alpha=.5, color="r", label="fake")
ax[0].set_title("Real vs generated (WGAN-GP)"); ax[0].legend(); ax[0].set_aspect("equal")
ax[1].plot(gan.w_hist, alpha=.6)
ax[1].set_xlabel("critic step"); ax[1].set_ylabel(r"$E[f(real)]-E[f(fake)]$")
ax[1].set_title("Wasserstein estimate (rises then plateaus)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- WGAN swaps the JS-divergence value function for the **Wasserstein distance** via
  Kantorovich–Rubinstein duality, giving usable gradients for disjoint supports.
- The critic must be **1-Lipschitz**; gradient penalty is smoother and more robust
  than weight clipping.
- Pitfalls: **no BatchNorm in the GP critic** (it breaks the per-sample gradient
  penalty); too-large clip values destroy the Lipschitz constraint; too-few
  `n_critic` steps leave the $W$-estimate unreliable.
"""),
    ]

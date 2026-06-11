from tools.nbreg import register, md, code, show, run_demo

MOD = "vanilla_gan"


@register("vanilla_gan", "03.generative-models/gan/vanilla_gan.ipynb")
def build():
    return [
        md(r"""
# Generative Adversarial Networks — learning by competition

> Tutorial pair for [`vanilla_gan.py`](vanilla_gan.py).

## 1. Intuition
A forger (Generator) makes fake samples from noise; a detective (Discriminator)
judges real vs fake. Each pushes the other to improve. At the ideal equilibrium
the forgeries are perfect and the detective is reduced to guessing (50/50).
"""),
        md(r"""
## 2. Concept (the slide)
- **Generator** $G(z)$: noise $z\sim\mathcal N(0,I)\to$ a sample.
- **Discriminator** $D(x)\in(0,1)$: probability that $x$ is real.
- **Game:** $D$ maximizes its accuracy, $G$ minimizes it → a *minimax* objective.
- **Practical loss:** the non-saturating generator loss to avoid early vanishing
  gradients. Watch for **mode collapse**.
"""),
        md(r"""
## 3. Math derivation — the minimax game

$$\min_G\max_D\; V(D,G)=\mathbb E_{x\sim p_{\text{data}}}[\log D(x)]
 +\mathbb E_{z\sim p_z}[\log(1-D(G(z)))].$$

**Optimal discriminator (fix $G$).** Pointwise maximize
$p_{\text{data}}(x)\log D + p_g(x)\log(1-D)$ over $D\in(0,1)$:
$$\frac{p_{\text{data}}}{D}-\frac{p_g}{1-D}=0
 \;\Rightarrow\;\boxed{D^\star(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}}.$$

**What $G$ then minimizes.** Substituting $D^\star$ back gives
$$V(D^\star,G)=-\log 4+2\,\mathrm{JSD}\!\big(p_{\text{data}}\,\Vert\,p_g\big),$$
the **Jensen–Shannon divergence**. It is minimized iff $p_g=p_{\text{data}}$ —
the generator matches the data distribution.

**The saturating-gradient problem.** Early on, fakes are obvious so $D(G(z))\approx0$
and $\log(1-D(G(z)))$ is **flat** — $G$ gets almost no gradient. Fix: instead of
minimizing $\log(1-D(G(z)))$, **maximize $\log D(G(z))$** (the *non-saturating*
loss). Same fixed point, strong gradients when $G$ is losing. This is exactly the
generator step coded in the module.

**Failure modes.** Non-convex, two moving objectives → oscillation; **mode
collapse** (G maps everything to one good sample). The demo counts how many of 8
ground-truth modes are covered.
"""),
        md("## 4. NumPy implementation — G, D, and the adversarial loop by hand"),
        show(MOD, "GANNumPy"),
        md("## 5. PyTorch implementation — a reusable GAN base (for DCGAN/WGAN/cGAN)"),
        show(MOD, "GANTorch"),
        md("## 6. Train on a 2-D ring of 8 Gaussians (mode-collapse is visible)"),
        run_demo(MOD),
        md("## 7. Visualization — real vs generated, and the adversarial losses"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
import vanilla_gan as M

real = M.make_ring(2000)
gan = M.GANTorch(lr=2e-4).fit(real, epochs=3000)
fake = gan.generate(1000)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(real[:,0], real[:,1], s=6, alpha=.3, label="real")
ax[0].scatter(fake[:,0], fake[:,1], s=6, alpha=.5, label="fake", color="r")
ax[0].set_title("Real vs generated"); ax[0].legend(); ax[0].set_aspect("equal")
ax[1].plot(gan.d_hist, label="D loss", alpha=.7)
ax[1].plot(gan.g_hist, label="G loss", alpha=.7)
ax[1].set_xlabel("step"); ax[1].set_title("Adversarial losses oscillate"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- The optimal $D$ turns the game into minimizing JS divergence between $p_g$ and
  the data.
- Use the **non-saturating** G loss for usable gradients.
- GAN training is unstable → **WGAN / WGAN-GP** replace JS with the Wasserstein
  distance for smoother gradients; **DCGAN** sets good conv architecture defaults;
  **conditional GAN** adds labels. Those are the next files under `gan/` (MAP.md).
"""),
    ]

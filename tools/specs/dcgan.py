from tools.nbreg import register, md, code, show, run_demo

MOD = "dcgan"


@register("dcgan", "03.generative-models/gan/dcgan.ipynb")
def build():
    return [
        md(r"""
# DCGAN — convolutions make GAN training stable

> Tutorial pair for [`dcgan.py`](dcgan.py).

## 1. Intuition
Vanilla GANs on images with fully-connected nets are fragile. **DCGAN** keeps
the exact same adversarial *game* but swaps the architecture for convolutions
and a recipe of choices (strided convs instead of pooling, BatchNorm, ReLU in
G / LeakyReLU in D, Tanh output) that empirically tames the unstable two-player
optimization. Here we shrink it to tiny 1x16x16 images so it runs in seconds.
"""),
        md(r"""
## 2. Concept (the slide)
- **Generator** upsamples a noise vector to an image with `ConvTranspose2d`:
  $1\times1 \to 4\times4 \to 8\times8 \to 16\times16$.
- **Discriminator** mirrors it with strided `Conv2d` down to a single logit.
- **DCGAN guidelines:** no pooling (use strided / fractional-strided convs),
  BatchNorm in both nets (except D's input and G's output), ReLU in G with a
  **Tanh** output, LeakyReLU in D, weights $\sim\mathcal N(0, 0.02)$.
- The **loss is unchanged** from vanilla GAN — DCGAN is an *architecture* result.
"""),
        md(r"""
## 3. Math derivation — same game, better-conditioned gradients

DCGAN optimizes the **same** non-saturating GAN objective as the vanilla model.
The discriminator maximizes
$$\mathcal L_D=\mathbb E_{x\sim p_{\text{data}}}[\log D(x)]
 +\mathbb E_{z\sim p_z}[\log(1-D(G(z)))],$$
and the generator uses the non-saturating loss
$$\mathcal L_G=-\,\mathbb E_{z\sim p_z}\big[\log D(G(z))\big].$$
With the optimal discriminator $D^\star(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$
this still drives $p_g\to p_{\text{data}}$ via the Jensen–Shannon divergence.

**So what does DCGAN change?** Not the objective but the *parameterization* of
$G$ and $D$, which changes the conditioning of the gradients:

- A **transposed convolution** with stride $s$ is the adjoint of a strided
  convolution; it learns its upsampling kernel rather than fixing it, so $G$ can
  synthesize spatial structure directly.
- **BatchNorm** normalizes each pre-activation $\hat h=\frac{h-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$
  then rescales $\gamma\hat h+\beta$. This keeps activations in a well-scaled
  regime, preventing one network from collapsing the other early — the dominant
  GAN failure mode. It is omitted at D's input and G's Tanh output to avoid
  constraining the data statistics.
- **Tanh** at G's output matches data scaled to $[-1,1]$, giving bounded,
  symmetric gradients.

Net effect: the JS-divergence game is the same, but $\nabla_\theta\mathcal L$ is
far better conditioned, so training converges where the MLP GAN oscillates.
"""),
        md("## 4. Generator / key component"),
        show(MOD, "Generator"),
        md("## 5. Trainer / losses"),
        show(MOD, "DCGANTorch"),
        md("## 6. Train"),
        run_demo(MOD),
        md("## 7. Visualization"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import dcgan as M

real = M.make_images(256)
gan = M.DCGANTorch().fit(real, steps=200, batch=64)  # lighter retrain just for the picture
fake = gan.generate(8)

fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
for j in range(8):
    axes[0, j].imshow(real[j, 0], cmap="gray", vmin=-1, vmax=1); axes[0, j].axis("off")
    axes[1, j].imshow(fake[j, 0], cmap="gray", vmin=-1, vmax=1); axes[1, j].axis("off")
axes[0, 0].set_ylabel("real", rotation=0, labelpad=20); axes[1, 0].set_ylabel("fake", rotation=0, labelpad=20)
fig.suptitle("DCGAN: real (top) vs generated (bottom)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- DCGAN is an **architecture** contribution: the game (JS divergence) is identical
  to vanilla GAN, but strided convs + BatchNorm + Tanh make gradients well-behaved.
- Pitfalls: BatchNorm on D's *input* or G's *output* hurts; forgetting to scale
  data to $[-1,1]$ fights the Tanh; too-large LR still triggers mode collapse.
- Next steps: **WGAN/WGAN-GP** change the *objective* for smoother gradients;
  **conditional GAN** adds labels for controllable generation.
"""),
    ]

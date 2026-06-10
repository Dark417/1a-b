from tools.nbreg import register, md, code, show, run_demo

MOD = "pix2pix"


@register("pix2pix", "generative-models/gan/pix2pix.ipynb")
def build():
    return [
        md(r"""
# Pix2Pix — paired image-to-image translation with a conditional GAN

> Tutorial pair for [`pix2pix.py`](pix2pix.py).

## 1. Intuition
Given **aligned pairs** (input image, target image), learn the mapping. A pure
L1/L2 regression produces blurry outputs (it averages over plausible targets); a
pure GAN produces sharp but unfaithful ones. Pix2Pix combines them: a
**conditional GAN** for realism plus an **L1** term for fidelity to the specific
target. The discriminator is a **PatchGAN** that judges local patches, which
sharpens textures.
"""),
        md(r"""
## 2. Concept (the slide)
- **Generator** $G$: a small **U-Net** (encoder-decoder with skip connections)
  mapping input $a$ to output $G(a)$; skips carry low-level structure across.
- **Discriminator** $D$ is **conditional**: it sees the pair $(a, b)$ and judges
  realism. As a **PatchGAN** it outputs a grid of logits, one per local
  receptive field, so it enforces realism at the patch level.
- **Loss:** conditional adversarial + $\lambda$-weighted L1 reconstruction.
- Toy task here: map a filled rectangle to its photometric inverse on 1x16x16.
"""),
        md(r"""
## 3. Math derivation — conditional GAN objective + L1

**Conditional adversarial loss.** Both nets are conditioned on the input $a$:
$$\mathcal L_{\text{cGAN}}(G,D)=\mathbb E_{a,b}\big[\log D(a,b)\big]
 +\mathbb E_{a}\big[\log\big(1-D(a,G(a))\big)\big].$$
$D$ maximizes this (learn real pairs vs generated pairs); $G$ minimizes it.
Conditioning on $a$ is the key difference from vanilla GAN: $D$ judges not "is
this a real image?" but "is this a real *translation of $a$*?".

**L1 reconstruction.** Add a term tying $G(a)$ to the true target:
$$\mathcal L_{\text{L1}}(G)=\mathbb E_{a,b}\big[\lVert b-G(a)\rVert_1\big].$$
L1 over L2 because it penalizes errors linearly, encouraging **sharper** outputs
(L2 over-smooths by averaging). It captures low-frequency structure; the
adversarial term supplies high-frequency detail.

**Full objective.**
$$G^\star=\arg\min_{G}\max_{D}\;\mathcal L_{\text{cGAN}}(G,D)
 +\lambda\,\mathcal L_{\text{L1}}(G),\qquad \lambda\approx 100.$$

**PatchGAN.** Rather than one global decision, $D$ outputs an $N\times N$ grid of
logits, each classifying a patch; the loss averages over the grid. This models
the image as a Markov random field of patch-realism, gives sharper textures, and
uses far fewer parameters than a full-image discriminator.

**Why it differs from vanilla GAN.** Vanilla GAN maps noise$\to$image
unconditionally. Pix2Pix maps input$\to$output deterministically (noise is mostly
dropped), and the L1 term plus conditional PatchGAN give faithful, sharp pairs.
"""),
        md("## 4. Generator / key component"),
        show(MOD, "UNetGenerator"),
        md("## 5. Trainer / losses"),
        show(MOD, "Pix2PixTorch"),
        md("## 6. Train"),
        run_demo(MOD),
        md("## 7. Visualization"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import pix2pix as M

inp, tgt = M.make_pairs(256)
gan = M.Pix2PixTorch().fit(inp, tgt, steps=250, batch=32)  # lighter retrain just for the picture
ti, tt = M.make_pairs(8, seed=123)
pred = gan.generate(ti)

fig, axes = plt.subplots(3, 8, figsize=(12, 4.6))
rows = [(ti, "input"), (pred, "G(input)"), (tt, "target")]
for r, (imgs, name) in enumerate(rows):
    for j in range(8):
        axes[r, j].imshow(imgs[j, 0], cmap="gray", vmin=-1, vmax=1)
        axes[r, j].axis("off")
    axes[r, 0].set_title(name, loc="left")
fig.suptitle("Pix2Pix: input -> generated -> ground-truth target")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Pix2Pix = conditional GAN (realism) + L1 (fidelity). L1 alone blurs; GAN alone
  drifts; together they give sharp, correct translations.
- PatchGAN judges local patches for sharper textures with fewer parameters;
  the U-Net's skip connections preserve spatial structure.
- Pitfalls: needs **paired** data (use CycleGAN otherwise); $\lambda$ too small
  drifts from the target, too large reverts to blurry regression; output noise is
  usually injected via dropout, not a noise vector.
"""),
    ]

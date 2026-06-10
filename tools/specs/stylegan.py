from tools.nbreg import register, md, code, show, run_demo

MOD = "stylegan"


@register("stylegan", "generative-models/gan/stylegan.ipynb")
def build():
    return [
        md(r"""
# StyleGAN (simplified) — style-based synthesis with AdaIN

> Tutorial pair for [`stylegan.py`](stylegan.py).

## 1. Intuition
A vanilla generator pipes the latent code straight into the first layer, so all
factors of variation are tangled at the input. StyleGAN instead turns the latent
into a **style** that *modulates every layer*. It (1) maps $z$ to a disentangled
space $w$, (2) injects $w$ at each synthesis layer via **AdaIN**, and (3) adds
per-pixel **noise** for stochastic detail. Coarse layers control coarse
attributes, fine layers fine ones — giving the famous layer-wise control.
"""),
        md(r"""
## 2. Concept (the slide)
- **Mapping network** $f:z\mapsto w$ — an MLP that disentangles the latent
  (after normalizing $z$ onto the hypersphere).
- **Learned constant input** — synthesis starts from a learned $4\times4$ tensor,
  *not* from $z$.
- **AdaIN** — each layer's activations are instance-normalized then re-styled by
  a scale/bias predicted from $w$.
- **Noise injection** — per-pixel Gaussian noise with a learned per-channel scale
  adds texture the style need not encode.
- The discriminator is ordinary. Toy data: tiny 1x16x16 images.
"""),
        md(r"""
## 3. Math derivation — AdaIN and style-based synthesis

**Mapping network.** First normalize $z$ (PixelNorm onto the hypersphere),
$\hat z = z/\sqrt{\frac1d\sum_i z_i^2+\epsilon}$, then
$$w=f(\hat z),\qquad f:\mathcal Z\to\mathcal W,$$
an 8-layer MLP (4 here). The intermediate space $\mathcal W$ need not match the
fixed prior of $\mathcal Z$, so it can **unwarp** the data manifold and
disentangle factors of variation.

**AdaIN (adaptive instance normalization).** For a feature map $x_i$ of channel
$i$, normalize per-instance then apply a style-derived affine transform:
$$\mathrm{AdaIN}(x_i, w)=\gamma_i(w)\,\frac{x_i-\mu(x_i)}{\sigma(x_i)}+\beta_i(w),$$
where $\mu, \sigma$ are computed over the spatial dimensions of that instance and
$(\gamma(w),\beta(w))=A\,w$ is a learned affine map ("style"). Each layer
**discards the previous statistics** (via the normalization) and overwrites them
with the style, which is why a single $w$ exerts global, layer-localized control.
(In code we predict a residual scale $1+\gamma$ for stability.)

**Noise injection.** Before activation, add scaled noise:
$$x \leftarrow x + \psi\odot n,\qquad n\sim\mathcal N(0,I)\text{ per pixel},$$
with a learned per-channel scale $\psi$. This models stochastic detail
independently of the style, freeing $w$ to encode high-level structure.

**Synthesis layer.** Each block is
$$x \leftarrow \mathrm{LReLU}\big(\mathrm{AdaIN}(\,\text{noise}(\mathrm{conv}(\mathrm{up}(x)))\,,\,w)\big).$$
Training still uses the **non-saturating GAN loss**
$\mathcal L_G=-\mathbb E_z[\log D(G(z))]$, with $D$ minimizing the usual BCE.

**Why it differs from vanilla GAN.** The *objective* is unchanged; the
*generator design* changes — latent disentanglement ($\mathcal W$), per-layer
AdaIN style modulation, a learned constant input, and noise injection — yielding
controllable, higher-quality synthesis.
"""),
        md("## 4. Generator / key component"),
        show(MOD, "MappingNetwork", "AdaIN", "StyleGenerator"),
        md("## 5. Trainer / losses"),
        show(MOD, "StyleGANTorch"),
        md("## 6. Train"),
        run_demo(MOD),
        md("## 7. Visualization"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import stylegan as M

real = M.make_images(256)
gan = M.StyleGANTorch().fit(real, steps=150, batch=32)  # lighter retrain just for the picture
fake = gan.generate(8)

fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
for j in range(8):
    axes[0, j].imshow(real[j, 0], cmap="gray", vmin=-1, vmax=1); axes[0, j].axis("off")
    axes[1, j].imshow(fake[j, 0], cmap="gray", vmin=-1, vmax=1); axes[1, j].axis("off")
axes[0, 0].set_title("real", loc="left"); axes[1, 0].set_title("style-generated", loc="left")
fig.suptitle("StyleGAN (simplified): real (top) vs style-based synthesis (bottom)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- StyleGAN keeps the GAN game but redesigns $G$: $z\to w$ mapping, AdaIN style
  modulation at every layer, a learned constant input, and noise injection.
- AdaIN is the workhorse — normalize, then overwrite statistics with a
  $w$-derived scale/bias, giving layer-localized control over attributes.
- Pitfalls: AdaIN's normalization can create "blob" artifacts (StyleGAN2 replaces
  it with weight demodulation); the mapping network needs enough depth to
  disentangle; noise injection helps texture but can dominate if its scale grows.
"""),
    ]

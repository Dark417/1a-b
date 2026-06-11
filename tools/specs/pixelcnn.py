from tools.nbreg import register, md, code, show, run_demo

MOD = "pixelcnn"


@register("pixelcnn", "03.generative-models/autoregressive/pixelcnn.ipynb")
def build():
    return [
        md(r"""
# PixelCNN — autoregressive images with masked convolutions

> Tutorial pair for [`pixelcnn.py`](pixelcnn.py).

## 1. Intuition
Read an image like text: left-to-right, top-to-bottom. PixelCNN predicts each
pixel from only the pixels that came before it in this raster scan. The clever
part is doing this with ordinary convolutions: by **masking** the kernel so it
can never look at the current or future pixels, one convolution computes every
pixel's conditional distribution at once during training -- yet generation still
proceeds strictly one pixel at a time.
"""),
        md(r"""
## 2. Concept (the slide)
- Factor the image likelihood as a product of per-pixel conditionals.
- Enforce that ordering with **masked convolutions**.
- **Mask A** (first layer): a pixel may not see its own value -> no cheating.
- **Mask B** (deeper layers): a pixel may see its own *feature* (which already
  only summarizes the past), so information still never flows backward.
- Train by maximum likelihood (teacher forcing); sample pixel by pixel.
"""),
        md(r"""
## 3. Math derivation — factorization & masking

**Autoregressive factorization.** Any joint distribution factorizes exactly via
the chain rule. Order the $N=H\cdot W$ pixels in raster scan $x_1,\dots,x_N$:
$$\boxed{\,p(x)=\prod_{i=1}^{N}p\big(x_i\mid x_1,\dots,x_{i-1}\big)\,}.$$
No approximation -- this is just conditional probability. The model only needs to
parameterize each conditional. For binary pixels we use a Bernoulli with logit
$\ell_i=f_\theta(x_{<i})$:
$$p(x_i\mid x_{<i})=\sigma(\ell_i)^{x_i}\,(1-\sigma(\ell_i))^{1-x_i}.$$

**Log-likelihood / loss.** Negative log-likelihood is a sum of per-pixel binary
cross-entropies, which is *exact* (unlike the VAE's bound):
$$-\log p(x)=\sum_{i=1}^{N}\big[-x_i\log\sigma(\ell_i)-(1-x_i)\log(1-\sigma(\ell_i))\big].$$

**Why masking enforces the ordering.** A conv output at pixel $i$ must depend only
on $x_{<i}$. Zero the kernel weights covering the current and future positions:
for a $k\times k$ kernel with center $c=\lfloor k/2\rfloor$,
$$M[r,:]=0\ \text{for } r>c,\qquad M[c,\,c{+}1{:}]=0,$$
and for the **type-A** mask additionally $M[c,c]=0$. The first conv uses mask A
(it touches raw pixel values, so it must exclude $x_i$ itself); all deeper convs
use **type B**, where the center channel is a *feature* of $x_{<i}$, so allowing
$M[c,c]=1$ is safe and gives the network access to its own receptive field.
Because every layer respects causality, the composition does too: a single forward
pass yields all $\ell_i$ with **no leakage** from later pixels (teacher forcing).

**Generation.** Sampling cannot be parallel: draw $x_1\sim p(x_1)$, feed it back,
draw $x_2\sim p(x_2\mid x_1)$, and so on for all $N$ pixels.
"""),
        md("## 4. Model — masked convolution + the PixelCNN stack"),
        show(MOD, "MaskedConv2d", "PixelCNN"),
        md("## 5. Training / sampling — exact NLL loss + raster-order sampler"),
        show(MOD, "PixelCNN"),
        md("## 6. Train & sample on binarized 8×8 digits"),
        run_demo(MOD),
        md("## 7. Visualization — the type A/B masks and generated digits"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import pixelcnn as M

# the causal masks themselves
maskA = M._make_mask_numpy(5, "A")
maskB = M._make_mask_numpy(5, "B")
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
for a, (m, t) in zip(ax, [(maskA, "mask A"), (maskB, "mask B")]):
    a.imshow(m, cmap="gray", vmin=0, vmax=1); a.set_title(t)
    a.set_xticks(range(5)); a.set_yticks(range(5))
plt.tight_layout(); plt.show()

# train and sample
X = load_digits().data.reshape(-1, 1, 8, 8) / 16.0
X = (X > 0.3).astype("float32")
m = M.PixelCNN(channels=32, n_layers=4).fit(X, epochs=25)
samples = m.sample(16)

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, a in enumerate(axes.ravel()):
    a.imshow(samples[i, 0], cmap="gray"); a.axis("off")
fig.suptitle("PixelCNN samples (generated pixel by pixel)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- The autoregressive factorization is *exact*: PixelCNN gives a true (tractable)
  log-likelihood, unlike VAEs (a bound) or GANs (none).
- The **mask** is the whole idea -- get mask A vs B wrong and the model either
  cheats (sees the answer) or loses its own receptive field.
- **Blind spot**: a naive masked conv cannot see some pixels above-right; the
  Gated PixelCNN fixes this with separate horizontal/vertical stacks.
- Sampling is inherently sequential and slow ($O(N)$ forward passes); training is
  fully parallel.
- This discrete autoregressive model is exactly what is used as the **prior over
  VQ-VAE codes** to turn that discrete encoder into a generator.
"""),
    ]

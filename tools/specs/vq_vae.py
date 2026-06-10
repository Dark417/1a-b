from tools.nbreg import register, md, code, show, run_demo

MOD = "vq_vae"


@register("vq_vae", "generative-models/vae/vq_vae.ipynb")
def build():
    return [
        md(r"""
# Vector-Quantized VAE (VQ-VAE) — a *discrete* latent space

> Tutorial pair for [`vq_vae.py`](vq_vae.py).

## 1. Intuition
A plain VAE has a *continuous* Gaussian latent. VQ-VAE instead keeps a small
**codebook** of $K$ learnable vectors and snaps every encoder output to its
nearest codebook entry. The latent becomes a grid of **integers** (which code
won), which is perfect for later modelling with a discrete autoregressive prior
(PixelCNN, transformers). The catch: "pick the nearest code" is an $\arg\min$ —
it has no gradient — so we need a trick to train through it.
"""),
        md(r"""
## 2. Concept (the slide)
- **Encoder** $z_e=E(x)\in\mathbb R^D$.
- **Codebook** $\{e_1,\dots,e_K\}\subset\mathbb R^D$.
- **Quantize:** $k=\arg\min_j\|z_e-e_j\|$, output $z_q=e_k$.
- **Decoder** reconstructs $\hat x=D(z_q)$.
- **Straight-through estimator (STE):** in the backward pass pretend the
  quantizer is the identity, $\partial z_q/\partial z_e:=1$, so the decoder's
  gradient is copied straight onto the encoder.
- Two extra losses pull the codebook and the encoder together; an optional
  **EMA** update replaces the codebook loss with a running average.
"""),
        md(r"""
## 3. Math derivation — the VQ objective & straight-through gradient

**Quantization.** With codebook $E\in\mathbb R^{K\times D}$,
$$k=\operatorname*{arg\,min}_j\|z_e-e_j\|_2^2,\qquad z_q=e_k.$$
The squared distance is expanded the cheap way,
$\|z_e-e_j\|^2=\|z_e\|^2-2\,z_e^\top e_j+\|e_j\|^2$, so the whole $(N,K)$ table is
one matmul.

**The gradient problem.** $z_q$ is a piecewise-constant function of $z_e$, so
$\partial z_q/\partial z_e=0$ almost everywhere and the encoder would never learn.

**Straight-through estimator.** Define the forward value as $z_q$ but *route the
gradient around* the quantizer. Algebraically,
$$\boxed{\,z_q^{\text{st}}=z_e+\operatorname{sg}[\,e_k-z_e\,]\,}$$
where $\operatorname{sg}[\cdot]$ is the stop-gradient. Numerically
$z_q^{\text{st}}=e_k$, but $\partial z_q^{\text{st}}/\partial z_e=1$, so
$\partial L/\partial z_e=\partial L/\partial z_q$.

**Full objective.** Reconstruction plus two vector-quantization terms:
$$\mathcal L=\underbrace{\log p(x\mid z_q)}_{\text{recon}}
 +\underbrace{\|\operatorname{sg}[z_e]-e_k\|_2^2}_{\text{codebook loss}}
 +\;\beta\underbrace{\|z_e-\operatorname{sg}[e_k]\|_2^2}_{\text{commitment loss}}.$$
- The **codebook loss** moves the chosen code $e_k$ toward the encoder output
  (it only sees $e_k$, since $z_e$ is detached).
- The **commitment loss** moves the encoder output toward its code so the
  encoder "commits" and the embeddings do not grow unboundedly; $\beta\approx0.25$.

**EMA variant.** Instead of learning the codebook by gradient descent, treat each
code as the mean of the encoder vectors assigned to it, maintained online:
$$N_k\leftarrow\gamma N_k+(1-\gamma)n_k,\qquad
  m_k\leftarrow\gamma m_k+(1-\gamma)\!\!\sum_{i:k_i=k}\!\!z_{e,i},\qquad
  e_k=\frac{m_k}{N_k},$$
with Laplace smoothing on $N_k$ to revive dead codes. This drops the codebook
loss term and is usually more stable.

**Why no KL?** With a uniform categorical prior over $K$ codes the KL to the
posterior is the constant $\log K$, so it vanishes from the gradient — the prior
is instead *learned afterwards* by an autoregressive model over the code indices.
"""),
        md("## 4. Model — encoder, vector quantizer (STE), decoder"),
        show(MOD, "VectorQuantizer", "VQVAE"),
        md("## 5. Training / sampling — reconstruct & inspect codebook usage"),
        show(MOD, "VQVAE"),
        md("## 6. Train & sample on 8×8 digits (loss-codebook vs EMA)"),
        run_demo(MOD),
        md("## 7. Visualization — reconstructions and codebook-usage histogram"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import vq_vae as M

X = (load_digits().data / 16.0).astype("float32")
m = M.VQVAE(64, num_codes=32, dim=16).fit(X, epochs=40)
xhat, idx = m.reconstruct(X[:512])

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axes[0, i].imshow(X[i].reshape(8, 8), cmap="gray")
    axes[1, i].imshow(xhat[i].reshape(8, 8), cmap="gray")
for a in axes.ravel():
    a.axis("off")
axes[0, 0].set_title("real", loc="left"); axes[1, 0].set_title("recon", loc="left")
plt.tight_layout(); plt.show()

plt.figure(figsize=(7, 3))
plt.hist(idx, bins=np.arange(33) - 0.5, rwidth=0.9)
plt.xlabel("codebook index"); plt.ylabel("count")
plt.title(f"Codebook usage ({len(np.unique(idx))}/32 codes active)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- The **straight-through estimator** is the whole game: it lets gradients skip a
  non-differentiable $\arg\min$ by defining the backward pass as the identity.
- **Codebook collapse**: only a few codes get used. Fixes — EMA updates with
  Laplace smoothing, reinitializing dead codes, smaller codebook, or normalizing
  embeddings.
- The commitment weight $\beta$ trades encoder "commitment" against flexibility;
  $0.25$ is the standard default.
- VQ-VAE itself is *not* a generator — it learns a discrete code. To *sample* you
  train an autoregressive prior (PixelCNN / transformer) over the code indices,
  which connects directly to the autoregressive models in this repo.
"""),
    ]

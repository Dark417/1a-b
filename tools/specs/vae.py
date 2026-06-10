from tools.nbreg import register, md, code, show, run_demo

MOD = "vae"


@register("vae", "generative-models/vae/vae.ipynb")
def build():
    return [
        md(r"""
# Variational Autoencoders — generative modeling with a tractable bound

> Tutorial pair for [`vae.py`](vae.py).

## 1. Intuition
An ordinary autoencoder compresses $x\to z\to \hat x$ but its latent space is a
mess — you can't sample from it. A VAE forces the latents to look like a simple
prior $\mathcal N(0,I)$, so afterwards you can **draw $z$ from noise and decode it
into new data**. The price is learning a *distribution* per input, not a point.
"""),
        md(r"""
## 2. Concept (the slide)
- **Encoder** $q_\phi(z|x)=\mathcal N(\mu_\phi(x),\sigma^2_\phi(x))$.
- **Decoder** $p_\theta(x|z)$ (Bernoulli for pixels in $[0,1]$).
- **Prior** $p(z)=\mathcal N(0,I)$.
- Train by maximizing the **ELBO** = reconstruction quality − KL(latent ‖ prior).
- The **reparameterization trick** makes the sampling step differentiable.
"""),
        md(r"""
## 3. Math derivation — the ELBO

We want $\log p_\theta(x)$ but it's intractable. Introduce $q_\phi(z|x)$ and use
Jensen / a KL identity:

$$\log p_\theta(x)=\underbrace{\mathbb E_{q_\phi}\!\big[\log p_\theta(x|z)\big]-\mathrm{KL}\!\big(q_\phi(z|x)\,\Vert\,p(z)\big)}_{\text{ELBO }\mathcal L(\theta,\phi;x)}
 +\underbrace{\mathrm{KL}\!\big(q_\phi(z|x)\,\Vert\,p_\theta(z|x)\big)}_{\ge 0}.$$

Since the last term is $\ge 0$, the ELBO is a **lower bound** on $\log p(x)$;
maximizing it both fits the data (first term) and pulls the posterior toward the
prior (second term).

**Closed-form KL** for two Gaussians ($q=\mathcal N(\mu,\sigma^2)$, $p=\mathcal N(0,I)$):
$$\mathrm{KL}=-\tfrac12\sum_{j}\big(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\big).$$

**Reparameterization trick.** We can't backprop through $z\sim\mathcal N(\mu,\sigma^2)$
directly. Rewrite the sample as a deterministic function of an *external* noise:
$$\boxed{\,z=\mu+\sigma\odot\epsilon,\quad \epsilon\sim\mathcal N(0,I)\,}$$
Now $z$ depends smoothly on $(\mu,\sigma)$, with $\partial z/\partial\mu=1$ and
$\partial z/\partial\sigma=\epsilon$, so gradients flow into the encoder. This one
trick is what makes the VAE trainable end-to-end.

**$\beta$-VAE.** Scale the KL by $\beta>1$ to push latents toward independence
(disentanglement), trading reconstruction for a cleaner latent space.
"""),
        md("## 4. NumPy implementation — encoder, decoder, reparam, full backward"),
        show(MOD, "VAENumPy"),
        md("## 5. PyTorch implementation (+ conditional VAE)"),
        show(MOD, "VAETorch"),
        md("## 6. Train on 8×8 digits — vanilla / Torch / conditional"),
        run_demo(MOD),
        md("## 7. Visualization — reconstructions and samples from the prior"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt, torch
from sklearn.datasets import load_digits
import vae as M

X = (load_digits().data / 16.0).astype("float32")
m = M.VAETorch(64, latent=2).fit(X, epochs=40)

# reconstructions
with torch.no_grad():
    xb = torch.tensor(X[:8])
    xhat, _, _ = m(xb)
    samples = m.decode(torch.randn(8, 2)).numpy()

fig, axes = plt.subplots(3, 8, figsize=(12, 4.5))
for i in range(8):
    axes[0,i].imshow(X[i].reshape(8,8), cmap="gray")
    axes[1,i].imshow(xhat[i].numpy().reshape(8,8), cmap="gray")
    axes[2,i].imshow(samples[i].reshape(8,8), cmap="gray")
for a in axes.ravel(): a.axis("off")
axes[0,0].set_title("real", loc="left"); axes[1,0].set_title("recon", loc="left")
axes[2,0].set_title("sampled z~N(0,I)", loc="left")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- ELBO = reconstruction − KL; the KL is a *regularizer* toward the prior.
- The reparameterization trick is the key enabler (reused in diffusion models).
- **Posterior collapse**: if the decoder is too strong, $q\to p(z)$ and latents
  go unused — KL annealing / $\beta<1$ early helps.
- VAEs give blurry samples (Gaussian/Bernoulli likelihood); **GANs** trade the
  likelihood for sharpness → next file.
"""),
    ]

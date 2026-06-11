# `03.generative-models/`

Models that learn a data distribution $p(x)$ and can sample from it.

| Sub-folder | Algorithms |
|---|---|
| `gan/` | vanilla GAN, DCGAN, WGAN(-GP), conditional, LSGAN, InfoGAN, CycleGAN, pix2pix, StyleGAN |
| `vae/` | VAE (ELBO, reparameterization), β-VAE, CVAE, VQ-VAE |
| `diffusion/` | DDPM, DDIM, score-based |
| `autoregressive/` | PixelCNN, PixelRNN |
| `normalizing-flows/` | RealNVP, Glow |

Each major **GAN variant gets its own file** under `gan/` (per `AGENTS.md` §4).
Training techniques highlighted here: **adversarial/minimax training** (GANs)
and the **reparameterization trick** (VAEs).

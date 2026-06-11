"""
StyleGAN (simplified)
=====================
StyleGAN redesigns the generator around the idea of *styles*. Instead of feeding
the latent code straight into the first conv layer, it (1) passes the latent
$z$ through a **mapping network** to an intermediate space $w$, which disentangles
factors of variation; (2) injects $w$ into *every* synthesis layer as a **style**
that modulates activations via **AdaIN** (adaptive instance normalization); and
(3) adds **per-pixel noise** at each layer to supply stochastic detail (texture)
the style does not need to encode. The discriminator is an ordinary one. This
file implements the faithful *mechanism* scaled down to tiny 1x16x16 images: a
mapping MLP, two synthesis blocks each with a learned constant input, AdaIN style
modulation, and noise injection. The demo does a fast forward pass and a few
training steps on CPU.

Variants implemented here:
    - Mapping network z -> w (the disentangled latent space W)
    - Style-based synthesis with AdaIN modulation per layer
    - Per-layer learned-scale noise injection; learned constant input

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - ADAPTIVE INSTANCE NORMALIZATION (AdaIN) as style modulation
    - Noise injection for stochastic detail; non-saturating GAN loss

References:
    - Karras, Laine & Aila (2019), "A Style-Based Generator Architecture for
      Generative Adversarial Networks" (StyleGAN)
    - Huang & Belongie (2017), "Arbitrary Style Transfer ... AdaIN"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SEED = 0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Tiny synthetic image data: 1x16x16 patches with a smooth radial gradient and a
# bright square in a random corner (same family as the DCGAN toy). [-1, 1] range.
# ---------------------------------------------------------------------------
def make_images(n: int = 256, size: int = 16, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2
    radial = -((xx - cx) ** 2 + (yy - cy) ** 2)
    radial = radial / radial.min()
    imgs = np.empty((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        img = 0.3 * radial.copy()
        c = rng.integers(0, 4)
        h = size // 3
        ys = 0 if c < 2 else size - h
        xs = 0 if c % 2 == 0 else size - h
        img[ys:ys + h, xs:xs + h] += 0.7
        img += 0.03 * rng.standard_normal((size, size)).astype(np.float32)
        imgs[i, 0] = np.clip(img, 0, 1)
    return (imgs * 2 - 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Mapping network: z -> w. A small MLP that produces the disentangled latent w.
# ---------------------------------------------------------------------------
class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int = 32, w_dim: int = 32, layers: int = 4):
        super().__init__()
        net = []
        d = z_dim
        for _ in range(layers):
            net += [nn.Linear(d, w_dim), nn.LeakyReLU(0.2, True)]
            d = w_dim
        self.net = nn.Sequential(*net)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Normalize z onto the hypersphere (PixelNorm), as in StyleGAN.
        z = z * torch.rsqrt(z.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        return self.net(z)


# ---------------------------------------------------------------------------
# Noise injection: add per-pixel Gaussian noise scaled by a learned per-channel
# weight. Supplies stochastic detail that the style need not encode.
# ---------------------------------------------------------------------------
class NoiseInjection(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


# ---------------------------------------------------------------------------
# AdaIN: normalize each feature map per-instance, then apply a style-derived
# scale (gamma) and bias (beta) computed from w by an affine layer.
#     AdaIN(x, y) = y_scale * (x - mu(x)) / sigma(x) + y_bias
# ---------------------------------------------------------------------------
class AdaIN(nn.Module):
    def __init__(self, w_dim: int, channels: int):
        super().__init__()
        self.affine = nn.Linear(w_dim, channels * 2)  # -> (scale, bias) per channel
        self.channels = channels

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        style = self.affine(w).view(x.size(0), 2, self.channels, 1, 1)
        scale, bias = style[:, 0], style[:, 1]
        mu = x.mean(dim=(2, 3), keepdim=True)
        sigma = x.std(dim=(2, 3), keepdim=True) + 1e-8
        return (1 + scale) * (x - mu) / sigma + bias


# ---------------------------------------------------------------------------
# A synthesis block: (optional upsample) conv -> noise -> AdaIN -> activation.
# ---------------------------------------------------------------------------
class SynthesisBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, w_dim: int, upsample: bool):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.noise = NoiseInjection(out_ch)
        self.adain = AdaIN(w_dim, out_ch)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        x = self.noise(x)
        x = self.adain(x, w)
        return self.act(x)


# ---------------------------------------------------------------------------
# Style-based generator: learned constant 4x4 input, two synthesis blocks
# (4->8->16) each modulated by the SAME w (a single style here), then to RGB/gray.
# ---------------------------------------------------------------------------
class StyleGenerator(nn.Module):
    def __init__(self, z_dim: int = 32, w_dim: int = 32, ch: int = 32, img_ch: int = 1):
        super().__init__()
        self.z_dim = z_dim
        self.mapping = MappingNetwork(z_dim, w_dim)
        # Learned constant input (StyleGAN starts synthesis from a constant tensor).
        self.const = nn.Parameter(torch.randn(1, ch, 4, 4))
        self.adain0 = AdaIN(w_dim, ch)
        self.noise0 = NoiseInjection(ch)
        self.block1 = SynthesisBlock(ch, ch, w_dim, upsample=True)   # 4 -> 8
        self.block2 = SynthesisBlock(ch, ch, w_dim, upsample=True)   # 8 -> 16
        self.to_img = nn.Sequential(nn.Conv2d(ch, img_ch, 1), nn.Tanh())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        x = self.const.expand(z.size(0), -1, -1, -1)
        x = self.adain0(self.noise0(x), w)         # style the constant input
        x = self.block1(x, w)
        x = self.block2(x, w)
        return self.to_img(x)


# ---------------------------------------------------------------------------
# Discriminator: a plain strided-conv classifier (no style mechanism).
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, ndf: int = 32, img_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),       # 16->8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1), nn.LeakyReLU(0.2, True),      # 8->4
            nn.Conv2d(ndf * 2, 1, 4, 1, 0),                                # ->1 logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), 1)


# ---------------------------------------------------------------------------
# Trainer (non-saturating GAN loss).
# ---------------------------------------------------------------------------
class StyleGANTorch:
    def __init__(self, z_dim: int = 32, lr: float = 2e-4):
        torch.manual_seed(SEED)
        self.dev = get_device()
        self.z_dim = z_dim
        self.G = StyleGenerator(z_dim).to(self.dev)
        self.D = Discriminator().to(self.dev)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.0, 0.99))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.0, 0.99))
        self.bce = nn.BCEWithLogitsLoss()

    def fit(self, real: np.ndarray, steps: int = 250, batch: int = 32):
        real = torch.as_tensor(real, dtype=torch.float32, device=self.dev)
        self.d_hist, self.g_hist = [], []
        ones = torch.ones(batch, 1, device=self.dev)
        zeros = torch.zeros(batch, 1, device=self.dev)
        for _ in range(steps):
            x = real[torch.randint(0, len(real), (batch,), device=self.dev)]
            # --- D step ---
            z = torch.randn(batch, self.z_dim, device=self.dev)
            fake = self.G(z).detach()
            lossD = self.bce(self.D(x), ones) + self.bce(self.D(fake), zeros)
            self.optD.zero_grad(); lossD.backward(); self.optD.step()
            # --- G step (non-saturating) ---
            z = torch.randn(batch, self.z_dim, device=self.dev)
            lossG = self.bce(self.D(self.G(z)), ones)
            self.optG.zero_grad(); lossG.backward(); self.optG.step()
            self.d_hist.append(lossD.item()); self.g_hist.append(lossG.item())
        return self

    @torch.no_grad()
    def generate(self, n: int) -> np.ndarray:
        self.G.eval()
        z = torch.randn(n, self.z_dim, device=self.dev)
        out = self.G(z).cpu().numpy()
        self.G.train()
        return out


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    real = make_images(256)
    print(f"data: {real.shape}, range [{real.min():.2f}, {real.max():.2f}]")

    gan = StyleGANTorch()
    # Fast forward pass sanity check (the style-based synthesis mechanism).
    sample = gan.generate(4)
    print(f"forward pass OK: generated {sample.shape}, range "
          f"[{sample.min():.2f}, {sample.max():.2f}]")

    gan.fit(real, steps=250, batch=32)
    print(f"D loss trend: {np.mean(gan.d_hist[:30]):.3f} -> {np.mean(gan.d_hist[-30:]):.3f}")
    print(f"G loss trend: {np.mean(gan.g_hist[:30]):.3f} -> {np.mean(gan.g_hist[-30:]):.3f}")

    fake = gan.generate(64)
    print(f"real pixel mean={real.mean():.3f} std={real.std():.3f}")
    print(f"fake pixel mean={fake.mean():.3f} std={fake.std():.3f} "
          f"(approaching real => style synthesis is learning structure)")


if __name__ == "__main__":
    demo()

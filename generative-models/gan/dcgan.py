"""
Deep Convolutional GAN (DCGAN)
==============================
DCGAN replaces the MLP generator/discriminator of a vanilla GAN with
convolutional architectures and a set of architectural guidelines that made
GAN training markedly more stable: strided / fractionally-strided (transpose)
convolutions instead of pooling, BatchNorm in both nets, ReLU in the generator
(Tanh at the output) and LeakyReLU in the discriminator, and no fully-connected
hidden layers. Here we scale the idea down to tiny 1x16x16 synthetic images so
the whole demo runs in a few seconds on CPU.

Variants implemented here:
    - DCGAN generator (ConvTranspose2d stack, BatchNorm, ReLU, Tanh output)
    - DCGAN discriminator (strided Conv2d stack, BatchNorm, LeakyReLU)
    - Non-saturating adversarial loss (same game as vanilla GAN)

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - BATCH NORMALIZATION to stabilize the two-network game
    - The DCGAN architectural guidelines (strided convs, no pooling, Tanh out)

References:
    - Radford, Metz & Chintala (2016), "Unsupervised Representation Learning
      with Deep Convolutional Generative Adversarial Networks"
    - Goodfellow et al. (2014), "Generative Adversarial Networks"
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
# Tiny synthetic image data: 1x16x16 patches with a smooth radial gradient and
# a bright square in a random corner. Pixel values in [-1, 1] to match Tanh.
# ---------------------------------------------------------------------------
def make_images(n: int = 256, size: int = 16, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2
    radial = -((xx - cx) ** 2 + (yy - cy) ** 2)
    radial = radial / radial.min()  # normalize roughly to [0, 1]
    imgs = np.empty((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        img = 0.3 * radial.copy()
        # bright square in one of 4 corners (the "structure" G must learn)
        c = rng.integers(0, 4)
        h = size // 3
        ys = 0 if c < 2 else size - h
        xs = 0 if c % 2 == 0 else size - h
        img[ys:ys + h, xs:xs + h] += 0.7
        img += 0.03 * rng.standard_normal((size, size)).astype(np.float32)
        imgs[i, 0] = np.clip(img, 0, 1)
    return (imgs * 2 - 1).astype(np.float32)  # -> [-1, 1]


# ---------------------------------------------------------------------------
# DCGAN generator: project noise to a small feature map, then upsample with
# ConvTranspose2d. 4x4 -> 8x8 -> 16x16. BatchNorm + ReLU; Tanh at the output.
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim: int = 32, ngf: int = 32, img_ch: int = 1):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            # 1x1 -> 4x4
            nn.ConvTranspose2d(noise_dim, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # 4x4 -> 8x8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(ngf, img_ch, 4, 2, 1, bias=False),
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), self.noise_dim, 1, 1))


# ---------------------------------------------------------------------------
# DCGAN discriminator: mirror of G. Strided convs (no pooling), LeakyReLU,
# BatchNorm (except the first layer, per the DCGAN guidelines). Outputs a logit.
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, ndf: int = 32, img_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            # 16x16 -> 8x8 (no BatchNorm on the input layer)
            nn.Conv2d(img_ch, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            # 8x8 -> 4x4
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            # 4x4 -> 1x1 logit
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), 1)  # logits


def _weights_init(m: nn.Module) -> None:
    """DCGAN init: weights ~ N(0, 0.02), BatchNorm gamma ~ N(1, 0.02)."""
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class DCGANTorch:
    def __init__(self, noise_dim: int = 32, img_size: int = 16, lr: float = 2e-4):
        torch.manual_seed(SEED)
        self.dev = get_device()
        self.noise_dim, self.img_size = noise_dim, img_size
        self.G = Generator(noise_dim).to(self.dev).apply(_weights_init)
        self.D = Discriminator().to(self.dev).apply(_weights_init)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.bce = nn.BCEWithLogitsLoss()

    def fit(self, real: np.ndarray, steps: int = 400, batch: int = 64):
        real = torch.as_tensor(real, dtype=torch.float32, device=self.dev)
        self.d_hist, self.g_hist = [], []
        for _ in range(steps):
            idx = torch.randint(0, len(real), (batch,), device=self.dev)
            x = real[idx]
            # --- D step: real -> 1, fake -> 0 ---
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            fake = self.G(z).detach()
            lossD = self.bce(self.D(x), torch.ones(batch, 1, device=self.dev)) \
                + self.bce(self.D(fake), torch.zeros(batch, 1, device=self.dev))
            self.optD.zero_grad(); lossD.backward(); self.optD.step()
            # --- G step: non-saturating, label fakes as real ---
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            lossG = self.bce(self.D(self.G(z)), torch.ones(batch, 1, device=self.dev))
            self.optG.zero_grad(); lossG.backward(); self.optG.step()
            self.d_hist.append(lossD.item()); self.g_hist.append(lossG.item())
        return self

    @torch.no_grad()
    def generate(self, n: int) -> np.ndarray:
        self.G.eval()
        z = torch.randn(n, self.noise_dim, device=self.dev)
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

    gan = DCGANTorch().fit(real, steps=400, batch=64)
    fake = gan.generate(64)

    d0, d1 = np.mean(gan.d_hist[:50]), np.mean(gan.d_hist[-50:])
    print(f"D loss trend: {d0:.3f} -> {d1:.3f}")
    print(f"G loss trend: {np.mean(gan.g_hist[:50]):.3f} -> {np.mean(gan.g_hist[-50:]):.3f}")

    # quality proxy: per-pixel mean/std of fakes should approach the real data's
    print(f"real  pixel mean={real.mean():.3f} std={real.std():.3f}")
    print(f"fake  pixel mean={fake.mean():.3f} std={fake.std():.3f}")
    # the bright corner square should give fakes high-intensity corners
    corner = fake[:, 0, :5, :5].mean()
    print(f"fake top-left 5x5 mean={corner:.3f} (real corners are bright)")


if __name__ == "__main__":
    demo()

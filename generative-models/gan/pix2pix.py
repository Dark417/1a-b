"""
Pix2Pix
=======
Pix2Pix is a *conditional* GAN for **paired** image-to-image translation: given
an input image, produce its corresponding output (edges->photo, label-map->scene,
etc.). Unlike CycleGAN it assumes aligned (input, target) pairs. Two ideas make
it work well: (1) a **conditional discriminator** that sees both the input and a
candidate output and judges whether the pair is real, implemented as a
**PatchGAN** that classifies overlapping local patches rather than the whole
image (sharper textures, fewer parameters); and (2) an **L1 reconstruction loss**
that ties the generator's output to the ground-truth target, capturing the
low-frequency structure the adversarial loss alone tends to miss. The generator
is a small **U-Net** with skip connections. Here we map a filled shape to its
"inverse" on tiny 1x16x16 images so the demo runs in seconds on CPU.

Variants implemented here:
    - U-Net generator (encoder-decoder with skip connections)
    - PatchGAN conditional discriminator (judges input||output pairs)
    - cGAN adversarial loss + L1 reconstruction loss

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - L1 reconstruction loss alongside the adversarial loss
    - PatchGAN: local patch realism instead of a single global decision
    - Skip connections (U-Net) to preserve spatial detail

References:
    - Isola et al. (2017), "Image-to-Image Translation with Conditional
      Adversarial Networks" (pix2pix)
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
# Tiny PAIRED image data: input = a filled rectangle in a random location;
# target = its photometric inverse (1 - input). 1x16x16, values in [-1, 1].
# ---------------------------------------------------------------------------
def make_pairs(n: int = 256, size: int = 16, seed: int = SEED):
    rng = np.random.default_rng(seed)
    inp = np.zeros((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        h = rng.integers(4, 8); w = rng.integers(4, 8)
        ys = rng.integers(0, size - h); xs = rng.integers(0, size - w)
        inp[i, 0, ys:ys + h, xs:xs + w] = 1.0
    tgt = 1.0 - inp                      # the deterministic target mapping
    inp = inp * 2 - 1; tgt = tgt * 2 - 1  # -> [-1, 1]
    return inp.astype(np.float32), tgt.astype(np.float32)


# ---------------------------------------------------------------------------
# U-Net generator: 2-level encoder/decoder with a skip connection. 16->8->4->8->16.
# ---------------------------------------------------------------------------
class UNetGenerator(nn.Module):
    def __init__(self, ch: int = 1, ngf: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(ch, ngf, 4, 2, 1), nn.LeakyReLU(0.2, True))      # 16->8
        self.enc2 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
                                  nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(0.2, True))           # 8->4
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                                  nn.BatchNorm2d(ngf), nn.ReLU(True))                          # 4->8
        # decoder takes the skip-concatenated feature map (ngf + ngf)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ch, 4, 2, 1), nn.Tanh())        # 8->16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)            # ngf  x 8 x 8
        e2 = self.enc2(e1)           # 2ngf x 4 x 4
        d1 = self.dec1(e2)           # ngf  x 8 x 8
        d1 = torch.cat([d1, e1], dim=1)  # skip connection -> 2ngf x 8 x 8
        return self.dec2(d1)         # ch x 16 x 16


# ---------------------------------------------------------------------------
# PatchGAN discriminator: conditioned on the input (concat input||output), it
# outputs a grid of logits, each judging a local receptive-field patch.
# ---------------------------------------------------------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, ch: int = 1, ndf: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch * 2, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),          # 16->8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),                  # 8->4
            nn.Conv2d(ndf * 2, 1, 3, 1, 1),                                   # 4x4 grid of patch logits
        )

    def forward(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([inp, out], dim=1))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Pix2PixTorch:
    def __init__(self, ch: int = 1, lr: float = 2e-4, lambda_l1: float = 100.0):
        torch.manual_seed(SEED)
        self.dev = get_device()
        self.lambda_l1 = lambda_l1
        self.G = UNetGenerator(ch).to(self.dev)
        self.D = PatchDiscriminator(ch).to(self.dev)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()

    def fit(self, inp: np.ndarray, tgt: np.ndarray, steps: int = 400, batch: int = 32):
        inp = torch.as_tensor(inp, dtype=torch.float32, device=self.dev)
        tgt = torch.as_tensor(tgt, dtype=torch.float32, device=self.dev)
        self.d_hist, self.g_hist, self.l1_hist = [], [], []
        for _ in range(steps):
            idx = torch.randint(0, len(inp), (batch,), device=self.dev)
            a, b = inp[idx], tgt[idx]      # paired input/target

            # --- D step: (a, b) real -> 1, (a, G(a)) fake -> 0 ---
            fake = self.G(a)
            d_real = self.D(a, b)
            d_fake = self.D(a, fake.detach())
            lossD = 0.5 * (self.bce(d_real, torch.ones_like(d_real))
                           + self.bce(d_fake, torch.zeros_like(d_fake)))
            self.optD.zero_grad(); lossD.backward(); self.optD.step()

            # --- G step: fool D on the pair + match target with L1 ---
            d_gen = self.D(a, fake)
            l1 = self.l1(fake, b)
            lossG = self.bce(d_gen, torch.ones_like(d_gen)) + self.lambda_l1 * l1
            self.optG.zero_grad(); lossG.backward(); self.optG.step()

            self.d_hist.append(lossD.item()); self.g_hist.append(lossG.item())
            self.l1_hist.append(l1.item())
        return self

    @torch.no_grad()
    def generate(self, inp: np.ndarray) -> np.ndarray:
        self.G.eval()
        a = torch.as_tensor(inp, dtype=torch.float32, device=self.dev)
        out = self.G(a).cpu().numpy()
        self.G.train()
        return out


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    inp, tgt = make_pairs(256)
    print(f"data: input {inp.shape}, target {tgt.shape}, range [{inp.min():.1f}, {inp.max():.1f}]")

    gan = Pix2PixTorch().fit(inp, tgt, steps=400, batch=32)
    print(f"D loss trend:  {np.mean(gan.d_hist[:50]):.3f} -> {np.mean(gan.d_hist[-50:]):.3f}")
    print(f"L1 loss trend: {np.mean(gan.l1_hist[:50]):.3f} -> {np.mean(gan.l1_hist[-50:]):.3f}"
          f"  (falling => generator matches the paired target)")

    # Quality proxy: per-pixel L1 between G(input) and the true target.
    ti, tt = make_pairs(64, seed=123)
    pred = gan.generate(ti)
    mae = np.abs(pred - tt).mean()
    print(f"held-out per-pixel MAE = {mae:.3f} (0 = perfect; ~2.0 = worst on [-1,1])")


if __name__ == "__main__":
    demo()

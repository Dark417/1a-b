"""
Least Squares GAN (LSGAN)
=========================
A regular GAN uses the sigmoid cross-entropy (BCE) discriminator loss. Once a
fake sample lands on the correct side of the decision boundary, BCE saturates
and the gradient it sends back to the generator nearly vanishes - even if that
fake is still far from the real data manifold. LSGAN replaces BCE with a simple
least-squares (L2) penalty on the discriminator's raw output. The L2 loss keeps
penalizing samples in proportion to how far they sit from the target value, so
"correct but far" fakes still receive a strong gradient that pulls them toward
the data. Minimizing the LSGAN objective is equivalent to minimizing a Pearson
chi-square divergence between the real+fake mixture and the data.

Variants implemented here:
    - LSGAN with the standard a=0, b=1, c=1 label coding
    - (For contrast the demo also reports a BCE-trained GAN's mode coverage.)

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - Least-squares loss to combat vanishing generator gradients

References:
    - Mao et al. (2017), "Least Squares Generative Adversarial Networks"
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
# 2-D toy data: a ring of k Gaussian modes.
# ---------------------------------------------------------------------------
def make_ring(n: int = 2000, k: int = 8, r: float = 2.0, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * rng.integers(0, k, n) / k
    centers = np.c_[r * np.cos(ang), r * np.sin(ang)]
    return (centers + 0.1 * rng.normal(size=(n, 2))).astype(np.float32)


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 8, data_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, data_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Outputs a single *unbounded* real value (no sigmoid): LSGAN scores it with L2."""

    def __init__(self, data_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class LSGANTorch:
    """Least-squares GAN. Labels a (fake target), b (real target), c (G target)."""

    def __init__(self, noise_dim: int = 8, data_dim: int = 2, lr: float = 2e-4,
                 a: float = 0.0, b: float = 1.0, c: float = 1.0):
        torch.manual_seed(SEED)
        self.dev = get_device()
        self.noise_dim = noise_dim
        self.a, self.b, self.c = a, b, c
        self.G = Generator(noise_dim, data_dim).to(self.dev)
        self.D = Discriminator(data_dim).to(self.dev)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))

    @staticmethod
    def _mse(out: torch.Tensor, target: float) -> torch.Tensor:
        return 0.5 * ((out - target) ** 2).mean()

    def fit(self, real: np.ndarray, steps: int = 1500, batch: int = 128):
        real = torch.as_tensor(real, dtype=torch.float32, device=self.dev)
        self.d_hist, self.g_hist = [], []
        for _ in range(steps):
            idx = torch.randint(0, len(real), (batch,), device=self.dev)
            x = real[idx]
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            # --- D step: push D(real)->b, D(fake)->a ---
            fake = self.G(z).detach()
            lossD = self._mse(self.D(x), self.b) + self._mse(self.D(fake), self.a)
            self.optD.zero_grad(); lossD.backward(); self.optD.step()
            # --- G step: push D(fake)->c (so fakes look "real" to D) ---
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            lossG = self._mse(self.D(self.G(z)), self.c)
            self.optG.zero_grad(); lossG.backward(); self.optG.step()
            self.d_hist.append(lossD.item()); self.g_hist.append(lossG.item())
        return self

    @torch.no_grad()
    def generate(self, n: int) -> np.ndarray:
        z = torch.randn(n, self.noise_dim, device=self.dev)
        return self.G(z).cpu().numpy()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _coverage(fake: np.ndarray, k: int = 8) -> int:
    modes = np.arctan2(fake[:, 1], fake[:, 0])
    return len(np.unique(np.round(modes / (2 * np.pi / k)).astype(int) % k))


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    real = make_ring(2000)
    print(f"real mean={real.mean(0).round(2)} std={real.std(0).round(2)}")

    gan = LSGANTorch().fit(real, steps=1500, batch=128)
    fake = gan.generate(800)
    d0, d1 = np.mean(gan.d_hist[:100]), np.mean(gan.d_hist[-100:])
    print(f"D (L2) loss trend: {d0:.3f} -> {d1:.3f}")
    print(f"G (L2) loss trend: {np.mean(gan.g_hist[:100]):.3f} -> "
          f"{np.mean(gan.g_hist[-100:]):.3f}")
    print(f"covered {_coverage(fake)}/8 modes; fake mean={fake.mean(0).round(2)} "
          f"std={fake.std(0).round(2)}")


if __name__ == "__main__":
    demo()

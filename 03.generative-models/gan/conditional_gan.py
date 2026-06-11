"""
Conditional GAN (cGAN)
======================
A plain GAN draws samples from p(x) with no control over *which* sample. A
conditional GAN draws from p(x | y): both the generator and the discriminator
receive an extra label y, so we can ask the generator for a specific class. The
discriminator's job becomes harder and more informative - it must judge not just
"is this real?" but "is this a real example *of class y*?". Here the label is
turned into a learned embedding vector and concatenated into the input of both
networks; the demo learns labeled 2-D clusters and then samples a chosen class.

Variants implemented here:
    - Conditional GAN with label embeddings fed to G and D
    - Non-saturating adversarial loss (same game as vanilla GAN)

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - Conditioning via learned label embeddings concatenated to inputs

References:
    - Mirza & Osindero (2014), "Conditional Generative Adversarial Nets"
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
# Labeled 2-D data: k clusters placed on a ring, each with its own class id.
# ---------------------------------------------------------------------------
def make_clusters(n: int = 2000, k: int = 4, r: float = 2.0, seed: int = SEED):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, n)
    ang = 2 * np.pi * labels / k
    centers = np.c_[r * np.cos(ang), r * np.sin(ang)]
    x = (centers + 0.1 * rng.normal(size=(n, 2))).astype(np.float32)
    return x, labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Generator: [noise ; embed(y)] -> 2-D sample.
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim: int = 8, n_classes: int = 4, emb_dim: int = 8,
                 data_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + emb_dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, data_dim))

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.emb(y)], dim=1))


# ---------------------------------------------------------------------------
# Discriminator: [x ; embed(y)] -> logit "is this a real class-y sample?".
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, n_classes: int = 4, emb_dim: int = 8, data_dim: int = 2,
                 hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + emb_dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.emb(y)], dim=1))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ConditionalGANTorch:
    def __init__(self, noise_dim: int = 8, n_classes: int = 4, data_dim: int = 2,
                 lr: float = 2e-4):
        torch.manual_seed(SEED)
        self.dev = get_device()
        self.noise_dim, self.n_classes = noise_dim, n_classes
        self.G = Generator(noise_dim, n_classes, data_dim=data_dim).to(self.dev)
        self.D = Discriminator(n_classes, data_dim=data_dim).to(self.dev)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.bce = nn.BCEWithLogitsLoss()

    def fit(self, real: np.ndarray, labels: np.ndarray, steps: int = 1500, batch: int = 128):
        real = torch.as_tensor(real, dtype=torch.float32, device=self.dev)
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.dev)
        ones = torch.ones(batch, 1, device=self.dev)
        zeros = torch.zeros(batch, 1, device=self.dev)
        self.d_hist, self.g_hist = [], []
        for _ in range(steps):
            idx = torch.randint(0, len(real), (batch,), device=self.dev)
            x, y = real[idx], labels[idx]
            # --- D step: (real, true y) -> 1, (fake, sampled y) -> 0 ---
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            yf = torch.randint(0, self.n_classes, (batch,), device=self.dev)
            fake = self.G(z, yf).detach()
            lossD = self.bce(self.D(x, y), ones) + self.bce(self.D(fake, yf), zeros)
            self.optD.zero_grad(); lossD.backward(); self.optD.step()
            # --- G step: non-saturating, make (fake, yf) look real ---
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            yf = torch.randint(0, self.n_classes, (batch,), device=self.dev)
            lossG = self.bce(self.D(self.G(z, yf), yf), ones)
            self.optG.zero_grad(); lossG.backward(); self.optG.step()
            self.d_hist.append(lossD.item()); self.g_hist.append(lossG.item())
        return self

    @torch.no_grad()
    def generate(self, n: int, label: int) -> np.ndarray:
        z = torch.randn(n, self.noise_dim, device=self.dev)
        y = torch.full((n,), int(label), dtype=torch.long, device=self.dev)
        return self.G(z, y).cpu().numpy()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    k = 4
    x, y = make_clusters(2000, k=k)
    centers = np.array([[2 * np.cos(2 * np.pi * c / k), 2 * np.sin(2 * np.pi * c / k)]
                        for c in range(k)], dtype=np.float32)

    gan = ConditionalGANTorch(n_classes=k).fit(x, y, steps=1500, batch=128)
    print(f"D loss {np.mean(gan.d_hist[:100]):.3f} -> {np.mean(gan.d_hist[-100:]):.3f}")
    # Conditioning check: each requested class should land near its own center.
    ok = 0
    for c in range(k):
        s = gan.generate(200, label=c)
        d = np.linalg.norm(s.mean(0) - centers[c])
        near = int(np.argmin(np.linalg.norm(s.mean(0) - centers, axis=1)) == c)
        ok += near
        print(f"  class {c}: sample mean={s.mean(0).round(2)} "
              f"target={centers[c].round(2)} dist={d:.2f} "
              f"{'OK' if near else 'MISS'}")
    print(f"conditioning correct for {ok}/{k} classes")


if __name__ == "__main__":
    demo()

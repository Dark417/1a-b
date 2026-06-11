"""
Generative Adversarial Network (vanilla GAN)
============================================
Two networks play a game: a Generator turns noise into fake samples, a
Discriminator tries to tell real from fake. They train against each other until
the fakes are indistinguishable from real data. The original (Goodfellow 2014)
formulation, plus the practical non-saturating loss.

Variants implemented here:
    - Minimax GAN and the non-saturating generator loss
    - A from-scratch NumPy GAN on a 2-D toy distribution
    - PyTorch GAN (reusable as the base for DCGAN/WGAN/cGAN, see MAP.md)

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - Why the saturating generator loss vanishes early → the non-saturating fix
    - Failure modes: mode collapse, oscillation

References:
    - Goodfellow et al. (2014), "Generative Adversarial Networks"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
def _lrelu(z, a=0.2): return np.where(z > 0, z, a * z)
def _dlrelu(z, a=0.2): return np.where(z > 0, 1.0, a)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — tiny G and D, full backprop, 2-D toy data
# ---------------------------------------------------------------------------
class _MLP:
    """A minimal 1-hidden-layer net with manual backprop (LeakyReLU hidden)."""

    def __init__(self, nin, nh, nout, out_act, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / nin), (nin, nh)); self.b1 = np.zeros(nh)
        self.W2 = rng.normal(0, np.sqrt(2 / nh), (nh, nout)); self.b2 = np.zeros(nout)
        self.out_act = out_act

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = _lrelu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.out = _sigmoid(self.z2) if self.out_act == "sigmoid" else self.z2
        return self.out

    def backward(self, dout):
        """Return grads dict and dX (for the generator, chained from D)."""
        dz2 = dout
        dW2 = self.a1.T @ dz2; db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * _dlrelu(self.z1)
        dW1 = self.X.T @ dz1; db1 = dz1.sum(0)
        dX = dz1 @ self.W1.T
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, dX

    def step(self, g, lr):
        self.W1 -= lr * g["W1"]; self.b1 -= lr * g["b1"]
        self.W2 -= lr * g["W2"]; self.b2 -= lr * g["b2"]


class GANNumPy:
    def __init__(self, data_dim=2, noise_dim=8, hidden=32, lr=5e-3, seed=SEED):
        self.G = _MLP(noise_dim, hidden, data_dim, "linear", seed)
        self.D = _MLP(data_dim, hidden, 1, "sigmoid", seed + 1)
        self.noise_dim, self.lr = noise_dim, lr

    def _noise(self, n, rng): return rng.normal(size=(n, self.noise_dim))

    def fit(self, real, epochs=3000, batch=128, seed=SEED):
        rng = np.random.default_rng(seed)
        self.d_hist, self.g_hist = [], []
        eps = 1e-8
        for _ in range(epochs):
            # --- train D: maximize log D(real) + log(1 - D(G(z))) ---
            idx = rng.integers(0, len(real), batch)
            x = real[idx]
            z = self._noise(batch, rng)
            fake = self.G.forward(z)
            d_real = self.D.forward(x)
            g_real, _ = self.D.backward((d_real - 1.0) / batch)  # -dlogD(real)
            d_fake = self.D.forward(fake)
            g_fake, _ = self.D.backward((d_fake - 0.0) / batch)  # -dlog(1-D(fake))
            self.D.step({k: g_real[k] + g_fake[k] for k in g_real}, self.lr)

            # --- train G: NON-SATURATING  max log D(G(z)) ---
            z = self._noise(batch, rng)
            fake = self.G.forward(z)
            d_out = self.D.forward(fake)
            # dL_G/d(D logits): want D->1, grad of -log D(fake) is (d_out - 1)
            _, d_dx = self.D.backward((d_out - 1.0) / batch)
            g_grad, _ = self.G.backward(d_dx)
            self.G.step(g_grad, self.lr)

            self.d_hist.append(-np.mean(np.log(d_real + eps) + np.log(1 - d_fake + eps)))
            self.g_hist.append(-np.mean(np.log(d_out + eps)))
        return self

    def generate(self, n, seed=SEED):
        return self.G.forward(self._noise(n, np.random.default_rng(seed)))


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — clean base for GAN variants
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, data_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, data_dim))

    def forward(self, z): return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1))

    def forward(self, x): return self.net(x)          # logits


class GANTorch:
    def __init__(self, data_dim=2, noise_dim=8, lr=2e-4):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dev, self.noise_dim = dev, noise_dim
        self.G = Generator(noise_dim, data_dim).to(dev)
        self.D = Discriminator(data_dim).to(dev)
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.bce = nn.BCEWithLogitsLoss()

    def fit(self, real, epochs=3000, batch=128):
        real = torch.as_tensor(real, dtype=torch.float32, device=self.dev)
        self.d_hist, self.g_hist = [], []
        for _ in range(epochs):
            idx = torch.randint(0, len(real), (batch,), device=self.dev)
            x = real[idx]
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            # D step
            fake = self.G(z).detach()
            lossD = self.bce(self.D(x), torch.ones(batch, 1, device=self.dev)) \
                + self.bce(self.D(fake), torch.zeros(batch, 1, device=self.dev))
            self.optD.zero_grad(); lossD.backward(); self.optD.step()
            # G step (non-saturating: label fakes as "real")
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            lossG = self.bce(self.D(self.G(z)), torch.ones(batch, 1, device=self.dev))
            self.optG.zero_grad(); lossG.backward(); self.optG.step()
            self.d_hist.append(lossD.item()); self.g_hist.append(lossG.item())
        return self

    @torch.no_grad()
    def generate(self, n):
        z = torch.randn(n, self.noise_dim, device=self.dev)
        return self.G(z).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo — learn a 2-D mixture of Gaussians (easy to eyeball mode collapse)
# ---------------------------------------------------------------------------
def make_ring(n=2000, k=8, r=2.0, seed=SEED):
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * rng.integers(0, k, n) / k
    centers = np.c_[r * np.cos(ang), r * np.sin(ang)]
    return (centers + 0.1 * rng.normal(size=(n, 2))).astype(np.float32)


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    real = make_ring(2000)

    g = GANNumPy(lr=5e-3).fit(real, epochs=2000)
    fake = g.generate(500)
    print(f"NumPy GAN  fake mean={fake.mean(0).round(2)}  std={fake.std(0).round(2)}")
    print(f"           (real mean={real.mean(0).round(2)} std={real.std(0).round(2)})")

    t = GANTorch(lr=2e-4).fit(real, epochs=3000)
    ft = t.generate(500)
    print(f"Torch GAN  fake mean={ft.mean(0).round(2)}  std={ft.std(0).round(2)}")
    # coverage: how many of the 8 modes did we hit?
    modes = np.arctan2(ft[:, 1], ft[:, 0])
    hit = len(np.unique(np.round(modes / (2 * np.pi / 8)).astype(int) % 8))
    print(f"           covered {hit}/8 modes (low number = mode collapse)")


if __name__ == "__main__":
    demo()

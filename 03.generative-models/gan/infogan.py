"""
InfoGAN
=======
A vanilla GAN's latent code z is entangled: no individual coordinate has a clear
meaning. InfoGAN splits the latent input into incompressible noise z and a small
set of *structured* latent codes c, then adds a term that maximizes the mutual
information I(c; G(z, c)) between the codes and the generated output. This forces
the generator to use c in a recoverable, disentangled way (e.g. one continuous
code ends up controlling one factor of variation). Mutual information is
intractable, so InfoGAN maximizes a variational lower bound using an auxiliary
network Q that tries to reconstruct c from a generated sample. Q shares its body
with the discriminator. Here a single continuous code learns to control the
angle around a 2-D ring.

Variants implemented here:
    - InfoGAN with one continuous latent code (Gaussian Q posterior)
    - Auxiliary Q network sharing features with D; variational MI lower bound

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - Variational mutual-information maximization (the InfoGAN lower bound)

References:
    - Chen et al. (2016), "InfoGAN: Interpretable Representation Learning by
      Information Maximizing Generative Adversarial Nets"
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
# 2-D toy data: a continuous ring (a circle of radius r with a little noise).
# A single latent code should learn to parameterize the angle.
# ---------------------------------------------------------------------------
def make_ring(n: int = 2000, r: float = 2.0, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    pts = np.c_[r * np.cos(ang), r * np.sin(ang)]
    return (pts + 0.05 * rng.normal(size=(n, 2))).astype(np.float32)


# ---------------------------------------------------------------------------
# Generator: [noise ; code c] -> 2-D sample.
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim: int = 4, code_dim: int = 1, data_dim: int = 2,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + code_dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, data_dim))

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, c], dim=1))


# ---------------------------------------------------------------------------
# Shared trunk -> two heads:
#   D head: real/fake logit.
#   Q head: parameters (mu, log_var) of the Gaussian posterior over the code c,
#           used for the variational MI bound.
# ---------------------------------------------------------------------------
class DiscriminatorQ(nn.Module):
    def __init__(self, data_dim: int = 2, code_dim: int = 1, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True))
        self.d_head = nn.Linear(hidden, 1)              # real/fake logit
        self.q_mu = nn.Linear(hidden, code_dim)         # posterior mean
        self.q_logvar = nn.Linear(hidden, code_dim)     # posterior log-variance

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.d_head(h), self.q_mu(h), self.q_logvar(h)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class InfoGANTorch:
    def __init__(self, noise_dim: int = 4, code_dim: int = 1, data_dim: int = 2,
                 lr: float = 2e-4, lambda_mi: float = 0.5):
        torch.manual_seed(SEED)
        self.dev = get_device()
        self.noise_dim, self.code_dim, self.lambda_mi = noise_dim, code_dim, lambda_mi
        self.G = Generator(noise_dim, code_dim, data_dim).to(self.dev)
        self.DQ = DiscriminatorQ(data_dim, code_dim).to(self.dev)
        self.bce = nn.BCEWithLogitsLoss()
        # G and Q are optimized together (they jointly maximize the MI bound).
        self.optG = torch.optim.Adam(
            list(self.G.parameters()) + list(self.DQ.q_mu.parameters())
            + list(self.DQ.q_logvar.parameters()),
            lr=lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(self.DQ.parameters(), lr=lr, betas=(0.5, 0.999))

    def _sample_code(self, batch: int) -> torch.Tensor:
        # continuous code ~ U(-1, 1)
        return torch.rand(batch, self.code_dim, device=self.dev) * 2 - 1

    @staticmethod
    def _gaussian_nll(c: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """Negative log-likelihood of c under N(mu, exp(logvar)) (the -Q term)."""
        return 0.5 * (logvar + (c - mu) ** 2 / logvar.exp()).sum(1).mean()

    def fit(self, real: np.ndarray, steps: int = 1500, batch: int = 128):
        real = torch.as_tensor(real, dtype=torch.float32, device=self.dev)
        ones = torch.ones(batch, 1, device=self.dev)
        zeros = torch.zeros(batch, 1, device=self.dev)
        self.d_hist, self.g_hist, self.mi_hist = [], [], []
        for _ in range(steps):
            idx = torch.randint(0, len(real), (batch,), device=self.dev)
            x = real[idx]
            # --- D step ---
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            c = self._sample_code(batch)
            fake = self.G(z, c).detach()
            d_real, _, _ = self.DQ(x)
            d_fake, _, _ = self.DQ(fake)
            lossD = self.bce(d_real, ones) + self.bce(d_fake, zeros)
            self.optD.zero_grad(); lossD.backward(); self.optD.step()
            # --- G + Q step: fool D AND let Q recover the code (MI bound) ---
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            c = self._sample_code(batch)
            gen = self.G(z, c)
            d_g, q_mu, q_logvar = self.DQ(gen)
            lossG = self.bce(d_g, ones)
            mi = self._gaussian_nll(c, q_mu, q_logvar)  # minimizing NLL maximizes MI bound
            (lossG + self.lambda_mi * mi).backward()
            self.optG.step(); self.optG.zero_grad()
            self.d_hist.append(lossD.item()); self.g_hist.append(lossG.item())
            self.mi_hist.append(mi.item())
        return self

    @torch.no_grad()
    def generate(self, n: int, code: float | None = None) -> np.ndarray:
        z = torch.randn(n, self.noise_dim, device=self.dev)
        if code is None:
            c = self._sample_code(n)
        else:
            c = torch.full((n, self.code_dim), float(code), device=self.dev)
        return self.G(z, c).cpu().numpy()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    real = make_ring(2000)

    gan = InfoGANTorch().fit(real, steps=1500, batch=128)
    print(f"D loss {np.mean(gan.d_hist[:100]):.3f} -> {np.mean(gan.d_hist[-100:]):.3f}")
    print(f"Q NLL  {np.mean(gan.mi_hist[:100]):.3f} -> {np.mean(gan.mi_hist[-100:]):.3f}"
          f"  (falling => MI bound tightening, code is recoverable)")

    # Disentanglement check: sweeping the code should sweep the ring angle.
    angles = []
    for cval in np.linspace(-1, 1, 9):
        s = gan.generate(200, code=cval)
        angles.append(np.arctan2(s[:, 1].mean(), s[:, 0].mean()))
    angles = np.unwrap(angles)
    span = angles.max() - angles.min()
    print(f"angle span as code goes -1->1: {np.degrees(span):.0f} deg "
          f"(large => the code controls position on the ring)")


if __name__ == "__main__":
    demo()

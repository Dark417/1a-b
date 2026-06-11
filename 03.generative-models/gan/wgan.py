"""
Wasserstein GAN (WGAN) and WGAN-GP
==================================
The vanilla GAN minimizes the Jensen-Shannon divergence, which saturates to a
constant (giving the generator no gradient) whenever the real and fake
distributions have disjoint supports - a common situation early in training.
WGAN instead minimizes the *Wasserstein-1 (earth-mover) distance*, which varies
smoothly even for non-overlapping supports. By Kantorovich-Rubinstein duality
the distance equals a maximization over all 1-Lipschitz "critic" functions; the
critic outputs an unbounded score (not a probability). The Lipschitz constraint
is enforced two ways here: the original WGAN's crude weight clipping, and the
WGAN-GP gradient penalty that softly pins the critic's gradient norm to 1.

Variants implemented here:
    - WGAN with weight clipping (clip="weight")
    - WGAN-GP with a gradient penalty (clip="gp")

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - The 1-Lipschitz constraint (weight clipping vs gradient penalty)
    - n_critic critic updates per generator update; RMSProp (clip) / Adam (gp)

References:
    - Arjovsky, Chintala & Bottou (2017), "Wasserstein GAN"
    - Gulrajani et al. (2017), "Improved Training of Wasserstein GANs" (WGAN-GP)
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
# 2-D toy data: a ring of k Gaussian modes (same helper as vanilla_gan).
# ---------------------------------------------------------------------------
def make_ring(n: int = 2000, k: int = 8, r: float = 2.0, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * rng.integers(0, k, n) / k
    centers = np.c_[r * np.cos(ang), r * np.sin(ang)]
    return (centers + 0.1 * rng.normal(size=(n, 2))).astype(np.float32)


# ---------------------------------------------------------------------------
# Generator: noise -> 2-D sample (a plain MLP).
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim: int = 8, data_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, data_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------------------------------------------------------------------------
# Critic: a *scalar score*, NOT a probability. No sigmoid, no BatchNorm
# (BatchNorm breaks the per-sample gradient penalty). It approximates the
# 1-Lipschitz witness function of Kantorovich-Rubinstein duality.
# ---------------------------------------------------------------------------
class Critic(nn.Module):
    def __init__(self, data_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # real-valued score


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class WGANTorch:
    """WGAN / WGAN-GP. clip='weight' (clipping) or clip='gp' (gradient penalty)."""

    def __init__(self, noise_dim: int = 8, data_dim: int = 2, clip: str = "gp",
                 lr: float = 1e-4, clip_val: float = 0.01, gp_lambda: float = 10.0,
                 n_critic: int = 5):
        torch.manual_seed(SEED)
        assert clip in ("weight", "gp")
        self.dev = get_device()
        self.noise_dim, self.clip = noise_dim, clip
        self.clip_val, self.gp_lambda, self.n_critic = clip_val, gp_lambda, n_critic
        self.G = Generator(noise_dim, data_dim).to(self.dev)
        self.D = Critic(data_dim).to(self.dev)
        if clip == "weight":
            # The WGAN paper recommends RMSProp (Adam was unstable with clipping).
            self.optG = torch.optim.RMSprop(self.G.parameters(), lr=lr)
            self.optD = torch.optim.RMSprop(self.D.parameters(), lr=lr)
        else:
            self.optG = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.9))
            self.optD = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.9))

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """E[(||grad_x D(x_hat)||_2 - 1)^2] on points x_hat on lines real<->fake."""
        b = real.size(0)
        eps = torch.rand(b, 1, device=self.dev)
        xhat = (eps * real + (1 - eps) * fake).requires_grad_(True)
        score = self.D(xhat)
        grad = torch.autograd.grad(
            outputs=score, inputs=xhat,
            grad_outputs=torch.ones_like(score),
            create_graph=True, retain_graph=True)[0]
        gnorm = grad.norm(2, dim=1)
        return ((gnorm - 1.0) ** 2).mean()

    def fit(self, real: np.ndarray, steps: int = 1500, batch: int = 128):
        real = torch.as_tensor(real, dtype=torch.float32, device=self.dev)
        self.d_hist, self.w_hist = [], []  # critic loss; Wasserstein estimate
        for _ in range(steps):
            # --- train the critic n_critic times ---
            for _ in range(self.n_critic):
                idx = torch.randint(0, len(real), (batch,), device=self.dev)
                x = real[idx]
                z = torch.randn(batch, self.noise_dim, device=self.dev)
                fake = self.G(z).detach()
                # Critic maximizes E[D(real)] - E[D(fake)]  => minimize negation.
                d_real = self.D(x).mean()
                d_fake = self.D(fake).mean()
                lossD = -(d_real - d_fake)
                if self.clip == "gp":
                    lossD = lossD + self.gp_lambda * self._gradient_penalty(x, fake)
                self.optD.zero_grad(); lossD.backward(); self.optD.step()
                if self.clip == "weight":
                    # crude 1-Lipschitz enforcement: box-clip every weight
                    for p in self.D.parameters():
                        p.data.clamp_(-self.clip_val, self.clip_val)
                self.w_hist.append((d_real - d_fake).item())  # Wasserstein estimate
            # --- train the generator once: maximize E[D(G(z))] ---
            z = torch.randn(batch, self.noise_dim, device=self.dev)
            lossG = -self.D(self.G(z)).mean()
            self.optG.zero_grad(); lossG.backward(); self.optG.step()
            self.d_hist.append(lossD.item())
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

    for mode in ("gp", "weight"):
        gan = WGANTorch(clip=mode).fit(real, steps=600, batch=128)
        fake = gan.generate(800)
        w0 = np.mean(gan.w_hist[:100]); w1 = np.mean(gan.w_hist[-100:])
        print(f"[{mode:6s}] Wasserstein estimate {w0:.3f} -> {w1:.3f} "
              f"(should rise then plateau); covered {_coverage(fake)}/8 modes; "
              f"fake std={fake.std(0).round(2)}")


if __name__ == "__main__":
    demo()

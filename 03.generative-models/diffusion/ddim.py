"""
Denoising Diffusion Implicit Models (DDIM)
==========================================
DDIM keeps the *same* training as a DDPM (the noise-prediction MSE objective) but
replaces the stochastic Markov reverse chain with a **non-Markovian**,
deterministic update. The consequence is dramatic acceleration: you can skip most
of the T timesteps and still get sharp samples, and with the deterministic
(eta=0) update the map from noise to data becomes a reproducible, invertible ODE
flow.

This file is self-contained: it defines a small DDPM-style eps-model locally
(trained with the standard simplified objective) and then samples it with both
the stochastic and the deterministic DDIM updates over a *sub-sequence* of steps.

Variants implemented here:
    - Deterministic DDIM (eta = 0) — an implicit ODE sampler
    - Stochastic DDIM (0 < eta <= 1, eta = 1 recovers the DDPM ancestral sampler)
    - Accelerated sampling over an arbitrary step subset (e.g. 20 of 200 steps)

Training techniques demonstrated:
    - Reusing a DDPM eps-network unchanged (only the sampler differs)
    - Closed-form forward marginal via alpha-bar
    - Time-conditioning via a sinusoidal embedding

References:
    - Song, Meng, Ermon (2021), "Denoising Diffusion Implicit Models"
    - Ho, Jain, Abbeel (2020), DDPM (the training objective DDIM reuses)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# The DDIM update, isolated (NumPy sketch).
# ---------------------------------------------------------------------------
# Given eps_theta(x_t, t), DDIM first predicts the clean point x0, then steps to
# an earlier time s along the SAME forward marginal, adding a controllable amount
# of fresh noise sigma:
#
#     x0_pred = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t)
#     sigma   = eta * sqrt((1-abar_s)/(1-abar_t)) * sqrt(1 - abar_t/abar_s)
#     x_s     = sqrt(abar_s) * x0_pred
#               + sqrt(1 - abar_s - sigma^2) * eps      (direction pointing to x_t)
#               + sigma * z,   z ~ N(0,I)
#
# eta = 0  -> deterministic (no z): an implicit probability-flow ODE step.
def _ddim_step_numpy(x_t, eps, abar_t, abar_s, eta, rng):
    x0 = (x_t - np.sqrt(1 - abar_t) * eps) / np.sqrt(abar_t)
    sigma = eta * np.sqrt((1 - abar_s) / (1 - abar_t)) * np.sqrt(1 - abar_t / abar_s)
    dir_xt = np.sqrt(np.maximum(1 - abar_s - sigma ** 2, 0.0)) * eps
    noise = sigma * rng.normal(size=x_t.shape) if eta > 0 else 0.0
    return np.sqrt(abar_s) * x0 + dir_xt + noise


# ---------------------------------------------------------------------------
# PyTorch implementation
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class EpsMLP(nn.Module):
    """Noise predictor eps_theta(x_t, t) — identical in spirit to the DDPM net."""

    def __init__(self, data_dim: int = 2, hidden: int = 128, t_dim: int = 32):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, sinusoidal_embedding(t, self.t_dim)], dim=1))


class DDIM(nn.Module):
    r"""
    Trains a standard eps-model (DDPM simplified loss) and samples it with the
    DDIM update over a chosen sub-sequence of timesteps.

    DDIM step from time t to an earlier time s (with x0-prediction):
        x0      = (x_t - sqrt(1-abar_t) eps) / sqrt(abar_t)
        sigma   = eta sqrt((1-abar_s)/(1-abar_t)) sqrt(1 - abar_t/abar_s)
        x_s     = sqrt(abar_s) x0 + sqrt(1-abar_s-sigma^2) eps + sigma z
    eta=0 is deterministic; eta=1 reproduces DDPM ancestral sampling.
    """

    def __init__(self, data_dim: int = 2, T: int = 200, hidden: int = 128):
        super().__init__()
        self.T = T
        self.model = EpsMLP(data_dim, hidden)
        betas = torch.linspace(1e-4, 0.02, T)
        abars = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("abars", abars)

    def q_sample(self, x0, t, eps):
        ab = self.abars[t].unsqueeze(1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps

    def loss(self, x0):
        n = len(x0)
        t = torch.randint(0, self.T, (n,), device=x0.device)
        eps = torch.randn_like(x0)
        return F.mse_loss(self.model(self.q_sample(x0, t, eps), t), eps)

    def fit(self, X, epochs: int = 400, batch: int = 256, lr: float = 2e-3):
        dev = get_device()
        self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.history = []
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=dev)
            tot = 0.0
            for s in range(0, len(X), batch):
                loss = self.loss(X[perm[s:s + batch]])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            self.history.append(tot / max(1, len(X) // batch))
        return self

    @torch.no_grad()
    def sample(self, n: int, steps: int = 20, eta: float = 0.0, data_dim: int = 2):
        """Accelerated DDIM sampling over `steps` of the T timesteps."""
        dev = next(self.parameters()).device
        # evenly spaced sub-sequence of timesteps, descending to 0
        seq = torch.linspace(0, self.T - 1, steps, device=dev).round().long()
        seq = torch.unique(seq).sort(descending=True).values
        x = torch.randn(n, data_dim, device=dev)
        for i, ti in enumerate(seq):
            t = torch.full((n,), int(ti), device=dev, dtype=torch.long)
            eps = self.model(x, t)
            abar_t = self.abars[ti]
            abar_s = self.abars[seq[i + 1]] if i + 1 < len(seq) else torch.tensor(1.0, device=dev)
            x0 = (x - (1 - abar_t).sqrt() * eps) / abar_t.sqrt()
            if i + 1 < len(seq):
                sigma = eta * ((1 - abar_s) / (1 - abar_t)).sqrt() * (1 - abar_t / abar_s).sqrt()
                dir_xt = torch.clamp(1 - abar_s - sigma ** 2, min=0.0).sqrt() * eps
                noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
                x = abar_s.sqrt() * x0 + dir_xt + noise
            else:
                x = x0  # final step lands on the clean prediction
        return x.cpu().numpy()


# ---------------------------------------------------------------------------
# Demo — train once, then compare full DDPM-style vs few-step DDIM sampling
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)  # tiny model: 1 thread avoids CPU thrashing
    from sklearn.datasets import make_moons
    X, _ = make_moons(2000, noise=0.05, random_state=SEED)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)

    m = DDIM(data_dim=2, T=200).fit(X, epochs=400)
    print(f"DDIM (eps-model) final MSE = {m.history[-1]:.4f}")

    for steps, eta, tag in [(200, 0.0, "det 200"), (20, 0.0, "det  20"), (20, 1.0, "stoch 20")]:
        s = m.sample(2000, steps=steps, eta=eta)
        print(f"  {tag}-step: samp mean={s.mean(0).round(2)} std={s.std(0).round(2)}")
    print(f"  data:        mean={X.mean(0).round(2)} std={X.std(0).round(2)}")


if __name__ == "__main__":
    demo()

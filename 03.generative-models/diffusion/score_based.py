"""
Score-Based Generative Model (denoising score matching + Langevin)
==================================================================
Instead of modelling a density directly, learn its **score** -- the gradient of
the log-density, s(x) = grad_x log p(x). Once you can estimate the score, you can
draw samples by **Langevin dynamics**: take small steps uphill in log-density and
add a little noise. Plain score matching is unstable in low-density regions, so we
use **denoising score matching** at several noise scales and sample with
**annealed Langevin dynamics** (Song & Ermon 2019).

Variants implemented here:
    - Denoising score matching (DSM) at a single noise level
    - Noise-conditional score network across multiple sigmas (NCSN style)
    - Annealed Langevin dynamics sampler

Training techniques demonstrated:
    - The DSM identity: regress the score on the (scaled) noise direction
    - Noise conditioning of the network on sigma
    - Langevin / SGLD sampling (the continuous-time relative of the DDPM sampler)

References:
    - Vincent (2011), "A Connection Between Score Matching and Denoising
      Autoencoders"
    - Song & Ermon (2019), "Generative Modeling by Estimating Gradients of the
      Data Distribution" (NCSN)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# The denoising score matching identity (NumPy sketch).
# ---------------------------------------------------------------------------
# Perturb x with Gaussian noise: x~ = x + sigma * eps. The score of the perturbed
# density q_sigma(x~|x) = N(x, sigma^2 I) is known exactly:
#
#     grad_{x~} log q_sigma(x~|x) = (x - x~) / sigma^2 = -eps / sigma
#
# DSM trains s_theta(x~, sigma) to match this target. Hence the optimal network
# learns the score of the *perturbed data* distribution, and as sigma -> 0 this
# approaches the true data score.
def _dsm_target_numpy(x, x_tilde, sigma):
    return (x - x_tilde) / (sigma ** 2)


# ---------------------------------------------------------------------------
# PyTorch implementation
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ScoreNet(nn.Module):
    r"""
    Noise-conditional score network s_theta(x, sigma) ~ grad_x log p_sigma(x).

    We condition on log(sigma) (a single scalar feature broadcast in) so one
    network covers all noise scales. The network outputs a vector field in R^d.
    """

    def __init__(self, data_dim: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim))

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (N,) ; feed log-sigma as an extra input feature
        cond = torch.log(sigma).unsqueeze(1)
        return self.net(torch.cat([x, cond], dim=1))


class ScoreModel(nn.Module):
    r"""
    NCSN-style training and annealed Langevin sampling.

    Geometric noise scales sigma_1 > ... > sigma_L. DSM loss (lambda(sigma)=sigma^2
    weighting makes the loss scale-balanced):
        L = E_{sigma, x, eps}  || sigma * s_theta(x + sigma eps, sigma) + eps ||^2 .
    (Target score is -eps/sigma; multiplying through by sigma gives this form.)

    Annealed Langevin at level i with step alpha_i = step * (sigma_i/sigma_L)^2:
        x <- x + (alpha_i/2) s_theta(x, sigma_i) + sqrt(alpha_i) z .
    """

    def __init__(self, data_dim: int = 2, hidden: int = 128,
                 sigma_min: float = 0.02, sigma_max: float = 2.0, n_scales: int = 10):
        super().__init__()
        self.net = ScoreNet(data_dim, hidden)
        sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), n_scales))
        self.register_buffer("sigmas", sigmas)   # descending
        self.data_dim = data_dim

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        n = len(x0)
        idx = torch.randint(0, len(self.sigmas), (n,), device=x0.device)
        sigma = self.sigmas[idx]                              # (N,)
        eps = torch.randn_like(x0)
        x_tilde = x0 + sigma.unsqueeze(1) * eps
        score = self.net(x_tilde, sigma)
        # target: -eps/sigma ; weighted by sigma^2 -> regress sigma*score onto -eps
        return ((sigma.unsqueeze(1) * score + eps) ** 2).sum(1).mean()

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
    def sample(self, n: int, n_steps: int = 50, step: float = 1e-4):
        """Annealed Langevin dynamics from large to small noise scale."""
        dev = next(self.parameters()).device
        x = torch.randn(n, self.data_dim, device=dev) * self.sigmas[0]
        sigma_min = self.sigmas[-1]
        for sigma in self.sigmas:
            alpha = step * (sigma / sigma_min) ** 2          # scale step with sigma
            s_in = torch.full((n,), float(sigma), device=dev)
            for _ in range(n_steps):
                score = self.net(x, s_in)
                x = x + 0.5 * alpha * score + alpha.sqrt() * torch.randn_like(x)
        return x.cpu().numpy()


# ---------------------------------------------------------------------------
# Demo — 2-D two-moons via annealed Langevin
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)  # tiny model: 1 thread avoids CPU thrashing
    from sklearn.datasets import make_moons
    X, _ = make_moons(2000, noise=0.05, random_state=SEED)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)

    m = ScoreModel(data_dim=2, n_scales=10).fit(X, epochs=400)
    print(f"Score model final DSM loss = {m.history[-1]:.4f}")

    s = m.sample(2000, n_steps=50, step=2e-4)
    print(f"  data    mean={X.mean(0).round(2)}  std={X.std(0).round(2)}")
    print(f"  samples mean={s.mean(0).round(2)}  std={s.std(0).round(2)}")


if __name__ == "__main__":
    demo()

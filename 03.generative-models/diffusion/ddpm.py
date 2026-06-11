"""
Denoising Diffusion Probabilistic Model (DDPM)
==============================================
A generative model that learns to *reverse* a fixed noising process. The forward
process gradually corrupts data into pure Gaussian noise over T steps; a neural
network is trained to undo one step at a time, so sampling starts from noise and
denoises back to a data sample. The headline result (Ho et al. 2020) is that the
whole variational bound collapses to a simple noise-prediction MSE objective.

Variants implemented here:
    - Standard DDPM with a fixed linear beta schedule
    - The simplified epsilon-prediction (noise-prediction) training objective
    - Ancestral (stochastic) reverse sampler
    - A tiny MLP eps-network on a 2-D toy (two-moons)

Training techniques demonstrated:
    - Closed-form forward marginal q(x_t|x_0) via alpha-bar (one-shot noising)
    - Reparameterization (reused from the VAE) to write x_t = sqrt(abar) x0 + ...
    - Time-conditioning via a sinusoidal embedding

References:
    - Ho, Jain, Abbeel (2020), "Denoising Diffusion Probabilistic Models"
    - Sohl-Dickstein et al. (2015), "Deep Unsupervised Learning using
      Nonequilibrium Thermodynamics"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# The forward process in closed form (NumPy sketch).
# ---------------------------------------------------------------------------
# With betas b_1..b_T and alpha_t = 1 - b_t, abar_t = prod_{s<=t} alpha_s, the
# forward marginal has a closed form (no need to iterate t steps):
#
#     q(x_t | x_0) = N( sqrt(abar_t) x_0 , (1 - abar_t) I )
#     => x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) * eps,   eps ~ N(0, I)
#
def _forward_sample_numpy(x0, abar_t, rng):
    """Sample x_t directly from x_0 using the closed-form marginal."""
    eps = rng.normal(size=x0.shape)
    return np.sqrt(abar_t) * x0 + np.sqrt(1.0 - abar_t) * eps, eps


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


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Linear beta schedule and the precomputed alpha / alpha-bar tensors."""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    abars = torch.cumprod(alphas, dim=0)
    return betas, alphas, abars


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style timestep embedding for integer steps t (shape (N,))."""
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class EpsMLP(nn.Module):
    """Predict the noise eps added to x_t, conditioned on the timestep t."""

    def __init__(self, data_dim: int = 2, hidden: int = 128, t_dim: int = 32):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(data_dim + t_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding(t, self.t_dim)
        return self.net(torch.cat([x, emb], dim=1))


class DDPM(nn.Module):
    r"""
    Discrete-time DDPM. Trains the simplified objective

        L = E_{x0, t, eps} || eps - eps_theta(x_t, t) ||^2 ,
        x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps.

    Sampling is the ancestral reverse chain
        x_{t-1} = 1/sqrt(alpha_t) ( x_t - (beta_t / sqrt(1-abar_t)) eps_theta )
                  + sqrt(beta_t) z,   z ~ N(0, I)  (z = 0 at t = 0).
    """

    def __init__(self, data_dim: int = 2, T: int = 200, hidden: int = 128):
        super().__init__()
        self.T = T
        self.model = EpsMLP(data_dim, hidden)
        betas, alphas, abars = make_beta_schedule(T)
        # store schedule as buffers so .to(device) moves them too
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abars", abars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        """Closed-form forward marginal x_t ~ q(x_t | x0)."""
        ab = self.abars[t].unsqueeze(1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        n = len(x0)
        t = torch.randint(0, self.T, (n,), device=x0.device)
        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        eps_hat = self.model(x_t, t)
        return F.mse_loss(eps_hat, eps)

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
    def sample(self, n: int, data_dim: int = 2):
        """Ancestral sampling: start from noise, denoise down to x_0."""
        dev = next(self.parameters()).device
        x = torch.randn(n, data_dim, device=dev)
        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, device=dev, dtype=torch.long)
            eps_hat = self.model(x, t)
            alpha = self.alphas[ti]
            abar = self.abars[ti]
            beta = self.betas[ti]
            mean = (x - beta / (1.0 - abar).sqrt() * eps_hat) / alpha.sqrt()
            if ti > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.cpu().numpy()


# ---------------------------------------------------------------------------
# Demo — learn a 2-D two-moons distribution
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)  # tiny model: 1 thread avoids CPU thrashing
    from sklearn.datasets import make_moons
    X, _ = make_moons(2000, noise=0.05, random_state=SEED)
    X = (X - X.mean(0)) / X.std(0)            # standardize
    X = X.astype(np.float32)

    m = DDPM(data_dim=2, T=200).fit(X, epochs=400)
    print(f"DDPM final eps-MSE = {m.history[-1]:.4f}")

    s = m.sample(2000)
    print(f"  data  mean={X.mean(0).round(2)}  std={X.std(0).round(2)}")
    print(f"  samp  mean={s.mean(0).round(2)}  std={s.std(0).round(2)}")


if __name__ == "__main__":
    demo()

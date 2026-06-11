"""
Glow — flows with actnorm, invertible 1x1 convolutions, and affine coupling
===========================================================================
Glow (Kingma & Dhariwal 2018) improves on RealNVP by replacing the *fixed*
coordinate-permuting mask with a **learned invertible 1x1 convolution** (a general
linear mixing of channels), and by adding **actnorm** (a data-dependent
per-channel affine normalization) in place of batchnorm. Each Glow step is

    actnorm  ->  invertible 1x1 conv  ->  affine coupling

and the whole thing is an exact-likelihood normalizing flow. This file implements
Glow on 2-D data (so the "channels" are the 2 coordinates and the 1x1 conv is a
learned 2x2 mixing matrix) -- tiny and fast on CPU, while keeping every Glow
component and its exact log-det.

Variants implemented here:
    - ActNorm (data-dependent initialization to zero-mean/unit-variance)
    - Invertible 1x1 convolution (learned linear mixing; log-det = log|det W|)
    - Affine coupling (reused idea from RealNVP)
    - A stack of Glow steps with exact log-likelihood

Training techniques demonstrated:
    - Data-dependent initialization (actnorm) as a normalization technique
    - Exact change-of-variables likelihood; per-component log-dets
    - tanh-bounded log-scale for stability

References:
    - Kingma & Dhariwal (2018), "Glow: Generative Flow with Invertible 1x1
      Convolutions"
    - Dinh et al. (2017), RealNVP (the coupling layers Glow builds on)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# The invertible 1x1 convolution log-det (NumPy sketch).
# ---------------------------------------------------------------------------
# A 1x1 conv with weight matrix W (C x C) mixes the channels of every spatial
# location: y = W x. For an H x W image this is applied H*W times, so the total
# log-determinant is
#
#     log|det J| = H * W * log|det W|
#
# For 2-D data (H=W=1, C=2) this is just log|det W| of a learned 2x2 matrix. The
# inverse pass multiplies by W^{-1}.
def _inv1x1_logdet_numpy(W, H=1, Wd=1):
    """Total log|det| of an invertible 1x1 conv over an HxWd feature map."""
    return H * Wd * np.log(abs(np.linalg.det(W)))


# ---------------------------------------------------------------------------
# PyTorch implementation (2-D data: "channels" = coordinates)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ActNorm(nn.Module):
    r"""
    Per-dimension affine transform y = (x - mu) * exp(log_s), initialized from the
    first batch so that y has zero mean and unit variance (data-dependent init).
    log|det| = sum(log_s) per example.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_s = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("initialized", torch.tensor(0))

    def _init(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mu = x.mean(0)
            std = x.std(0) + 1e-6
            self.bias.data.copy_(mu)
            self.log_s.data.copy_(-torch.log(std))    # so output has unit std
            self.initialized.fill_(1)

    def forward(self, x: torch.Tensor):
        if self.initialized.item() == 0 and self.training:
            self._init(x)
        y = (x - self.bias) * torch.exp(self.log_s)
        log_det = self.log_s.sum().expand(len(x))
        return y, log_det

    def inverse(self, y: torch.Tensor):
        return y * torch.exp(-self.log_s) + self.bias


class Inv1x1(nn.Module):
    r"""
    Invertible 1x1 convolution = a learned linear mixing y = W x of the
    coordinates. log|det J| = log|det W| (per example). W is initialized as a
    random rotation (orthogonal) so it starts volume-preserving and invertible.
    """

    def __init__(self, dim: int):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(dim, dim))   # random orthogonal init
        self.W = nn.Parameter(q)

    def forward(self, x: torch.Tensor):
        y = x @ self.W.t()
        log_det = torch.slogdet(self.W)[1].expand(len(x))
        return y, log_det

    def inverse(self, y: torch.Tensor):
        return y @ torch.inverse(self.W).t()


class AffineCoupling(nn.Module):
    """Affine coupling on the first / second half of the coordinates."""

    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim))
        self.scale = nn.Parameter(torch.zeros(dim))

    def _st(self, x_a):
        s, t = self.net(x_a).chunk(2, dim=1)
        return torch.tanh(s) * self.scale, t

    def forward(self, x):
        x_a = x * self.mask
        s, t = self._st(x_a)
        s = s * (1 - self.mask); t = t * (1 - self.mask)
        y = x_a + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(1)

    def inverse(self, y):
        y_a = y * self.mask
        s, t = self._st(y_a)
        s = s * (1 - self.mask); t = t * (1 - self.mask)
        return y_a + (1 - self.mask) * ((y - t) * torch.exp(-s))


class GlowStep(nn.Module):
    """One Glow step: actnorm -> invertible 1x1 conv -> affine coupling."""

    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.actnorm = ActNorm(dim)
        self.inv1x1 = Inv1x1(dim)
        self.coupling = AffineCoupling(dim, mask, hidden)

    def forward(self, x):
        ld = torch.zeros(len(x), device=x.device)
        x, d = self.actnorm(x); ld = ld + d
        x, d = self.inv1x1(x); ld = ld + d
        x, d = self.coupling(x); ld = ld + d
        return x, ld

    def inverse(self, y):
        y = self.coupling.inverse(y)
        y = self.inv1x1.inverse(y)
        y = self.actnorm.inverse(y)
        return y


class Glow(nn.Module):
    """A small Glow flow over 2-D data with a standard-normal base."""

    def __init__(self, dim: int = 2, n_steps: int = 6, hidden: int = 64):
        super().__init__()
        steps = []
        for i in range(n_steps):
            m = torch.zeros(dim); m[i % 2::2] = 1.0
            steps.append(GlowStep(dim, m, hidden))
        self.steps = nn.ModuleList(steps)
        self.dim = dim

    def forward(self, x):
        ld = torch.zeros(len(x), device=x.device)
        z = x
        for st in self.steps:
            z, d = st(z); ld = ld + d
        return z, ld

    def inverse(self, z):
        x = z
        for st in reversed(self.steps):
            x = st.inverse(x)
        return x

    def log_prob(self, x):
        z, ld = self(x)
        base = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(1)
        return base + ld

    def fit(self, X, epochs: int = 400, batch: int = 256, lr: float = 5e-3):
        dev = get_device()
        self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        # one forward pass to trigger actnorm data-dependent init
        self.train()
        with torch.no_grad():
            self(X[:batch])
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.history = []
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=dev)
            tot = 0.0
            for s in range(0, len(X), batch):
                nll = -self.log_prob(X[perm[s:s + batch]]).mean()
                opt.zero_grad(); nll.backward(); opt.step()
                tot += nll.item()
            self.history.append(tot / max(1, len(X) // batch))
        return self

    @torch.no_grad()
    def sample(self, n: int):
        dev = next(self.parameters()).device
        z = torch.randn(n, self.dim, device=dev)
        return self.inverse(z).cpu().numpy()


# ---------------------------------------------------------------------------
# Demo — fit the 2-D two-moons density
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)  # tiny model: 1 thread avoids CPU thrashing
    from sklearn.datasets import make_moons
    X, _ = make_moons(2000, noise=0.05, random_state=SEED)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)

    m = Glow(dim=2, n_steps=6).fit(X, epochs=400)
    ll = m.log_prob(torch.tensor(X)).mean().item()
    print(f"Glow final NLL = {m.history[-1]:.3f}  (mean log-lik = {ll:.3f})")

    s = m.sample(2000)
    print(f"  data    mean={X.mean(0).round(2)}  std={X.std(0).round(2)}")
    print(f"  samples mean={s.mean(0).round(2)}  std={s.std(0).round(2)}")


if __name__ == "__main__":
    demo()

"""
RealNVP — exact-likelihood generative modelling with affine coupling layers
============================================================================
A normalizing flow transforms a simple base density (a Gaussian) into a complex
data density through a stack of *invertible* maps. RealNVP's key building block is
the **affine coupling layer**: split the input in two, leave one half untouched,
and scale-and-shift the other half by functions of the untouched half. This makes
both the inverse and the Jacobian determinant trivial -- the Jacobian is
triangular, so its determinant is just the product of the scales. That gives an
*exact* log-likelihood we can maximize directly.

Variants implemented here:
    - Affine coupling layers with alternating binary masks
    - A stack of couplings (a small flow) with a standard-normal base
    - Exact log-likelihood training and sampling by running the flow backward

Training techniques demonstrated:
    - Change of variables / exact likelihood (no bound, unlike VAEs)
    - The triangular-Jacobian trick that makes the log-det cheap
    - tanh-bounded log-scale for numerical stability

References:
    - Dinh, Sohl-Dickstein, Bengio (2017), "Density Estimation using Real NVP"
    - Dinh, Krueger, Bengio (2015), "NICE" (additive coupling predecessor)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# The coupling-layer Jacobian (NumPy sketch).
# ---------------------------------------------------------------------------
# Split x = (x_a, x_b). Keep x_a; transform x_b -> y_b = x_b * exp(s(x_a)) + t(x_a).
# The Jacobian of (x_a, x_b) -> (x_a, y_b) is block-triangular:
#
#     [ I            0          ]
#     [ d y_b/d x_a  diag(e^s)  ]
#
# Its determinant is the product of the diagonal of the lower-right block, so
#     log|det J| = sum(s(x_a))     -- the off-diagonal block never matters.
def _coupling_logdet_numpy(s):
    """log|det J| of one affine coupling = sum of the log-scales s."""
    return np.sum(s, axis=-1)


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


class AffineCoupling(nn.Module):
    r"""
    One affine coupling layer with a fixed binary mask b in {0,1}^d.

    Forward (data -> latent):
        x_a = b * x                      (kept unchanged)
        s, t = NN(x_a)                   (computed from the kept part only)
        y    = x_a + (1-b) * ( x * exp(s) + t )
        log|det| = sum over the transformed dims of s
    Inverse (latent -> data) just solves the affine map; the NN sees x_a = b*y = b*x
    unchanged, so it is exactly invertible without inverting the NN.
    """

    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 64):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * dim))           # outputs (s, t) stacked
        self.dim = dim
        # scale the raw log-scale by a learned factor + tanh -> bounded, stable
        self.scale = nn.Parameter(torch.zeros(dim))

    def _st(self, x_a: torch.Tensor):
        h = self.net(x_a)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s) * self.scale            # bounded log-scale
        return s, t

    def forward(self, x: torch.Tensor):
        """data -> latent; returns (y, log|det J|)."""
        x_a = x * self.mask
        s, t = self._st(x_a)
        s = s * (1 - self.mask); t = t * (1 - self.mask)   # only transform the other half
        y = x_a + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=1)
        return y, log_det

    def inverse(self, y: torch.Tensor):
        """latent -> data (exact)."""
        y_a = y * self.mask
        s, t = self._st(y_a)
        s = s * (1 - self.mask); t = t * (1 - self.mask)
        x = y_a + (1 - self.mask) * ((y - t) * torch.exp(-s))
        return x


class RealNVP(nn.Module):
    """A small stack of affine couplings with alternating masks, N(0,I) base."""

    def __init__(self, dim: int = 2, n_couplings: int = 6, hidden: int = 64):
        super().__init__()
        masks = []
        for i in range(n_couplings):
            m = torch.zeros(dim)
            m[i % 2::2] = 1.0                      # alternate which half is kept
            masks.append(m)
        self.layers = nn.ModuleList(
            AffineCoupling(dim, masks[i], hidden) for i in range(n_couplings))
        self.dim = dim

    def forward(self, x: torch.Tensor):
        """data -> latent z, accumulating the total log|det|."""
        log_det = torch.zeros(len(x), device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z: torch.Tensor):
        """latent z -> data x (run the stack backward)."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Exact log p(x) = log N(z;0,I) + log|det dz/dx|  (change of variables)."""
        z, log_det = self(x)
        base = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=1)   # standard normal
        return base + log_det

    def fit(self, X, epochs: int = 400, batch: int = 256, lr: float = 5e-3):
        dev = get_device()
        self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
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
# Demo — fit the 2-D two-moons density (exact likelihood)
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)  # tiny model: 1 thread avoids CPU thrashing
    from sklearn.datasets import make_moons
    X, _ = make_moons(2000, noise=0.05, random_state=SEED)
    X = ((X - X.mean(0)) / X.std(0)).astype(np.float32)

    m = RealNVP(dim=2, n_couplings=6).fit(X, epochs=400)
    ll = m.log_prob(torch.tensor(X)).mean().item()
    print(f"RealNVP final NLL = {m.history[-1]:.3f}  (mean log-lik = {ll:.3f})")

    s = m.sample(2000)
    print(f"  data    mean={X.mean(0).round(2)}  std={X.std(0).round(2)}")
    print(f"  samples mean={s.mean(0).round(2)}  std={s.std(0).round(2)}")


if __name__ == "__main__":
    demo()

"""
Vector-Quantized VAE (VQ-VAE)
=============================
A VAE whose latent space is **discrete**: the encoder output is snapped to the
nearest vector in a learned codebook of K embeddings, and the decoder
reconstructs from those quantized codes. Because nearest-neighbour lookup has no
useful gradient, training relies on the **straight-through estimator** (copy the
decoder gradient past the quantizer) plus two auxiliary losses that move the
codebook and the encoder toward each other.

Variants implemented here:
    - VQ-VAE with the original codebook + commitment loss (van den Oord 2017)
    - Optional exponential-moving-average (EMA) codebook updates (a flag) — the
      common, more stable alternative to the L2 codebook loss
    - A flattened-MLP VQ-VAE on 8x8 digits (tiny / CPU friendly)

Training techniques demonstrated:
    - STRAIGHT-THROUGH ESTIMATOR for the non-differentiable argmin
    - Commitment loss (stop-gradient trick) to stop encoder/codebook drift
    - EMA codebook updates as a variance-reduced alternative

References:
    - van den Oord, Vinyals, Kavukcuoglu (2017), "Neural Discrete Representation
      Learning" (VQ-VAE)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# The straight-through estimator, made explicit (NumPy sketch)
# ---------------------------------------------------------------------------
# Quantization z_q = e_k where k = argmin_j ||z_e - e_j|| has gradient 0 almost
# everywhere (it is piecewise constant). The straight-through trick *defines*
# the backward pass as the identity:
#
#     forward:   z_q = e_k
#     backward:  d L / d z_e  :=  d L / d z_q      (copy gradient through)
#
# In autograd frameworks this is implemented with the algebraic identity
#     z_q = z_e + stop_grad(e_k - z_e)
# which equals e_k numerically but has d z_q / d z_e = 1.
def _straight_through_numpy(z_e, e_k):
    """forward value e_k, but pretend d z_q/d z_e = 1 (returns value only)."""
    return z_e + (e_k - z_e)  # == e_k; in autograd the (e_k - z_e) term is detached


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


class VectorQuantizer(nn.Module):
    r"""
    Snap each input vector to its nearest codebook entry.

    Codebook  E = [e_1, ..., e_K],  e_j in R^D.  For input z_e:
        k        = argmin_j || z_e - e_j ||^2
        z_q      = e_k                                  (forward)
        d z_q/d z_e := 1                                (straight-through)

    Losses (added to the reconstruction loss):
        codebook   = || sg[z_e] - e_k ||^2     (move codes toward encoder)
        commitment = || z_e - sg[e_k] ||^2     (move encoder toward codes)
    where sg[.] is stop-gradient. With EMA the codebook term is replaced by an
    EMA update of the embeddings and is not backpropagated.
    """

    def __init__(self, num_codes: int = 32, dim: int = 16,
                 commitment: float = 0.25, ema: bool = False, decay: float = 0.99):
        super().__init__()
        self.K, self.D, self.beta = num_codes, dim, commitment
        self.ema, self.decay, self.eps = ema, decay, 1e-5
        self.embedding = nn.Embedding(num_codes, dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        if ema:
            # EMA accumulators are buffers (not trained by the optimizer)
            self.embedding.weight.requires_grad_(False)
            self.register_buffer("cluster_size", torch.zeros(num_codes))
            self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, z_e: torch.Tensor):
        # z_e: (N, D). Squared distances to every code: ||z||^2 - 2 z.e + ||e||^2
        e = self.embedding.weight                                  # (K, D)
        dist = (z_e.pow(2).sum(1, keepdim=True)
                - 2 * z_e @ e.t()
                + e.pow(2).sum(1))                                 # (N, K)
        idx = dist.argmin(1)                                       # (N,)
        z_q = self.embedding(idx)                                  # (N, D)

        # losses
        codebook = F.mse_loss(z_q, z_e.detach())                  # move codes -> enc
        commit = F.mse_loss(z_q.detach(), z_e)                    # move enc -> codes

        if self.ema and self.training:
            self._ema_update(z_e.detach(), idx)
            vq_loss = self.beta * commit                          # no codebook grad
        else:
            vq_loss = codebook + self.beta * commit

        # straight-through: value is z_q, gradient flows to z_e unchanged
        z_q_st = z_e + (z_q - z_e).detach()

        # perplexity = effective number of codes used (a usage diagnostic)
        probs = torch.bincount(idx, minlength=self.K).float() / idx.numel()
        perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
        return z_q_st, vq_loss, idx, perplexity

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.K).type(z_e.dtype)           # (N, K)
        n = onehot.sum(0)                                         # counts per code
        dw = onehot.t() @ z_e                                     # (K, D) sums
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        # Laplace smoothing of the counts to avoid dead/zero codes
        total = self.cluster_size.sum()
        cs = (self.cluster_size + self.eps) / (total + self.K * self.eps) * total
        self.embedding.weight.data.copy_(self.ema_w / cs.unsqueeze(1))


class VQVAE(nn.Module):
    """A tiny MLP VQ-VAE for flattened 8x8 images (in_dim=64)."""

    def __init__(self, in_dim: int = 64, hidden: int = 128, dim: int = 16,
                 num_codes: int = 32, commitment: float = 0.25, ema: bool = False):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, dim))
        self.vq = VectorQuantizer(num_codes, dim, commitment, ema)
        self.dec = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim))

    def forward(self, x: torch.Tensor):
        z_e = self.enc(x)
        z_q, vq_loss, idx, perplex = self.vq(z_e)
        xhat = torch.sigmoid(self.dec(z_q))
        return xhat, vq_loss, idx, perplex

    def loss(self, x: torch.Tensor):
        xhat, vq_loss, idx, perplex = self(x)
        recon = F.binary_cross_entropy(xhat, x, reduction="none").sum(1).mean()
        return recon + vq_loss, recon, vq_loss, perplex

    def fit(self, X, epochs: int = 60, batch: int = 128, lr: float = 2e-3):
        dev = get_device()
        self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.history = []
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=dev)
            tot = 0.0
            for s in range(0, len(X), batch):
                xb = X[perm[s:s + batch]]
                loss, recon, vq, perplex = self.loss(xb)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            self.history.append(tot / max(1, len(X) // batch))
        return self

    @torch.no_grad()
    def reconstruct(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        xhat, _, idx, _ = self(X)
        return xhat.cpu().numpy(), idx.cpu().numpy()


# ---------------------------------------------------------------------------
# Demo — 8x8 digits
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)  # tiny model: 1 thread avoids CPU thrashing
    from sklearn.datasets import load_digits
    X = (load_digits().data / 16.0).astype(np.float32)            # (1797, 64) in [0,1]

    m = VQVAE(64, num_codes=32, dim=16, ema=False).fit(X, epochs=40)
    xhat, idx = m.reconstruct(X[:512])
    mse = float(np.mean((xhat - X[:512]) ** 2))
    used = len(np.unique(idx))
    print(f"VQ-VAE (loss codebook) final loss={m.history[-1]:.3f}  "
          f"recon MSE={mse:.4f}  codes used={used}/32")

    me = VQVAE(64, num_codes=32, dim=16, ema=True).fit(X, epochs=40)
    xhat_e, idx_e = me.reconstruct(X[:512])
    mse_e = float(np.mean((xhat_e - X[:512]) ** 2))
    print(f"VQ-VAE (EMA codebook)  final loss={me.history[-1]:.3f}  "
          f"recon MSE={mse_e:.4f}  codes used={len(np.unique(idx_e))}/32")


if __name__ == "__main__":
    demo()

"""
PixelCNN — autoregressive image modelling with masked convolutions
===================================================================
PixelCNN models an image one pixel at a time: the probability of pixel i depends
only on the pixels *before* it in raster-scan order. The trick that makes this
efficient is the **masked convolution** -- a normal conv whose kernel has its
"future" entries zeroed, so a single forward pass computes the conditional for
every pixel in parallel while never letting information leak from later pixels.

Variants implemented here:
    - Mask type A (first layer: excludes the current pixel) and type B
      (later layers: includes the current pixel's own features)
    - A stack of masked-conv residual-ish blocks for binary 8x8 images
    - Exact log-likelihood (sum of per-pixel Bernoulli log-probs) and ancestral
      sampling

Training techniques demonstrated:
    - Causal masking to enforce the autoregressive factorization
    - Teacher forcing during training (the true pixels are the conditioning)

References:
    - van den Oord et al. (2016), "Pixel Recurrent Neural Networks" (PixelCNN)
    - van den Oord et al. (2016), "Conditional Image Generation with PixelCNN
      Decoders"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# What the mask looks like (NumPy sketch).
# ---------------------------------------------------------------------------
# For a kernel of size (2r+1)x(2r+1), in raster order a pixel may depend on:
#   - all rows above the center row,
#   - in the center row, the columns strictly to the left,
#   - and (mask B only) the center pixel itself.
# Everything to the right / below the center is zeroed:
def _make_mask_numpy(k: int, mask_type: str) -> np.ndarray:
    m = np.ones((k, k), dtype=np.float32)
    c = k // 2
    m[c, c + 1:] = 0.0      # center row, future columns
    m[c + 1:, :] = 0.0      # all rows below
    if mask_type == "A":
        m[c, c] = 0.0       # type A also blocks the current pixel
    return m                # multiply the conv weights by this (broadcast over channels)


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


class MaskedConv2d(nn.Conv2d):
    r"""
    A Conv2d whose kernel is multiplied by a causal mask before every forward
    pass. Type A excludes the center pixel (used once, on the input); type B
    includes it (used in all deeper layers, where the center channel is already a
    *feature* of past pixels, not the pixel value itself).
    """

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ("A", "B")
        k = self.kernel_size[0]
        mask = torch.ones_like(self.weight)            # (out, in, k, k)
        c = k // 2
        mask[:, :, c, c + 1:] = 0.0
        mask[:, :, c + 1:, :] = 0.0
        if mask_type == "A":
            mask[:, :, c, c] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data *= self.mask                  # zero the "future" weights
        return super().forward(x)


class PixelCNN(nn.Module):
    """Tiny PixelCNN for binary 1x8x8 images (Bernoulli per pixel)."""

    def __init__(self, channels: int = 32, n_layers: int = 5, k: int = 5):
        super().__init__()
        layers = [MaskedConv2d("A", 1, channels, k, padding=k // 2), nn.ReLU()]
        for _ in range(n_layers):
            layers += [MaskedConv2d("B", channels, channels, k, padding=k // 2),
                       nn.ReLU()]
        layers += [nn.Conv2d(channels, 1, 1)]          # 1x1: per-pixel logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel Bernoulli logits, shape like x."""
        return self.net(x)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood (bits are summed over pixels)."""
        logits = self(x)
        return F.binary_cross_entropy_with_logits(logits, x, reduction="none") \
            .sum(dim=[1, 2, 3]).mean()

    def fit(self, X, epochs: int = 30, batch: int = 128, lr: float = 1e-3):
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
    def sample(self, n: int, size: int = 8):
        """Ancestral sampling in raster order: fill one pixel at a time."""
        dev = next(self.parameters()).device
        x = torch.zeros(n, 1, size, size, device=dev)
        for i in range(size):
            for j in range(size):
                logits = self(x)
                p = torch.sigmoid(logits[:, :, i, j])
                x[:, :, i, j] = torch.bernoulli(p)     # only this pixel is committed
        return x.cpu().numpy()


# ---------------------------------------------------------------------------
# Demo — binarized 8x8 digits
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)  # tiny model: 1 thread avoids CPU thrashing
    from sklearn.datasets import load_digits
    X = load_digits().data.reshape(-1, 1, 8, 8) / 16.0
    X = (X > 0.3).astype(np.float32)                   # binarize to {0,1}

    m = PixelCNN(channels=32, n_layers=4).fit(X, epochs=25)
    nll = m.history[-1]
    print(f"PixelCNN final NLL = {nll:.2f} nats/image "
          f"({nll / 64:.3f} nats/pixel)")

    s = m.sample(16)
    print(f"  sampled {s.shape[0]} images, mean on-pixels="
          f"{s.mean():.3f} (data {X.mean():.3f})")


if __name__ == "__main__":
    demo()

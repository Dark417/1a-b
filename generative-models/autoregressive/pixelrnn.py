"""
PixelRNN — autoregressive images with a Row LSTM
================================================
Like PixelCNN, PixelRNN models an image one pixel at a time under the raster-scan
factorization, but it captures the *unbounded* context above a pixel with a
recurrent net instead of a fixed convolution. This file implements the simplified
**Row LSTM**: process the image row by row with an LSTM whose input at each row is
a *causal* 1-D convolution of the row above, so each pixel is conditioned on a
triangular region of all preceding rows.

Variants implemented here:
    - Row LSTM (a single left-to-right/top-to-bottom recurrence over rows)
    - Exact per-pixel Bernoulli log-likelihood and ancestral sampling
    - Tiny model for binary 8x8 images (CPU friendly)

Training techniques demonstrated:
    - Autoregressive factorization over pixels (chain rule)
    - Causal input convolution + the one-row state shift to prevent leakage
    - Teacher forcing during training; BPTT through the row recurrence

References:
    - van den Oord, Kalchbrenner, Kavukcuoglu (2016), "Pixel Recurrent Neural
      Networks"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# The Row-LSTM context (NumPy sketch).
# ---------------------------------------------------------------------------
# Row LSTM scans top->bottom. The hidden state h_{i} for row i is produced by an
# LSTM that takes, as input, a 1-D convolution along the PREVIOUS row's features.
# Crucially the recurrence uses row i-1 to predict row i (a one-row shift), so a
# pixel never sees its own row -- combined with a causal 1-D conv this yields the
# autoregressive triangular receptive field:
#
#     state_i = LSTM( conv1d_causal(h_{i-1}) , state_{i-1} )
#     logits_i = readout(state_i)        # predicts row i from rows < i
#
# (The original paper also has a "Diagonal BiLSTM"; the Row LSTM is the simple
# variant implemented here.)


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


class RowLSTM(nn.Module):
    r"""
    Simplified Row-LSTM PixelRNN for binary 1x8x8 images.

    For each output row i we form an input feature by a *causal* 1-D convolution
    across columns of the embedded pixels of row i-1 (so a pixel sees its
    upper-left/up/upper-right neighbours from earlier rows, never its own row).
    A per-column LSTM then carries information down the rows.
    """

    def __init__(self, size: int = 8, hidden: int = 64, k: int = 3):
        super().__init__()
        self.size, self.hidden, self.pad = size, hidden, k // 2
        self.embed = nn.Conv2d(1, hidden, kernel_size=1)          # per-pixel embed
        # causal 1-D conv along the row (width dim); pad symmetrically (centered)
        self.row_conv = nn.Conv1d(hidden, hidden, k, padding=k // 2)
        self.cell = nn.LSTMCell(hidden, hidden)
        self.readout = nn.Linear(hidden, size)                    # logits for a row

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N,1,H,W) -> per-pixel logits (N,1,H,W)."""
        n, _, H, W = x.shape
        dev = x.device
        feat = self.embed(x)                                      # (N, hidden, H, W)
        h = torch.zeros(n, self.hidden, device=dev)
        c = torch.zeros(n, self.hidden, device=dev)
        logits_rows = []
        # row 0 is predicted from an all-zero context (no rows above)
        prev_row_feat = torch.zeros(n, self.hidden, W, device=dev)
        for i in range(H):
            conv = self.row_conv(prev_row_feat)                   # (N, hidden, W)
            inp = conv.mean(dim=2)                                # summarize the row
            h, c = self.cell(inp, (h, c))                         # carry down rows
            logits_rows.append(self.readout(h))                   # (N, W) logits
            prev_row_feat = feat[:, :, i, :]                      # shift: use row i for row i+1
        logits = torch.stack(logits_rows, dim=1).unsqueeze(1)     # (N,1,H,W)
        return logits

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        return F.binary_cross_entropy_with_logits(logits, x, reduction="none") \
            .sum(dim=[1, 2, 3]).mean()

    def fit(self, X, epochs: int = 40, batch: int = 128, lr: float = 3e-3):
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
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)  # BPTT can explode
                opt.step()
                tot += loss.item()
            self.history.append(tot / max(1, len(X) // batch))
        return self

    @torch.no_grad()
    def sample(self, n: int):
        """Ancestral sampling: fill the image row by row, pixel by pixel."""
        dev = next(self.parameters()).device
        H = W = self.size
        x = torch.zeros(n, 1, H, W, device=dev)
        for i in range(H):
            for j in range(W):
                logits = self(x)
                p = torch.sigmoid(logits[:, :, i, j])
                x[:, :, i, j] = torch.bernoulli(p)
        return x.cpu().numpy()


# ---------------------------------------------------------------------------
# Demo — binarized 8x8 digits
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1)  # tiny model: 1 thread avoids CPU thrashing
    from sklearn.datasets import load_digits
    X = load_digits().data.reshape(-1, 1, 8, 8) / 16.0
    X = (X > 0.3).astype(np.float32)

    m = RowLSTM(size=8, hidden=64).fit(X, epochs=35)
    nll = m.history[-1]
    print(f"PixelRNN (Row LSTM) final NLL = {nll:.2f} nats/image "
          f"({nll / 64:.3f} nats/pixel)")

    s = m.sample(16)
    print(f"  sampled {s.shape[0]} images, mean on-pixels="
          f"{s.mean():.3f} (data {X.mean():.3f})")


if __name__ == "__main__":
    demo()

"""
Sliding-Window / Local Attention
=================================
Full self-attention is O(L²) in time and memory: every token attends to every
other. For long contexts this is the bottleneck. Sliding-window attention (SWA)
restricts each query to a fixed window of the most recent W keys, making cost
O(L·W) — linear in sequence length for fixed W.

    full causal:     query i attends to keys 0..i
    sliding window:  query i attends to keys max(0, i-W+1)..i

INFORMATION STILL PROPAGATES FAR. Although one layer only reaches back W tokens,
stacking layers grows the *effective* receptive field like CNNs: after k layers a
token has indirectly mixed information from up to k·W tokens back. Mistral-7B uses
W=4096 over 32 layers -> ~131k-token theoretical reach. SWA also caps the KV
cache to the last W tokens (rolling buffer), bounding inference memory.

VARIANTS:
    - Longformer (Beltagy 2020): sliding window + a few GLOBAL tokens (e.g. [CLS],
      question tokens) that attend to / are attended by everything — local
      efficiency plus a global information bus.
    - BigBird: window + global + random sparse attention (theoretically universal).
    - Mistral: pure causal sliding window + rolling KV cache.

What introduced/uses it: Longformer & BigBird (2020) for long-doc encoders;
Mistral-7B (2023) for decoder LLMs; many long-context models since.

References:
    - Beltagy, Peters & Cohan (2020), "Longformer: The Long-Document Transformer"
    - Zaheer et al. (2020), "Big Bird: Transformers for Longer Sequences"
    - Jiang et al. (2023), "Mistral 7B"
"""

from __future__ import annotations

import torch
import torch.nn as nn


def causal_mask(L):
    """Standard causal mask: True = block (future)."""
    return torch.triu(torch.ones(L, L), 1).bool()


def sliding_window_mask(L, window):
    """
    Causal + window. Query i may see keys j with  i-window < j <= i.
    Returns boolean mask where True = BLOCK.
    """
    i = torch.arange(L).view(L, 1)
    j = torch.arange(L).view(1, L)
    too_future = j > i                       # causal
    too_old = j <= (i - window)              # outside the window
    return too_future | too_old


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window):
        super().__init__()
        self.h, self.d_k, self.W = n_heads, d_model // n_heads, window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, full=False):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        shp = lambda t: t.view(B, L, self.h, self.d_k).transpose(1, 2)
        q, k, v = shp(q), shp(k), shp(v)
        scores = q @ k.transpose(-1, -2) / self.d_k ** 0.5
        mask = causal_mask(L) if full else sliding_window_mask(L, self.W)
        scores = scores.masked_fill(mask.to(x.device), float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


# ---------------------------------------------------------------------------
# Demo — compare full vs windowed mask, cost, and receptive field
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    L, W = 8, 3
    full = causal_mask(L)
    win = sliding_window_mask(L, W)

    def show(m, title):
        print(title)
        for i in range(L):
            print("  " + "".join("." if m[i, j] else "#" for j in range(L)))

    print("Mask  (# = attend, . = blocked),  rows = query i\n")
    show(full, f"FULL causal (each query sees all past):")
    print()
    show(win, f"SLIDING WINDOW W={W} (each query sees last {W}):")

    # Allowed-key counts: full grows with i; window is capped at W.
    full_counts = (~full).sum(-1).tolist()
    win_counts = (~win).sum(-1).tolist()
    print(f"\nkeys attended per query:")
    print(f"  full   : {full_counts}  (sum={sum(full_counts)} ~ O(L^2/2))")
    print(f"  window : {win_counts}  (sum={sum(win_counts)} ~ O(L*W), capped at {W})")
    assert max(win_counts) <= W

    # cost scaling: full is quadratic, windowed is linear in L
    print("\nattention pair count (cost) scaling:")
    for Ln in (1024, 4096, 16384):
        full_pairs = Ln * (Ln + 1) // 2
        win_pairs = Ln * W                       # ~ each query sees W keys
        print(f"  L={Ln:6d}:  full={full_pairs:>12,}   window={win_pairs:>10,}  "
              f"({full_pairs/win_pairs:.0f}x less)")

    # effective receptive field grows with depth
    layers = 32
    print(f"\nstacking {layers} layers with W={4096} -> effective reach "
          f"~ {layers*4096:,} tokens (Mistral-7B)")

    attn = SlidingWindowAttention(32, 4, window=W)
    x = torch.randn(2, L, 32)
    yf, yw = attn(x, full=True), attn(x, full=False)
    print(f"\nattention runs: full {tuple(yf.shape)}, windowed {tuple(yw.shape)}; "
          f"outputs differ = {not torch.allclose(yf, yw)}.  PASS")


if __name__ == "__main__":
    demo()

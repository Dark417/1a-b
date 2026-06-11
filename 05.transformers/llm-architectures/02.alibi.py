"""
ALiBi — Attention with Linear Biases
====================================
ALiBi removes positional embeddings entirely. Instead it adds a static, *linear*
penalty to the attention scores that grows with the distance between query and
key. Closer tokens are favoured; far tokens are penalised more. Because the bias
is just −slope·distance (no learned parameters, no position vectors), a model
trained on short sequences extrapolates to much longer ones at inference time —
ALiBi's headline result.

The bias added to the pre-softmax scores for a query at position i and key at
position j (with j ≤ i in a causal model) is:

    score_ij  +=  −m_h · (i − j)

where m_h is a per-head slope. The slopes are a geometric sequence so different
heads "see" different effective context lengths. For H heads (H a power of two)
the slopes are:

    m_h = 2^{-8h/H}   for h = 1..H   →   1/2, 1/4, 1/8, ... , 1/256

(For non-powers-of-two, ALiBi interpolates extra slopes from the next power of
two — implemented below.) Head 1 has the steepest penalty (very local); the last
head has the gentlest (almost global).

What introduced it: "Train Short, Test Long" (Press et al. 2021); used by
BLOOM, MPT, BloombergGPT, Falcon (early), and others as a RoPE alternative.

References:
    - Press, Smith & Lewis (2021), "Train Short, Test Long: Attention with
      Linear Biases Enables Input Length Extrapolation"
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Geometric slopes m_h = 2^{-8h/H}. For non-power-of-two head counts, take the
    power-of-two slopes then interpolate the remainder (the paper's recipe).
    """
    def pow2_slopes(n):
        start = 2 ** (-8.0 / n)                  # ratio = 2^{-8/n}
        return torch.tensor([start ** (i + 1) for i in range(n)])

    if math.log2(n_heads).is_integer():
        return pow2_slopes(n_heads)
    # nearest lower power of two, then fill from the next power of two
    closest = 2 ** math.floor(math.log2(n_heads))
    base = pow2_slopes(closest)
    extra = pow2_slopes(2 * closest)[0::2][: n_heads - closest]
    return torch.cat([base, extra])


def alibi_bias(n_heads: int, q_len: int, k_len: int,
               causal: bool = True) -> torch.Tensor:
    """
    Returns bias of shape (n_heads, q_len, k_len) to ADD to attention scores.
    bias[h,i,j] = -slope_h * (i - j) ; for causal models j>i is masked to -inf.
    """
    slopes = alibi_slopes(n_heads).view(n_heads, 1, 1)        # (H,1,1)
    i = torch.arange(q_len).view(q_len, 1)
    j = torch.arange(k_len).view(1, k_len)
    distance = (j - i).float()                  # negative for past, 0 on diag
    # We want -slope * (i - j) = slope * (j - i) = slope * distance (≤0 in past)
    bias = slopes * distance                    # (H, q_len, k_len), ≤ 0
    if causal:
        bias = bias.masked_fill(j > i, float("-inf"))
    return bias


class ALiBiAttention(nn.Module):
    """Causal multi-head self-attention using ALiBi instead of position emb."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("slopes", alibi_slopes(n_heads), persistent=False)

    def forward(self, x):                         # x: (B, L, d)
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]          # each (B, H, L, d_k)
        scores = q @ k.transpose(-1, -2) / self.d_k ** 0.5
        scores = scores + alibi_bias(self.h, L, L, causal=True).to(x.device)
        attn = scores.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.h * self.d_k)
        return self.out(out)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    H = 8
    slopes = alibi_slopes(H)
    print(f"slopes for {H} heads (geometric 2^-8h/H):")
    print("  ", [round(s, 4) for s in slopes.tolist()])

    # Bias grows (more negative) with distance — show head 0 (steepest).
    bias = alibi_bias(H, 6, 6, causal=True)
    print("\nALiBi bias, head 0 (rows=query i, cols=key j); -inf = masked future:")
    for i in range(6):
        row = [f"{bias[0,i,j].item():6.2f}" for j in range(6)]
        print("  i=%d " % i, " ".join(row))
    # The penalty for the current query grows with how far back the key is.
    print(f"\nbias[0, query=5, key=5]={bias[0,5,5]:.2f} (self, 0), "
          f"key=0 ={bias[0,5,0]:.2f} (far, more negative) -> grows with distance")
    assert bias[0, 5, 0] < bias[0, 5, 4] < 0

    # Gentler head penalises less at the same distance.
    print(f"\nsame distance (5), head0 slope steep: {bias[0,5,0]:.2f}  "
          f"head7 slope gentle: {bias[7,5,0]:.2f}")

    # Integrate into attention.
    attn = ALiBiAttention(d_model=32, n_heads=H)
    x = torch.randn(2, 6, 32)
    y = attn(x)
    print(f"\nALiBiAttention: in {tuple(x.shape)} -> out {tuple(y.shape)}")

    # Extrapolation: same module, longer sequence, no retraining needed.
    y_long = attn(torch.randn(2, 20, 32))
    print(f"extrapolates to L=20 unchanged: out {tuple(y_long.shape)}  PASS")


if __name__ == "__main__":
    demo()

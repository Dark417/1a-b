"""
FlashAttention — Tiling + Online Softmax (memory-efficient attention)
=====================================================================
Naive attention materializes the full L×L score matrix S = QKᵀ and the L×L
softmax matrix P in (slow) GPU HBM. That O(L²) memory traffic — not the FLOPs —
is the real bottleneck. FlashAttention computes the SAME result while never
writing the L×L matrices to HBM: it streams over blocks of K/V, keeping a running
softmax in fast on-chip SRAM. Memory drops to O(L), and it's faster because it is
IO-bound, not compute-bound.

THE ONLINE (STREAMING) SOFTMAX. Softmax needs a max (for stability) and a sum
over ALL keys — but we want to process keys block by block without seeing them
all first. Trick: maintain a running max m, running denominator ℓ, and running
weighted output o. When a new block arrives with local max m_blk:

    m_new = max(m, m_blk)
    correction = exp(m - m_new)                 # rescale old stats to new max
    ℓ = ℓ·correction + Σ exp(S_blk - m_new)
    o = o·correction + Σ exp(S_blk - m_new)·V_blk
    ... after all blocks ...
    output = o / ℓ

Each update rescales the accumulated output by exp(old_max - new_max) so the
final normalization is exactly the standard softmax — just computed incrementally.
This is mathematically identical to softmax, only reorganized to be tiled.

FlashAttention-2 improves parallelism/work partitioning; FlashAttention-3 adds
FP8 and Hopper-specific async. The math below is the FA-1 core, in pure PyTorch
(no custom kernel) — for teaching the algorithm, not for speed.

What introduced it: Dao et al. (2022). Now the default attention kernel in
PyTorch SDPA, vLLM, and essentially all efficient LLM training/inference.

References:
    - Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact
      Attention with IO-Awareness"
    - Dao (2023), "FlashAttention-2"
    - Milakov & Gimelshein (2018), "Online normalizer calculation for softmax"
"""

from __future__ import annotations

import math
import torch


def naive_attention(Q, K, V, causal=True):
    """Reference: materializes the full score matrix (the thing FA avoids)."""
    L, d = Q.shape
    scores = Q @ K.t() / math.sqrt(d)               # (L, L)  <- O(L^2) memory
    if causal:
        mask = torch.triu(torch.ones(L, L), 1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(-1) @ V


def flash_attention(Q, K, V, block=2, causal=True):
    """
    Tiled, online-softmax attention. Never builds the full L×L matrix; instead
    streams K/V in blocks, keeping running (max m, denom l, output acc) per query.
    Pure-PyTorch reference of the FlashAttention-1 algorithm.
    """
    L, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros(L, d)
    m = torch.full((L,), float("-inf"))             # running max per query
    l = torch.zeros(L)                              # running denominator per query

    # outer loop over KEY/VALUE blocks (these are what stay off-HBM in real FA)
    for ks in range(0, L, block):
        ke = min(ks + block, L)
        Kb, Vb = K[ks:ke], V[ks:ke]                 # (b, d)
        S = (Q @ Kb.t()) * scale                    # (L, b) — only one block wide
        if causal:
            qi = torch.arange(L).view(L, 1)
            kj = torch.arange(ks, ke).view(1, -1)
            S = S.masked_fill(kj > qi, float("-inf"))

        m_blk = S.max(dim=1).values                 # (L,)
        m_new = torch.maximum(m, m_blk)
        # guard rows that are still all -inf (no valid key yet under causal mask)
        corr = torch.exp(torch.nan_to_num(m - m_new, nan=0.0))   # rescale old
        p = torch.exp(S - m_new.unsqueeze(1))       # (L, b), nan-safe (-inf->0)
        p = torch.nan_to_num(p, nan=0.0)

        l = l * corr + p.sum(1)
        O = O * corr.unsqueeze(1) + p @ Vb
        m = m_new

    return O / l.clamp_min(1e-20).unsqueeze(1)


# ---------------------------------------------------------------------------
# Demo — assert flash == naive
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    L, d = 8, 16
    Q, K, V = torch.randn(L, d), torch.randn(L, d), torch.randn(L, d)

    ref = naive_attention(Q, K, V, causal=True)
    for blk in (1, 2, 3, 8):                          # any block size, same answer
        flash = flash_attention(Q, K, V, block=blk, causal=True)
        diff = (ref - flash).abs().max().item()
        print(f"block={blk}:  max |naive - flash| = {diff:.2e}")
        assert torch.allclose(ref, flash, atol=1e-5), f"mismatch at block={blk}"
    print("Online-softmax tiling reproduces exact attention.  PASS")

    # non-causal also matches
    rn = naive_attention(Q, K, V, causal=False)
    fn = flash_attention(Q, K, V, block=3, causal=False)
    assert torch.allclose(rn, fn, atol=1e-5)
    print("non-causal also exact.  PASS")

    # memory argument: naive needs L*L floats; flash needs L*block.
    print(f"\nscore-matrix memory:  naive O(L^2) vs flash O(L*block)")
    for Ln in (1024, 8192, 32768):
        naive_mb = Ln * Ln * 4 / 1e6
        flash_mb = Ln * 128 * 4 / 1e6                 # block~128 in real kernels
        print(f"  L={Ln:6d}:  naive {naive_mb:9.1f} MB   flash {flash_mb:6.1f} MB "
              f"({naive_mb/flash_mb:.0f}x less)")


if __name__ == "__main__":
    demo()

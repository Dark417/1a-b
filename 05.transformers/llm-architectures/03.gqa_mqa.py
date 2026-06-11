"""
MHA vs GQA vs MQA — sharing key/value heads
===========================================
In vanilla Multi-Head Attention (MHA) every one of the H query heads has its own
key and value head. At inference the KV cache stores K and V for *every* head and
every past token, and that cache — not the math — is what dominates memory and
bandwidth during autoregressive decoding.

The fix: let several query heads SHARE one key/value head.

    MHA  (Multi-Head)    : n_kv_heads = n_heads          (no sharing)
    GQA  (Grouped-Query) : 1 < n_kv_heads < n_heads      (groups share a KV head)
    MQA  (Multi-Query)   : n_kv_heads = 1                (all heads share one KV)

KV-cache size is proportional to n_kv_heads, so:
    MQA cache  = MHA cache / H        (smallest, but some quality loss)
    GQA cache  = MHA cache / (H/g)    (g = group size; the sweet spot)

One module below implements all three via a single `n_kv_heads` parameter. The
shared KV heads are simply repeat-interleaved up to H before the attention dot
product (`torch.repeat_interleave`), so the rest of attention is unchanged.

What introduced it: MQA — Shazeer (2019); GQA — Ainslie et al. (2023). GQA is
the default in LLaMA-2 70B, LLaMA-3, Mistral, Qwen2, Gemma, and most modern LLMs;
MQA is used in PaLM and Falcon.

References:
    - Shazeer (2019), "Fast Transformer Decoding: One Write-Head is All You Need"
    - Ainslie et al. (2023), "GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    MHA / GQA / MQA in one module, selected by n_kv_heads.
        n_kv_heads == n_heads  -> MHA
        1 < n_kv_heads < n_heads -> GQA
        n_kv_heads == 1        -> MQA
    """

    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.h, self.kv_h = n_heads, n_kv_heads
        self.d_k = d_model // n_heads
        self.group = n_heads // n_kv_heads          # query heads per KV head

        # Q projects to H heads; K,V project to only n_kv_heads heads.
        self.Wq = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.Wk = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.Wo = nn.Linear(n_heads * self.d_k, d_model, bias=False)

    def forward(self, x, causal=True):
        B, L, _ = x.shape
        q = self.Wq(x).view(B, L, self.h, self.d_k).transpose(1, 2)     # (B,H,L,d)
        k = self.Wk(x).view(B, L, self.kv_h, self.d_k).transpose(1, 2)  # (B,kv,L,d)
        v = self.Wv(x).view(B, L, self.kv_h, self.d_k).transpose(1, 2)

        # Broadcast the shared KV heads up to H by repeating each group times.
        k = k.repeat_interleave(self.group, dim=1)     # (B,H,L,d)
        v = v.repeat_interleave(self.group, dim=1)

        scores = q @ k.transpose(-1, -2) / self.d_k ** 0.5
        if causal:
            mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1)
        return self.Wo(out)

    def kv_cache_bytes(self, seq_len, dtype_bytes=2):
        """K + V cache for `seq_len` tokens (per sequence), in bytes."""
        return 2 * self.kv_h * self.d_k * seq_len * dtype_bytes


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    d_model, H, L = 64, 8, 16
    x = torch.randn(2, L, d_model)

    configs = [("MHA", H), ("GQA", 2), ("MQA", 1)]
    print(f"{'variant':6} {'n_kv':>4} {'params':>8} {'KVcache@4k':>12} {'out shape':>14}")
    for name, kv in configs:
        m = GroupedQueryAttention(d_model, H, n_kv_heads=kv)
        y = m(x)
        params = sum(p.numel() for p in m.parameters())
        cache = m.kv_cache_bytes(4096)
        print(f"{name:6} {kv:>4} {params:>8} {cache:>10} B {str(tuple(y.shape)):>14}")
        assert y.shape == (2, L, d_model)            # all give identical shape

    # Cache shrinkage ratios.
    full = GroupedQueryAttention(d_model, H, n_kv_heads=H).kv_cache_bytes(4096)
    gqa = GroupedQueryAttention(d_model, H, n_kv_heads=2).kv_cache_bytes(4096)
    mqa = GroupedQueryAttention(d_model, H, n_kv_heads=1).kv_cache_bytes(4096)
    print(f"\nKV-cache vs MHA:  GQA(2) = {full//gqa}x smaller,  "
          f"MQA = {full//mqa}x smaller")
    print("All three produce equal output shape; only KV heads differ.  PASS")


if __name__ == "__main__":
    demo()

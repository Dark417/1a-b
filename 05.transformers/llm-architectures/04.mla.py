"""
MLA — Multi-head Latent Attention (DeepSeek-V2 / V3)
====================================================
GQA/MQA shrink the KV cache by SHARING key/value heads across queries — at some
cost in quality. MLA takes a different route: it keeps full per-head expressivity
but compresses what is *cached* into a small low-rank LATENT vector, then projects
that latent back up to per-head K and V on the fly. The cache stores the latent
(dim d_c, e.g. 512) instead of full K and V (dim H·d_k, e.g. 128·128), so the
cache is tiny while the model still behaves like full multi-head attention.

The mechanism:

    1. Down-project the token to a compressed KV latent:
           c_KV = W_DKV · x          (d_model -> d_c,  d_c << H·d_k)
       *This latent is the only thing kept in the KV cache.*
    2. At attention time, up-project the latent to per-head K and V:
           K = W_UK · c_KV           (d_c -> H·d_k)
           V = W_UV · c_KV
    3. Queries are likewise (optionally) compressed via a query latent c_Q, then
       up-projected, which shrinks activation memory during training.

DECOUPLED RoPE — the catch. RoPE rotates K by an absolute-position-dependent
matrix R_m. If K = W_UK·c_KV, then R_m·K can't be folded into the cached latent
(R_m depends on position, breaking the low-rank absorb trick). DeepSeek's fix:
split each head into a NoPE part (carried by the compressed latent, no rotation)
and a small decoupled RoPE part computed from a *separate* shared key path that
DOES get rotated. Query/key = concat(content_part, rope_part); only the content
part flows through compression, only the rope part carries position. This keeps
both low-rank caching AND rotary position information.

    q = [q_C ; q_R],   k = [k_C ; k_R]
    score = q_C·k_C  +  q_R·k_R            (k_R shared across heads, rotated)

What introduced it: DeepSeek-V2 (2024); reused in DeepSeek-V3 / R1. Achieves
KV-cache ~1/10th of MHA at comparable or better quality.

References:
    - DeepSeek-AI (2024), "DeepSeek-V2: A Strong, Economical, and Efficient
      Mixture-of-Experts Language Model"
    - DeepSeek-AI (2024), "DeepSeek-V3 Technical Report"
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rope_cache(L, dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.outer(torch.arange(L).float(), inv)
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat([-x2, x1], -1)


def apply_rope(x, cos, sin):                       # x: (B,H,L,dim)
    return x * cos + rotate_half(x) * sin


class MultiHeadLatentAttention(nn.Module):
    """
    MLA with low-rank KV compression + decoupled RoPE.
        d_c     : KV latent (compressed) dim — this is what gets cached
        d_head  : NoPE (content) per-head dim
        d_rope  : decoupled RoPE per-head dim (small)
    """

    def __init__(self, d_model, n_heads, d_c=64, d_head=16, d_rope=8):
        super().__init__()
        self.h, self.d_head, self.d_rope, self.d_c = n_heads, d_head, d_rope, d_c

        # --- KV path: down to latent c_KV, then up to per-head content K and V ---
        self.W_DKV = nn.Linear(d_model, d_c, bias=False)            # compress
        self.W_UK = nn.Linear(d_c, n_heads * d_head, bias=False)    # K content up
        self.W_UV = nn.Linear(d_c, n_heads * d_head, bias=False)    # V up
        # --- decoupled RoPE key: a single shared rotated key (not per head) ---
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)

        # --- Query path: content + decoupled rope parts ---
        self.W_Q = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.W_QR = nn.Linear(d_model, n_heads * d_rope, bias=False)

        self.W_O = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.scale = (d_head + d_rope) ** -0.5

    def forward(self, x, causal=True):
        B, L, _ = x.shape
        H, dh, dr = self.h, self.d_head, self.d_rope

        c_KV = self.W_DKV(x)                                   # (B,L,d_c)  <-- cached
        k_C = self.W_UK(c_KV).view(B, L, H, dh).transpose(1, 2)   # (B,H,L,dh)
        v = self.W_UV(c_KV).view(B, L, H, dh).transpose(1, 2)
        q_C = self.W_Q(x).view(B, L, H, dh).transpose(1, 2)

        cos, sin = rope_cache(L, dr)
        cos, sin = cos[None, None], sin[None, None]            # (1,1,L,dr)
        k_R = apply_rope(self.W_KR(x).view(B, L, 1, dr).transpose(1, 2), cos, sin)
        k_R = k_R.expand(B, H, L, dr)                          # shared across heads
        q_R = apply_rope(self.W_QR(x).view(B, L, H, dr).transpose(1, 2), cos, sin)

        # concat content + rope parts; score is the sum of both inner products
        q = torch.cat([q_C, q_R], -1)
        k = torch.cat([k_C, k_R], -1)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        if causal:
            m = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
            scores = scores.masked_fill(m, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, L, H * dh)
        return self.W_O(out)

    def kv_cache_per_token_bytes(self, dtype_bytes=2):
        """MLA caches the latent (d_c) + the shared rope key (d_rope)."""
        return (self.d_c + self.d_rope) * dtype_bytes


class PlainMHA(nn.Module):
    """Reference MHA whose cache stores full per-head K and V."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def kv_cache_per_token_bytes(self, dtype_bytes=2):
        return 2 * self.h * self.d_k * dtype_bytes              # K + V, all heads


# ---------------------------------------------------------------------------
# Demo — KV-cache shrinkage vs MHA
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    d_model, H, L = 128, 8, 16
    x = torch.randn(2, L, d_model)

    mla = MultiHeadLatentAttention(d_model, H, d_c=64, d_head=16, d_rope=8)
    mha = PlainMHA(d_model, H)

    y = mla(x)
    print(f"MLA forward: in {tuple(x.shape)} -> out {tuple(y.shape)}")
    assert y.shape == (2, L, d_model)

    mla_b = mla.kv_cache_per_token_bytes()
    mha_b = mha.kv_cache_per_token_bytes()
    print(f"\nKV cache per token (fp16):")
    print(f"  MHA : 2 * H({H}) * d_k({d_model//H}) * 2B = {mha_b} B")
    print(f"  MLA : (d_c({mla.d_c}) + d_rope({mla.d_rope})) * 2B = {mla_b} B")
    print(f"  -> MLA cache is {mha_b / mla_b:.1f}x smaller than MHA")

    # at 4096 context
    seq = 4096
    print(f"\nAt {seq} tokens/seq:  MHA {mha_b*seq/1024:.0f} KB  vs  "
          f"MLA {mla_b*seq/1024:.0f} KB")
    assert mla_b < mha_b
    print("MLA keeps full multi-head expressivity with a fraction of cache.  PASS")


if __name__ == "__main__":
    demo()

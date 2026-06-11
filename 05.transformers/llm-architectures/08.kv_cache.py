"""
KV Cache — Incremental Autoregressive Decoding
==============================================
During generation the model emits one token at a time, each time running
attention over the *entire* prefix. Recomputing K and V for all past tokens at
every step is O(L²) wasted work — but K and V for already-seen tokens never
change. The KV cache stores them: at step t you compute K_t, V_t for the single
new token, append to the cache, and attend the new query against the whole cache.

    step t:  q_t, k_t, v_t = project(x_t)
             K = cat(K_cache, k_t) ;  V = cat(V_cache, v_t)
             out_t = softmax(q_t Kᵀ / √d) V          (q is length-1!)
    -> per-step cost drops from O(t·d) recompute to O(d); generation is O(L) total.

The cache size grows linearly with sequence length and dominates inference memory:
    bytes = 2 (K&V) · n_layers · n_kv_heads · d_head · seq_len · dtype_bytes · batch
This is exactly why GQA/MQA/MLA (which shrink n_kv_heads / cache width) matter.

PAGED ATTENTION (vLLM, conceptual). A naive cache reserves a contiguous buffer of
max_len per sequence -> huge internal fragmentation (most slots unused) and no
sharing. PagedAttention borrows OS virtual memory: the KV cache is split into
fixed-size BLOCKS (e.g. 16 tokens) stored in a pool; each sequence keeps a "block
table" mapping logical positions to physical blocks. Benefits:
    - near-zero fragmentation (allocate blocks on demand),
    - copy-on-write SHARING of a common prompt prefix across beams/samples,
    - dynamic growth without reserving max_len up front.
The attention kernel gathers K/V through the block table instead of a flat slice.

What introduced it: standard since GPT-2-era decoding; PagedAttention from
vLLM (Kwon et al. 2023).

References:
    - Kwon et al. (2023), "Efficient Memory Management for LLM Serving with
      PagedAttention" (vLLM)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CachedSelfAttention(nn.Module):
    """Causal single-head-group attention supporting incremental decoding."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def _proj(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        shp = lambda t: t.view(B, L, self.h, self.d_k).transpose(1, 2)
        return shp(q), shp(k), shp(v)

    def forward(self, x, cache=None):
        """
        x: (B, L, d). If cache=(K,V) given, x is the NEW tokens only; we append.
        Returns (out, new_cache). Causal within the appended block.
        """
        q, k, v = self._proj(x)
        if cache is not None:
            K_prev, V_prev = cache
            k = torch.cat([K_prev, k], dim=2)         # append along time
            v = torch.cat([V_prev, v], dim=2)
        Lq, Lk = q.size(2), k.size(2)
        scores = q @ k.transpose(-1, -2) / self.d_k ** 0.5
        # causal: query at absolute position (Lk-Lq+i) may see keys 0..that index
        offset = Lk - Lq
        idx_q = torch.arange(Lq).view(Lq, 1) + offset
        idx_k = torch.arange(Lk).view(1, Lk)
        scores = scores.masked_fill(idx_k > idx_q, float("-inf"))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(x.size(0), Lq, -1)
        return self.o(out), (k, v)


def cache_bytes(n_layers, n_kv_heads, d_head, seq_len, batch=1, dtype_bytes=2):
    """Total KV-cache memory for a model. 2 = K and V."""
    return 2 * n_layers * n_kv_heads * d_head * seq_len * batch * dtype_bytes


# ---------------------------------------------------------------------------
# Demo — cached incremental == full recompute; show memory growth
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    d_model, H, L = 32, 4, 8
    attn = CachedSelfAttention(d_model, H).eval()
    x = torch.randn(1, L, d_model)

    with torch.no_grad():
        # (a) FULL: process whole sequence at once (training-style).
        full_out, _ = attn(x)

        # (b) INCREMENTAL: feed one token at a time, growing the cache.
        cache = None
        inc_outs = []
        for t in range(L):
            o, cache = attn(x[:, t:t + 1], cache)
            inc_outs.append(o)
        inc_out = torch.cat(inc_outs, dim=1)

    diff = (full_out - inc_out).abs().max().item()
    print(f"max |full - cached_incremental| = {diff:.2e}")
    assert torch.allclose(full_out, inc_out, atol=1e-5), "cache must match recompute"
    print("Incremental KV-cache decoding matches full recompute exactly.  PASS")

    # cache grows linearly: shape after t steps is (B,H,t,d_k)
    print(f"\nfinal cache K shape: {tuple(cache[0].shape)}  (B,H,L={L},d_k)")
    print("\nKV-cache memory growth (Llama-2-7B-ish: 32 layers, 32 KV heads, d=128):")
    for s in (512, 2048, 8192, 32768):
        gb = cache_bytes(32, 32, 128, s) / 1e9
        print(f"  seq_len={s:6d}  ->  {gb:6.3f} GB / sequence (fp16)")
    # GQA shrinks it: 8 KV heads instead of 32 -> 4x less
    gqa = cache_bytes(32, 8, 128, 8192) / 1e9
    mha = cache_bytes(32, 32, 128, 8192) / 1e9
    print(f"\n@8192: MHA(32 kv)={mha:.3f} GB vs GQA(8 kv)={gqa:.3f} GB "
          f"-> {mha/gqa:.0f}x smaller (why GQA/MLA exist)")


if __name__ == "__main__":
    demo()

"""04 · Triton fused attention — FlashAttention-style, in Triton.

Standard attention is

    O = softmax(Q Kᵀ / √d) V        Q,K,V: (N, d)

The textbook implementation materializes the N×N score matrix S = QKᵀ. For a
sequence of length N that is O(N²) memory — the bottleneck for long context.
**FlashAttention** (Dao et al., 2022) never writes S to global memory: it tiles
Q, K, V and computes softmax **online**, keeping only a running max `m`, running
denominator `l`, and the running output `acc` per query block. Memory drops to
O(N), and because everything stays in SRAM it is also faster.

Online softmax (the core trick)
-------------------------------
Process K/V in blocks. Maintain, per query row:
    m   = running max of scores seen so far     (init -inf)
    l   = running sum of exp(score - m)         (init 0)
    acc = running weighted sum of V             (init 0)
For a new score block s:
    m_new = max(m, rowmax(s))
    p     = exp(s - m_new)                       # rescale new block
    alpha = exp(m - m_new)                       # correction for the OLD stats
    l   = l*alpha + rowsum(p)
    acc = acc*alpha + p @ V_block
    m   = m_new
At the end, O = acc / l. The `alpha` rescaling is what makes a streaming softmax
*exact* — it retroactively corrects earlier blocks when a bigger max appears.
This is Milakov & Gimelshein's online softmax lifted into the attention loop.

The Triton kernel below implements exactly this: one program per (batch·head,
query-block); an inner loop over key/value blocks doing `tl.dot` for QKᵀ and for
P·V, with the running (m, l, acc) update. This is the heart of the production
`flash-attention` Triton kernel, trimmed to the non-causal forward pass.

GPU-less fallback: kernel defined and run only under CUDA; on CPU we validate the
*online-softmax math itself* with a runnable NumPy/torch streaming reference
against dense attention, so you can see the algorithm is exact — and the file
exits 0.

References
----------
- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness", NeurIPS 2022. https://arxiv.org/abs/2205.14135
- Dao, "FlashAttention-2", 2023. https://arxiv.org/abs/2307.08691
- Triton fused-attention tutorial:
  https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018.
  https://arxiv.org/abs/1805.02867
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q, K, V, Out,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    N, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
):
    # one program == one block of query rows
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # running statistics for the online softmax
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        n_mask = offs_n < N
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # scores: BLOCK_M x BLOCK_N
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(n_mask[None, :], s, float("-inf"))   # mask padded keys

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)                       # correct old stats
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < N)


def triton_attention(q, k, v):
    N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M),)
    attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        N, scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
    )
    return out


def dense_attention(q, k, v):
    """Reference O = softmax(QKᵀ/√d) V, materializing the full score matrix."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = (q @ k.transpose(-1, -2)) * scale
    p = torch.softmax(s, dim=-1)
    return p @ v


def streaming_attention(q, k, v, block_n=16):
    """Pure-torch CPU implementation of the SAME online-softmax algorithm the
    Triton kernel uses — proves the streaming recurrence is exact vs dense."""
    N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    m_i = torch.full((N,), float("-inf"))
    l_i = torch.zeros(N)
    acc = torch.zeros(N, D)
    for start in range(0, N, block_n):
        kb = k[start:start + block_n]
        vb = v[start:start + block_n]
        s = (q @ kb.t()) * scale                  # (N, block_n)
        m_new = torch.maximum(m_i, s.max(dim=1).values)
        p = torch.exp(s - m_new[:, None])
        alpha = torch.exp(m_i - m_new)
        l_i = l_i * alpha + p.sum(dim=1)
        acc = acc * alpha[:, None] + p @ vb
        m_i = m_new
    return acc / l_i[:, None]


def main():
    torch.manual_seed(0)
    N, D = 128, 32
    q = torch.randn(N, D)
    k = torch.randn(N, D)
    v = torch.randn(N, D)
    ref = dense_attention(q, k, v)

    if torch.cuda.is_available():
        got = triton_attention(q.cuda(), k.cuda(), v.cuda()).cpu()
        ok = torch.allclose(got, ref, atol=1e-3, rtol=1e-3)
        print(f"[Triton kernel on GPU] flash-attention N={N} D={D}, "
              f"max|err|={(got - ref).abs().max():.2e}")
        print("PASS" if ok else "FAIL")
        assert ok
    else:
        print("Triton kernels require a CUDA GPU — running the CPU "
              "online-softmax reference (same algorithm) instead.")
        got = streaming_attention(q, k, v)
        ok = torch.allclose(got, ref, atol=1e-5)
        print(f"[CPU online-softmax reference] flash-attention N={N} D={D}, "
              f"max|err| vs dense={(got - ref).abs().max():.2e}")
        print("PASS" if ok else "FAIL")
        assert ok


if __name__ == "__main__":
    main()

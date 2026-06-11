# 04 · Triton — a Python DSL for GPU kernels at block granularity

[Triton](https://triton-lang.org/) (OpenAI) is a Python-embedded language and
compiler for writing high-performance GPU kernels. It sits between "call cuBLAS"
and "hand-write CUDA C++": you express the **block** structure of your
computation, and the compiler handles the painful parts — thread mapping, memory
coalescing, shared-memory staging, register allocation, double-buffering, and
targeting Tensor Cores. Triton kernels routinely match or beat hand-tuned CUDA
for the fused, memory-bound kernels that dominate deep learning, which is why
PyTorch's `torch.compile` (Inductor backend) generates Triton.

## The model: blocks, not threads

CUDA: "what does **thread** `i` do?" Triton: "what does **program** (one block of
elements) `pid` do?" You manipulate whole tensors of a block at a time —
`tl.load`, `tl.store`, `tl.dot`, `tl.max`, `tl.sum` — and Triton vectorizes
across the warp(s) under the hood. The recurring skeleton:

```python
@triton.jit
def kernel(x_ptr, ..., N, BLOCK: tl.constexpr):
    pid     = tl.program_id(0)                 # this block's index
    offs    = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offs < N                         # guard the ragged tail
    x       = tl.load(x_ptr + offs, mask=mask) # coalesced, masked
    ...                                        # block-level math
    tl.store(out_ptr + offs, result, mask=mask)
```

You launch with a **grid** of programs: `kernel[grid](...)`, where
`grid = (triton.cdiv(N, BLOCK),)`. `tl.constexpr` arguments are compile-time
constants (block sizes) Triton specializes on.

## Why these files use a CPU-reference fallback

**Triton compiles to GPU code and needs a CUDA GPU to execute** — there is no CPU
simulator (unlike Numba). This machine has no GPU. So each file:

1. **defines the real Triton kernel** — exactly what you would ship to a GPU
   (read these to learn the kernels), and
2. **guards execution**: runs the Triton kernel only `if torch.cuda.is_available()`,
   otherwise falls back to a **pure-PyTorch reference that does run on CPU** and
   validates the *math*, so the file exits 0 everywhere.

For `04.fused_attention.py` the CPU fallback is special: it implements the **same
online-softmax recurrence** the Triton kernel uses (running max/sum/output with
the `alpha` rescaling) in plain PyTorch, and checks it against dense attention —
so you can *see the FlashAttention algorithm is exact* even without a GPU.

## Files (run each with `python NN.name.py`; all exit 0 and print PASS)

| File | Kernel | What it teaches |
|---|---|---|
| `01.vector_add.py` | `C = A + B` | the Triton skeleton: program id, offsets, mask, load/store |
| `02.fused_softmax.py` | row softmax, one program/row | fusion (one DRAM round-trip), `tl.max`/`tl.sum` block reductions, stable softmax |
| `03.matmul.py` | blocked GEMM with `tl.dot` | output tiling, fp32 accumulation, **grouped program ordering** for L2 reuse |
| `04.fused_attention.py` | FlashAttention-style attention | **online softmax**, streaming K/V tiles, O(N) memory, `tl.dot` twice |

## Running on a real GPU

Delete nothing — the files already detect a GPU. On a CUDA machine,
`torch.cuda.is_available()` is `True`, the Triton path runs, and you'll see
`[Triton kernel on GPU] ... PASS`. To benchmark vs PyTorch, wrap the call in
`triton.testing.do_bench` (see the official tutorials).

## Install / requirements

```
pip install triton torch        # triton wheels are Linux/x86_64 + CUDA-oriented
```

Triton is installed here (3.7.0) and the kernels *compile*/import fine; they just
can't *launch* without a GPU.

## References

- Triton docs & tutorials (vector-add, fused-softmax, matmul, fused-attention):
  https://triton-lang.org/main/getting-started/tutorials/index.html
- Tillet, Kung, Cox, "Triton: an IL and compiler for tiled neural-network
  computations", MAPL 2019.
  https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
- Dao et al., "FlashAttention", NeurIPS 2022. https://arxiv.org/abs/2205.14135
- Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018.
  https://arxiv.org/abs/1805.02867

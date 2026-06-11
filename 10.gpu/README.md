# 10 · GPU / CUDA Engineering

A dense, runnable tour of GPU computing for ML engineers — from silicon to
serving. You'll learn **how a GPU is built and why it's fast**, **how to program
it** (CUDA model, then real kernels in Numba and Triton), **how to make PyTorch
fly**, the **optimization patterns** that recur everywhere, and how training and
inference **scale across many GPUs**.

> **No GPU? Everything still runs.** Every `.py` here exits 0 on a CPU-only
> machine. Numba CUDA kernels run on the **CUDA simulator** (real kernel
> semantics, interpreted on the CPU, checked against NumPy). Triton kernels need
> a GPU to *execute*, so each ships the **real kernel code** plus a guarded run
> and a **pure-CPU reference** that validates the math. See "How the
> CPU-simulated kernels work" below.

---

## Contents

| # | File | What you get |
|---|---|---|
| 01 | [`01.gpu-architecture.md`](01.gpu-architecture.md) | CPU vs GPU, SMs, warps/wavefronts, threads/blocks/grids, the memory hierarchy (registers→shared/L1→L2→global), occupancy, latency hiding |
| 02 | [`02.cuda-programming-model.md`](02.cuda-programming-model.md) | kernels, launch config, indexing, memory model & scopes, synchronization, the host/device workflow |
| 03 | [`03.numba-cuda/`](03.numba-cuda/) | **runnable** CUDA kernels in Python (CUDASIM): vector add, naive & tiled matmul (shared memory), parallel reduction, softmax — each checked vs NumPy |
| 04 | [`04.triton/`](04.triton/) | **Triton kernels**: vector add, fused softmax, blocked matmul, FlashAttention-style fused attention — real kernel code + CPU reference fallback |
| 05 | [`05.pytorch-performance.md`](05.pytorch-performance.md) + [`05.profile_demo.py`](05.profile_demo.py) | profiling (`torch.profiler`), AMP/autocast, fusion, `torch.compile`, CUDA graphs, memory format, pinned memory / async copies — with a runnable CPU demo |
| 06 | [`06.optimization-patterns.md`](06.optimization-patterns.md) | coalescing, bank conflicts, tiling/blocking, warp primitives, reductions, the roofline model |
| 07 | [`07.multi-gpu.md`](07.multi-gpu.md) | data / tensor / pipeline (3-D) parallelism, collectives (all-reduce/all-gather), NCCL, DDP / FSDP / DeepSpeed / Megatron |
| 08 | [`08.inference-optimization.md`](08.inference-optimization.md) | KV cache, paged attention, continuous batching, quantization (int8/int4/GGUF/fp8), speculative decoding, FlashAttention |

---

## Learning path

1. **Build the mental model** — read `01` (architecture) then `02` (programming
   model). Don't skip these; every later optimization is a consequence of them.
2. **Write real kernels** — work through `03.numba-cuda/` in order
   (`01`→`05`). These *run on this machine* via the CUDA simulator and check
   against NumPy, so you get immediate, correct feedback. Vector add → naive
   matmul → tiled matmul (shared memory) → reduction → softmax mirrors the
   classic CUDA learning curve.
3. **Meet the modern toolchain** — `04.triton/`. Same kernels, expressed at
   *block* granularity, the way `torch.compile` generates them. The fused
   attention file derives and validates the FlashAttention online-softmax
   algorithm on CPU.
4. **Make PyTorch fast** — `05` + `05.profile_demo.py`. This is what most ML
   engineers actually do day-to-day.
5. **Internalize the patterns** — `06`. The vocabulary (coalescing, tiling,
   roofline) you'll use to reason about any kernel.
6. **Scale out** — `07` (training across GPUs) and `08` (serving LLMs fast).

Suggested prerequisites from this repo: a working knowledge of transformers
(`05.transformers/`) helps for the attention and inference sections; the serving
engines referenced in `08` are catalogued under `07.frameworks/`.

---

## How the CPU-simulated kernels work

This repo is authored on a machine with **no NVIDIA GPU**, yet every kernel file
runs and passes. Two distinct strategies:

### Numba CUDA → the CUDA **simulator** (kernels actually execute)
Setting, at the top of each file *before* importing `numba.cuda`:

```python
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"   # interpret @cuda.jit on the CPU
```

makes Numba *interpret* `@cuda.jit` kernels on the CPU with the **exact same
semantics** — `threadIdx`/`blockIdx`, shared memory, `syncthreads`, the works.
The numbers come out identical to a NumPy reference (each file asserts this);
only the speed and parallelism are absent. Delete that one line on a real GPU and
the same code runs on hardware. (Caveats — `cuda.grid(2)` simulator bug, modest
sizes — are documented in `03.numba-cuda/README.md`.)

### Triton → real kernel code + **CPU reference fallback**
Triton compiles to GPU code and has **no CPU simulator**, so its kernels can't
*execute* here. Each Triton file therefore (a) **defines the genuine kernel** you
would ship, (b) runs it only `if torch.cuda.is_available()`, and (c) otherwise
falls back to a **pure-PyTorch reference that runs on CPU** and validates the
math — so the file exits 0 everywhere and you still learn the kernel. The fused
attention fallback even implements the *same online-softmax recurrence* in torch
and checks it against dense attention.

### What ran as what (on this CPU machine)

| File | Mode on CPU |
|---|---|
| `03.numba-cuda/01.vector_add.py` | **real kernel** (CUDASIM) |
| `03.numba-cuda/02.matmul_naive.py` | **real kernel** (CUDASIM) |
| `03.numba-cuda/03.matmul_tiled.py` | **real kernel** (CUDASIM) |
| `03.numba-cuda/04.reduction.py` | **real kernel** (CUDASIM) |
| `03.numba-cuda/05.softmax.py` | **real kernel** (CUDASIM) |
| `04.triton/01.vector_add.py` | CPU reference (kernel needs a GPU) |
| `04.triton/02.fused_softmax.py` | CPU reference (kernel needs a GPU) |
| `04.triton/03.matmul.py` | CPU reference (kernel needs a GPU) |
| `04.triton/04.fused_attention.py` | CPU online-softmax reference (kernel needs a GPU) |
| `05.profile_demo.py` | runs on CPU (profiler/AMP/compile/format APIs) |

On a CUDA GPU, the Triton files automatically switch to the real kernels and
print `[Triton kernel on GPU] ... PASS`.

---

## Setup

```bash
pip install numba numpy torch triton    # triton optional unless you have a GPU
python 03.numba-cuda/01.vector_add.py   # → PASS
python 04.triton/02.fused_softmax.py    # → PASS (CPU reference here)
python 05.profile_demo.py               # → PASS
```

The simulator needs **no CUDA toolkit**. For real-GPU execution you need a
CUDA-capable NVIDIA GPU and a matching toolkit/driver.

---

## Key references (full lists in each file)

- NVIDIA **CUDA C++ Programming Guide** & **Best Practices Guide** —
  https://docs.nvidia.com/cuda/
- **Numba CUDA** — https://numba.readthedocs.io/en/stable/cuda/index.html
- **Triton** — https://triton-lang.org/
- Kirk & Hwu, *Programming Massively Parallel Processors*, 4th ed.
- Harris, "Optimizing Parallel Reduction in CUDA"
- Williams et al., "Roofline", CACM 2009
- Dao et al., "FlashAttention" (2022/2023); Kwon et al., "PagedAttention"/vLLM (2023)
- Rajbhandari et al., "ZeRO" (2019); Shoeybi et al., "Megatron-LM" (2019)

# 03 · Numba CUDA — write real CUDA kernels in Python, run them on this CPU

[Numba](https://numba.readthedocs.io/en/stable/cuda/index.html) is a JIT
compiler that turns a subset of Python into machine code. Its `numba.cuda`
submodule compiles `@cuda.jit` functions into **PTX** — the same intermediate
representation a CUDA C++ kernel becomes — and runs them on an NVIDIA GPU. You
write genuine CUDA (threads, blocks, shared memory, `syncthreads`, atomics) but
in Python, which makes it the ideal teaching vehicle.

## Why these files run with no GPU: the CUDA simulator

This machine has no NVIDIA GPU. Numba ships a **CUDA simulator** that
*interprets* `@cuda.jit` kernels on the CPU with the exact same indexing and
memory semantics — same `threadIdx`/`blockIdx`, same shared memory, same
barriers — only without the parallelism or the speed. Enable it by setting, **at
the very top of the file, before importing `numba.cuda`**:

```python
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
import numpy as np
from numba import cuda            # now @cuda.jit runs on the CPU
```

Every kernel in this folder runs under the simulator and **checks its result
against a NumPy reference**, so the numbers are provably correct; only the
performance is unrepresentative. On a real GPU you would delete the one
`os.environ` line and the identical code would run on hardware.

## Files (run each with `python NN.name.py`; all exit 0 and print PASS)

| File | Kernel | Concepts introduced |
|---|---|---|
| `01.vector_add.py` | `C = A + B` | launch config, 1-D global index `cuda.grid(1)`, boundary guard, H2D/D2H |
| `02.matmul_naive.py` | `C = A @ B`, one thread per output | 2-D indexing, register accumulation, why it's memory-bound |
| `03.matmul_tiled.py` | tiled matmul with **shared memory** | `cuda.shared.array`, `syncthreads`, data reuse / tiling |
| `04.reduction.py` | parallel sum (tree reduction) | shared-memory reduction, halving-stride, partials |
| `05.softmax.py` | numerically-stable row softmax | two cooperative reductions (max, sum) + map, grid-stride-in-block |

Recommended reading order is the numeric order — each builds on the last. Read
them alongside `../02.cuda-programming-model.md`.

## CUDASIM gotchas (important, so the demos stay honest)

The simulator is faithful but not identical to hardware. Two quirks shaped the
code here:

1. **`cuda.grid(2)` is buggy in the simulator** (numba 0.65.x): in boundary
   blocks it can return wrong indices, corrupting results. We therefore write the
   index out explicitly in `02.matmul_naive.py`:
   ```python
   row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
   col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
   ```
   This is *exactly* what `cuda.grid(2)` expands to, and it is correct under both
   the simulator and real hardware. `cuda.grid(1)` (1-D) is fine.

2. **The simulator is slow** — every thread is interpreted in pure Python. A
   real launch of millions of threads would take minutes. So the demos use
   **modest sizes** (a few thousand elements, ≤100×100 matrices). On a real GPU
   you would scale these up by several orders of magnitude. Sizes are still
   chosen *not* to be multiples of the block size, so the boundary-guard logic is
   genuinely exercised.

3. Output arrays are explicitly **zero-initialized** (`cuda.to_device(np.zeros)`)
   rather than `cuda.device_array` where the simulator's uninitialized memory
   could otherwise surface as `NaN`.

## Install / requirements

```
pip install numba numpy        # numba pulls in llvmlite; numpy is the reference
```

No `cudatoolkit` is needed for the simulator. For real-GPU execution you'd need a
CUDA-capable GPU and matching toolkit.

## References

- Numba CUDA docs: https://numba.readthedocs.io/en/stable/cuda/index.html
- Numba CUDA examples (matmul, reduction): https://numba.readthedocs.io/en/stable/cuda/examples.html
- CUDASIM: https://numba.readthedocs.io/en/stable/cuda/simulator.html
- M. Harris, "Optimizing Parallel Reduction in CUDA",
  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

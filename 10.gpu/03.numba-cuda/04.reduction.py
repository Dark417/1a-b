"""04 · Parallel reduction — summing an array on the GPU.

A reduction collapses N values into one with an associative operator (here +).
On a CPU you loop. On a GPU you build a **tree**: pairs combine in parallel, then
pairs-of-pairs, and so on — log2(N) levels instead of N steps.

This file shows the textbook *shared-memory block reduction*:

  Phase 1 (per block, in shared memory):
    - each thread loads one element into shared array `sdata`;
    - the block reduces `sdata` in-place with a halving stride loop
      (`s = blockDim/2, /4, ...`), syncing between levels;
    - thread 0 writes the block's partial sum to `partials[blockIdx]`.

  Phase 2: there are now `num_blocks` partial sums (a much smaller array). For a
  one-pass demo we finish the partials on the host; on a real GPU you'd either
  launch the same kernel again on the partials, or use `cuda.atomic.add` to
  accumulate directly into a single global slot.

Key pitfalls this teaches
--------------------------
  - **Sync between every tree level** — threads run in lockstep only within a
    warp; across a block you need `cuda.syncthreads()` or you read stale data.
  - The classic *halving stride* (`s //= 2`) keeps active threads contiguous,
    which (on real hardware) avoids shared-memory bank conflicts and warp
    divergence — far better than a doubling stride with modulo.
  - **Padding**: threads past N load the identity element (0 for sum).

References
----------
- Mark Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA), the definitive
  treatment of the 7 successive optimizations.
  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
- CUDA C++ Programming Guide §B.5 "Synchronization Functions".
"""

import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import numpy as np
from numba import cuda, float32

THREADS = 128


@cuda.jit
def block_reduce_sum_kernel(x, partials, n):
    sdata = cuda.shared.array(shape=(THREADS,), dtype=float32)

    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + tid

    # Load one element (or the identity 0.0 for out-of-range threads).
    sdata[tid] = x[i] if i < n else float32(0.0)
    cuda.syncthreads()

    # Tree reduction with halving stride.
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            sdata[tid] += sdata[tid + s]
        cuda.syncthreads()
        s //= 2

    # Thread 0 emits this block's partial sum.
    if tid == 0:
        partials[cuda.blockIdx.x] = sdata[0]


def reduce_sum(x):
    n = x.shape[0]
    blocks = (n + THREADS - 1) // THREADS
    d_x = cuda.to_device(x)
    d_partials = cuda.device_array(blocks, dtype=np.float32)
    block_reduce_sum_kernel[blocks, THREADS](d_x, d_partials, n)
    # Phase 2: finish the (small) partials array. On real HW: relaunch or atomics.
    return float(d_partials.copy_to_host().sum())


def main():
    rng = np.random.default_rng(3)
    # Kept modest because every thread is *interpreted* in Python under the CUDA
    # simulator; on a real GPU you would push millions of elements per launch.
    n = 4096
    x = rng.standard_normal(n).astype(np.float32)

    got = reduce_sum(x)
    ref = float(x.sum())

    ok = abs(got - ref) < 1e-2 * max(1.0, abs(ref))
    print(f"reduction: n={n}  gpu_sum={got:.4f}  ref={ref:.4f}")
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()

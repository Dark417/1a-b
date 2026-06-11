"""01 · Vector addition — the "hello world" of CUDA, in Numba.

    C[i] = A[i] + B[i]      for i in 0 .. N-1

This is the canonical *embarrassingly parallel* problem: every output element is
independent, so we assign **one GPU thread per element**. It introduces the four
ideas you reuse in every kernel:

  1. The launch configuration: a 1-D **grid** of **blocks**, each block a fixed
     number of **threads** (`threads_per_block`). Total threads = blocks × tpb.
  2. Computing a unique global index from the block/thread coordinates.
  3. The **boundary guard** `if i < n:` — N is rarely a multiple of the block
     size, so the last block has idle threads that must not write out of bounds.
  4. Host↔device data movement (here implicit via `cuda.to_device` / `copy`).

Numba compiles the Python `@cuda.jit` function to PTX for a real GPU. On this
machine there is no GPU, so we set `NUMBA_ENABLE_CUDASIM=1`, which makes Numba
*interpret* the kernel on the CPU with the exact same indexing semantics. The
numbers come out identical to a NumPy reference — only the speed differs.

References
----------
- NVIDIA CUDA C++ Programming Guide, ch. 2 "Programming Model".
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Numba CUDA docs: https://numba.readthedocs.io/en/stable/cuda/index.html
- "An Even Easier Introduction to CUDA", Mark Harris, NVIDIA Developer Blog.
  https://developer.nvidia.com/blog/even-easier-introduction-cuda/
"""

import os

# MUST be set before importing numba.cuda: run the kernel on the CPU simulator.
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import numpy as np
from numba import cuda


@cuda.jit
def vector_add_kernel(a, b, c, n):
    """One thread computes one element of c."""
    # cuda.grid(1) == cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.grid(1)
    if i < n:                 # boundary guard: skip threads past the end
        c[i] = a[i] + b[i]


def vector_add(a, b, threads_per_block=128):
    """Host wrapper: move data to device, launch, copy result back."""
    n = a.shape[0]
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array_like(a)

    # ceil-div: enough blocks so blocks*tpb >= n (so every element is covered).
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # The [blocks, threads] syntax is Numba's launch configuration.
    vector_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c, n)
    return d_c.copy_to_host()


def main():
    rng = np.random.default_rng(0)
    n = 1_000_003                      # deliberately NOT a multiple of 128
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)

    got = vector_add(a, b)
    ref = a + b                        # NumPy reference

    ok = np.allclose(got, ref, atol=1e-5)
    print(f"vector_add: n={n}, max|err|={np.max(np.abs(got - ref)):.2e}")
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()

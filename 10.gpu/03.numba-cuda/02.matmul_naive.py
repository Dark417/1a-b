"""02 · Naive matrix multiply — one thread per output element.

    C[i, j] = sum_k A[i, k] * B[k, j]        A: (M,K)  B: (K,N)  C: (M,N)

We launch a **2-D grid of 2-D blocks**: thread (row, col) computes exactly one
C[row, col] by streaming the full row of A and column of B from *global memory*.

Why this is the "naive" version
-------------------------------
Each thread reads K elements of A and K elements of B straight from global
memory (the slow, off-chip DRAM, hundreds of cycles of latency). Adjacent
threads re-read overlapping data, so the same bytes cross the memory bus many
times. The arithmetic intensity is low: ~2 FLOPs per 2 loads. The next file
(`03.matmul_tiled.py`) fixes this with shared memory.

This still teaches three core skills:
  - 2-D indexing: `row, col = cuda.grid(2)`.
  - Mapping a 2-D problem onto a 2-D launch config.
  - Accumulating into a register (`acc`) before a single global write — writing
    to a local variable in a loop, then storing once, avoids K global writes.

References
----------
- CUDA C++ Programming Guide §3.2.4 "Shared Memory" (shows this matmul first).
- Numba CUDA examples: https://numba.readthedocs.io/en/stable/cuda/examples.html
"""

import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import numpy as np
from numba import cuda


@cuda.jit
def matmul_naive_kernel(a, b, c):
    row, col = cuda.grid(2)            # 2-D global thread index
    m, k = a.shape
    _, n = b.shape
    if row < m and col < n:
        acc = 0.0                      # accumulate in a register, not in C
        for kk in range(k):
            acc += a[row, kk] * b[kk, col]
        c[row, col] = acc              # one global write per thread


def matmul(a, b, block=(16, 16)):
    m, k = a.shape
    _, n = b.shape
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array((m, n), dtype=np.float32)

    # grid is sized in (x=cols, y=rows) so it tiles the whole C matrix.
    grid = ((n + block[0] - 1) // block[0],
            (m + block[1] - 1) // block[1])
    matmul_naive_kernel[grid, block](d_a, d_b, d_c)
    return d_c.copy_to_host()


def main():
    rng = np.random.default_rng(1)
    m, k, n = 64, 48, 80               # not multiples of 16 → exercises guard
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    got = matmul(a, b)
    ref = a @ b

    ok = np.allclose(got, ref, atol=1e-3)
    print(f"matmul_naive: ({m}x{k})@({k}x{n}), max|err|={np.max(np.abs(got-ref)):.2e}")
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()

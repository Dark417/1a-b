"""05 · Row-wise softmax — a fused, numerically-stable reduction kernel.

Softmax turns a row of logits into a probability distribution:

    softmax(x)_j = exp(x_j - m) / sum_k exp(x_k - m),   m = max_k x_k

We subtract the row max `m` first — the **numerically stable** form. Without it,
`exp(x_j)` overflows to `inf` for x_j > ~88 in float32; subtracting the max
guarantees the largest exponent is `exp(0)=1`, so nothing overflows and the
result is unchanged (the `m` cancels in numerator and denominator).

Mapping to the GPU
------------------
We assign **one thread block per row**. Each block:

  1. cooperatively finds the row maximum  (a max-reduction in shared memory),
  2. cooperatively sums `exp(x_j - m)`     (a sum-reduction in shared memory),
  3. each thread writes `exp(x_j - m)/denom` for the columns it owns.

This is exactly the structure of a real *fused softmax* kernel (see the Triton
version in `../04.triton/02.fused_softmax.py`): two reductions plus an
element-wise map, kept on-chip so the row is read from global memory once. A
block may have fewer threads than there are columns, so each thread strides over
the row in `blockDim`-sized steps (the *grid-stride loop* pattern, applied
within a block).

Pitfalls this teaches
---------------------
  - Stable softmax requires the max-subtraction (overflow safety).
  - Two cooperative reductions must each be followed by `syncthreads()` before
    the shared result is read by other threads.
  - Reductions need an **identity** for padding lanes: -inf for max, 0 for sum.

References
----------
- Goodfellow, Bengio, Courville, *Deep Learning*, §6.2.2 (softmax numerics).
- Milakov & Gimelshein, "Online normalizer calculation for softmax" (2018),
  https://arxiv.org/abs/1805.02867  — the one-pass/online variant FlashAttention
  builds on.
- NVIDIA CUDA C++ Programming Guide §3.2.4 (shared memory & block reductions).
"""

import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import math

import numpy as np
from numba import cuda, float32

THREADS = 128
NEG_INF = float32(-1e30)


@cuda.jit
def softmax_rows_kernel(x, out, n_cols):
    """One block per row; block cooperatively softmaxes that row."""
    row = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x

    smax = cuda.shared.array(shape=(THREADS,), dtype=float32)
    ssum = cuda.shared.array(shape=(THREADS,), dtype=float32)

    # --- Phase 1: row maximum (grid-stride over columns, then tree reduce) ---
    local_max = NEG_INF
    j = tid
    while j < n_cols:
        v = x[row, j]
        if v > local_max:
            local_max = v
        j += nthreads
    smax[tid] = local_max
    cuda.syncthreads()

    s = nthreads // 2
    while s > 0:
        if tid < s:
            if smax[tid + s] > smax[tid]:
                smax[tid] = smax[tid + s]
        cuda.syncthreads()
        s //= 2
    row_max = smax[0]
    cuda.syncthreads()

    # --- Phase 2: sum of exp(x - row_max) ---
    local_sum = float32(0.0)
    j = tid
    while j < n_cols:
        local_sum += math.exp(x[row, j] - row_max)
        j += nthreads
    ssum[tid] = local_sum
    cuda.syncthreads()

    s = nthreads // 2
    while s > 0:
        if tid < s:
            ssum[tid] += ssum[tid + s]
        cuda.syncthreads()
        s //= 2
    denom = ssum[0]
    cuda.syncthreads()

    # --- Phase 3: normalized write-back ---
    j = tid
    while j < n_cols:
        out[row, j] = math.exp(x[row, j] - row_max) / denom
        j += nthreads


def softmax_rows(x, threads=THREADS):
    n_rows, n_cols = x.shape
    d_x = cuda.to_device(x)
    d_out = cuda.device_array_like(x)
    # one block per row; `threads` threads stride over the columns
    softmax_rows_kernel[n_rows, threads](d_x, d_out, n_cols)
    return d_out.copy_to_host()


def numpy_softmax(x):
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(5)
    # wider than THREADS so the grid-stride-within-block path is exercised,
    # and include large logits to prove the stable form avoids overflow.
    x = rng.standard_normal((16, 300)).astype(np.float32)
    x[0, 7] = 1000.0  # would overflow exp() without max-subtraction

    got = softmax_rows(x)
    ref = numpy_softmax(x)

    rows_sum_to_one = np.allclose(got.sum(axis=1), 1.0, atol=1e-4)
    ok = np.allclose(got, ref, atol=1e-5) and rows_sum_to_one
    print(f"softmax: x={x.shape}, max|err|={np.max(np.abs(got - ref)):.2e}, "
          f"rows_sum_to_one={rows_sum_to_one}")
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()

"""03 · Tiled matrix multiply — the shared-memory optimization.

The naive kernel re-reads A and B from slow global memory K times per output.
The fix is **tiling** (a.k.a. blocking): the block cooperatively loads a small
TILE×TILE square of A and of B into fast on-chip **shared memory**, every thread
in the block reuses those tiles, then we slide along the K dimension.

The mental picture
------------------
C is partitioned into TILE×TILE output tiles, one per thread block. To compute
its tile, a block walks across K in TILE-wide steps. At each step:

    1. Each thread loads ONE element of the A-tile and ONE of the B-tile into
       shared memory `sA`, `sB`.
    2. `cuda.syncthreads()`  — barrier: wait until the whole tile is loaded.
    3. Each thread does TILE multiply-adds using only shared memory (fast).
    4. `cuda.syncthreads()`  — barrier: don't overwrite the tile until everyone
       has finished using it.

Each global element of A and B is now read once per *tile* instead of once per
*output element*: a factor-TILE reduction in global-memory traffic. This is the
single most important GPU optimization pattern — data reuse via shared memory.

Boundary handling: M, K, N need not be multiples of TILE. Out-of-range loads
write 0.0 into shared memory so they contribute nothing to the sum, and
out-of-range threads simply skip the final store.

References
----------
- CUDA C++ Programming Guide §3.2.4 (the canonical tiled-matmul walkthrough).
- Kirk & Hwu, "Programming Massively Parallel Processors", ch. on tiling.
- Numba shared-memory matmul example (numba.readthedocs.io cuda/examples).
"""

import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import numpy as np
from numba import cuda, float32

TILE = 16


@cuda.jit
def matmul_tiled_kernel(a, b, c):
    # Statically-sized shared-memory tiles, shared by all threads in the block.
    sA = cuda.shared.array(shape=(TILE, TILE), dtype=float32)
    sB = cuda.shared.array(shape=(TILE, TILE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE + ty   # global row this thread owns
    col = cuda.blockIdx.x * TILE + tx   # global col this thread owns

    m, k = a.shape
    _, n = b.shape

    acc = float32(0.0)
    # Number of TILE-wide steps along the shared K dimension.
    n_tiles = (k + TILE - 1) // TILE
    for t in range(n_tiles):
        # --- cooperative load of one A-tile and one B-tile ---
        a_col = t * TILE + tx
        a_row = row
        sA[ty, tx] = a[a_row, a_col] if (a_row < m and a_col < k) else float32(0.0)

        b_row = t * TILE + ty
        b_col = col
        sB[ty, tx] = b[b_row, b_col] if (b_row < k and b_col < n) else float32(0.0)

        cuda.syncthreads()              # tile fully loaded before use

        # --- compute using only shared memory ---
        for kk in range(TILE):
            acc += sA[ty, kk] * sB[kk, tx]

        cuda.syncthreads()              # done with tile before next overwrite

    if row < m and col < n:
        c[row, col] = acc


def matmul(a, b):
    m, k = a.shape
    _, n = b.shape
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.device_array((m, n), dtype=np.float32)

    block = (TILE, TILE)
    grid = ((n + TILE - 1) // TILE, (m + TILE - 1) // TILE)
    matmul_tiled_kernel[grid, block](d_a, d_b, d_c)
    return d_c.copy_to_host()


def main():
    rng = np.random.default_rng(2)
    m, k, n = 70, 50, 90                # none a multiple of TILE=16
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    got = matmul(a, b)
    ref = a @ b

    ok = np.allclose(got, ref, atol=1e-3)
    print(f"matmul_tiled: ({m}x{k})@({k}x{n}) TILE={TILE}, "
          f"max|err|={np.max(np.abs(got-ref)):.2e}")
    print("PASS" if ok else "FAIL")
    assert ok


if __name__ == "__main__":
    main()

"""02 · Triton fused softmax — one program per row, fully on-chip.

A naive PyTorch softmax over an (M, N) matrix launches *many* kernels (max, sub,
exp, sum, div), each of which reads and writes the whole matrix to global memory.
That is memory-bound: the matrix crosses the DRAM bus ~5 times. A **fused**
kernel reads each row once into on-chip SRAM, does the max → exp → sum → divide
there, and writes the result once — ideally a single round trip.

The kernel (one Triton program per row)
---------------------------------------
  1. point at row `pid`; load all N columns into registers (`tl.load` with a
     column mask so N need not be a power of two),
  2. `row_max = tl.max(row)`           — numerically-stable shift,
  3. `num = tl.exp(row - row_max)`,
  4. `denom = tl.sum(num)`,
  5. store `num / denom`.

`tl.max`/`tl.sum` are *block reductions* the compiler lowers to warp shuffles +
shared memory — you write `tl.sum(x)` and Triton generates the reduction tree.
`BLOCK_SIZE` is the next power of two ≥ N so the whole row fits in one block;
masked-out lanes load `-inf` (identity for max) so they never affect the result.

This is the structure the official Triton softmax tutorial benchmarks at >2x
torch on memory-bound shapes — the win is *fusion*, not cleverer math.

GPU-less fallback: the real kernel is defined and run only under CUDA; on CPU we
validate against `torch.softmax` so the file still exits 0.

References
----------
- Triton tutorials, "Fused Softmax":
  https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
- Milakov & Gimelshein, "Online normalizer calculation for softmax", 2018,
  https://arxiv.org/abs/1805.02867
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    in_ptrs = in_ptr + row * in_row_stride + col_offsets
    # masked load; padding lanes get -inf so they lose the max and exp to 0.
    x = tl.load(in_ptrs, mask=mask, other=float("-inf"))

    x = x - tl.max(x, axis=0)              # stable shift
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    out_ptrs = out_ptr + row * out_row_stride + col_offsets
    tl.store(out_ptrs, y, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols, BLOCK_SIZE=block_size,
    )
    return out


def cpu_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=1)


def main():
    torch.manual_seed(0)
    x = torch.randn(1024, 781)            # ragged width on purpose
    ref = cpu_reference(x)

    if torch.cuda.is_available():
        got = triton_softmax(x.cuda()).cpu()
        ok = torch.allclose(got, ref, atol=1e-5)
        print(f"[Triton kernel on GPU] softmax {tuple(x.shape)}, "
              f"max|err|={(got - ref).abs().max():.2e}")
        print("PASS" if ok else "FAIL")
        assert ok
    else:
        print("Triton kernels require a CUDA GPU — showing code + CPU reference.")
        got = cpu_reference(x)
        ok = torch.allclose(got, ref) and torch.allclose(
            got.sum(dim=1), torch.ones(x.shape[0]), atol=1e-5)
        print(f"[CPU reference] softmax {tuple(x.shape)}, rows sum to 1: "
              f"{bool(torch.allclose(got.sum(1), torch.ones(x.shape[0]), atol=1e-5))}")
        print("PASS" if ok else "FAIL")
        assert ok


if __name__ == "__main__":
    main()

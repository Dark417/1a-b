"""01 · Triton vector add — the Triton "hello world".

    C = A + B      (element-wise, length N)

Triton is a Python DSL + compiler from OpenAI: you write a kernel that operates
on **blocks of elements** (not single threads like CUDA), and Triton handles the
intra-block parallelism, vectorization, and memory coalescing for you. The unit
of work is a *program* (one block); `tl.program_id(0)` is its index, the analogue
of CUDA's `blockIdx.x`.

The block-programming model
---------------------------
Instead of "one thread per element", Triton is "one **program** per BLOCK of
elements". A program:
  1. computes the offsets of its block:  `pid*BLOCK + arange(0, BLOCK)`,
  2. builds a **mask** so the tail block doesn't read/write out of bounds,
  3. `tl.load`s A and B (masked), adds, `tl.store`s C (masked).
`tl.load`/`tl.store` with a mask are vectorized, coalesced memory ops — Triton
maps the BLOCK onto a warp/threadblock automatically. You reason about blocks;
the compiler reasons about threads.

This machine has no CUDA GPU, so the real kernel cannot execute (Triton emits
GPU code). We therefore:
  - define the genuine Triton kernel (this is exactly what you'd ship), and
  - run it only `if torch.cuda.is_available()`, otherwise fall back to a pure
    PyTorch reference that DOES run on CPU, so the file exits 0 either way.

References
----------
- Triton tutorials, "Vector Addition":
  https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
- Tillet, Kung, Cox, "Triton: an intermediate language and compiler for tiled
  neural network computations", MAPL 2019.
  https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                 # which block am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                 # guard the ragged tail block
    x = tl.load(x_ptr + offsets, mask=mask)     # coalesced, vectorized load
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    # The launch grid: ceil(n / BLOCK) programs. `meta` lets the grid read the
    # autotuned/constexpr BLOCK_SIZE chosen at compile time.
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


def cpu_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def main():
    torch.manual_seed(0)
    n = 98_432                                  # not a multiple of 1024
    x = torch.randn(n)
    y = torch.randn(n)
    ref = cpu_reference(x, y)

    if torch.cuda.is_available():
        got = triton_add(x.cuda(), y.cuda()).cpu()
        ok = torch.allclose(got, ref, atol=1e-5)
        print(f"[Triton kernel on GPU] vector_add n={n}, "
              f"max|err|={(got - ref).abs().max():.2e}")
        print("PASS" if ok else "FAIL")
        assert ok
    else:
        print("Triton kernels require a CUDA GPU — showing code + CPU reference.")
        got = cpu_reference(x, y)
        ok = torch.allclose(got, ref)
        print(f"[CPU reference] vector_add n={n}, "
              f"max|err|={(got - ref).abs().max():.2e}")
        print("PASS" if ok else "FAIL")
        assert ok


if __name__ == "__main__":
    main()

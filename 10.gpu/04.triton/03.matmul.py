"""03 · Triton matmul — blocked, accumulator-in-registers GEMM.

    C = A @ B     A: (M, K)  B: (K, N)  C: (M, N)

This is the workhorse of deep learning, and the kernel that best shows Triton's
strengths: you express the **block** structure (a BLOCK_M × BLOCK_N output tile,
swept along K in BLOCK_K steps), and Triton handles register allocation, shared
memory, double-buffering, and (with `tl.dot`) maps onto the Tensor Cores.

The blocked algorithm (one program per output tile)
---------------------------------------------------
Program `(pid_m, pid_n)` owns the C tile rows [pid_m*BM : +BM], cols
[pid_n*BN : +BN]. It keeps a BM×BN accumulator in registers and loops over K:

    acc = zeros(BM, BN)
    for k in range(0, K, BK):
        a = load A[rows, k:k+BK]          # BM x BK tile
        b = load B[k:k+BK, cols]          # BK x BN tile
        acc += tl.dot(a, b)               # BM x BN matrix-multiply-accumulate
    store C[rows, cols] = acc

`tl.dot` is the key primitive: a block-level matmul the compiler lowers to MMA /
Tensor-Core instructions. Masks on every load/store make M, N, K arbitrary.

Two production refinements appear here:
  - **Grouped / super-blocked program ordering** (`GROUP_M`): re-numbering
    programs so that nearby tiles run together improves **L2 cache** reuse of A
    and B (the same A-rows / B-cols are revisited by neighbouring tiles). This is
    the single biggest non-Tensor-Core win in the official tutorial.
  - Accumulate in fp32 even for fp16 inputs (`acc` is float32) for accuracy.

GPU-less fallback: kernel defined and run only under CUDA; on CPU we check the
shapes/logic against `torch.mm` so the file exits 0.

References
----------
- Triton tutorials, "Matrix Multiplication":
  https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- CUTLASS docs on tiled GEMM & the GEMM hierarchy (block/warp/thread):
  https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- grouped program ordering for L2 reuse ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- offsets for this tile ----
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # ---- main K loop: accumulate in fp32 registers ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ---- masked store ----
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) *
                         triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, GROUP_M=8,
    )
    return c


def cpu_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def main():
    torch.manual_seed(0)
    M, K, N = 321, 137, 289               # all ragged vs the block sizes
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    ref = cpu_reference(a, b)

    if torch.cuda.is_available():
        got = triton_matmul(a.cuda(), b.cuda()).cpu()
        ok = torch.allclose(got, ref, atol=1e-2, rtol=1e-2)
        print(f"[Triton kernel on GPU] matmul ({M}x{K})@({K}x{N}), "
              f"max|err|={(got - ref).abs().max():.2e}")
        print("PASS" if ok else "FAIL")
        assert ok
    else:
        print("Triton kernels require a CUDA GPU — showing code + CPU reference.")
        got = cpu_reference(a, b)
        ok = torch.allclose(got, ref)
        print(f"[CPU reference] matmul ({M}x{K})@({K}x{N}), "
              f"max|err|={(got - ref).abs().max():.2e}")
        print("PASS" if ok else "FAIL")
        assert ok


if __name__ == "__main__":
    main()

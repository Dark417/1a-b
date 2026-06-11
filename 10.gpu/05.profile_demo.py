"""05 · PyTorch performance — runnable CPU demos of the profiling/perf toolbox.

Companion code for `05.pytorch-performance.md`. Everything here RUNS ON CPU (no
GPU needed) and exits 0, demonstrating the *API* and *concepts* even though the
dramatic speedups only appear on a GPU. Each section prints what it measured.

Sections
--------
  1. torch.profiler — record ops, sort by self-CPU time, print a table.
  2. torch.autocast (AMP) — the mixed-precision context manager API on CPU.
  3. operator fusion via torch.compile — compile a pointwise chain; verify it
     still produces the same numbers (the speedup is from fusing the chain into
     one kernel; on GPU that removes several memory round-trips).
  4. pinned memory + non_blocking copies — the API you use to overlap H2D copies
     with compute (no-op effect on CPU, but the calls are real).
  5. memory_format (channels_last) — how to request NHWC layout for convs.

Run:  python 05.profile_demo.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)        # avoid many-core thread thrashing in demos


# --------------------------------------------------------------------------- #
# 1. torch.profiler — measure where time goes.
# --------------------------------------------------------------------------- #
def demo_profiler():
    from torch.profiler import profile, record_function, ProfilerActivity

    x = torch.randn(512, 512)
    w = torch.randn(512, 512)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("my_matmul_block"):       # custom user label
            for _ in range(10):
                y = (x @ w).relu()
                y = y @ w.t()

    # key_averages() aggregates per-op; we print the top rows by self CPU time.
    table = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=5)
    print("=== 1. torch.profiler (top ops by self CPU time) ===")
    print(table)
    print(f"(result tensor shape {tuple(y.shape)})\n")


# --------------------------------------------------------------------------- #
# 2. AMP / autocast — mixed precision context manager.
# --------------------------------------------------------------------------- #
def demo_autocast():
    print("=== 2. torch.autocast (AMP) ===")
    x = torch.randn(256, 256)
    w = torch.randn(256, 256)
    # On GPU, autocast(dtype=float16/bfloat16) runs eligible ops in low precision
    # (faster Tensor-Core matmuls) while keeping a float32 master copy. On CPU we
    # demonstrate the bfloat16 path, which CPU autocast supports.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = x @ w
    print(f"autocast output dtype: {y.dtype}  (matmul ran in bf16)")
    # GradScaler is the partner API for fp16 training (prevents grad underflow):
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)  # enabled on CUDA
    print(f"GradScaler created (enabled={scaler.is_enabled()} on CPU)\n")


# --------------------------------------------------------------------------- #
# 3. Operator fusion via torch.compile.
# --------------------------------------------------------------------------- #
def demo_compile():
    print("=== 3. torch.compile (operator fusion) ===")

    def chain(x):
        # a pointwise chain: each op is memory-bound; fusing them into one kernel
        # removes intermediate global-memory round-trips.
        return torch.sin(x) * 2.0 + torch.cos(x) ** 2

    x = torch.randn(1000)
    eager = chain(x)
    try:
        compiled = torch.compile(chain)
        fused = compiled(x)
        same = torch.allclose(eager, fused, atol=1e-5)
        print(f"torch.compile produced identical result: {same}")
    except Exception as e:                       # compile backend may be absent
        print(f"torch.compile unavailable in this env ({type(e).__name__}); "
              f"eager result still valid.")
    print()


# --------------------------------------------------------------------------- #
# 4. Pinned memory + non_blocking copies.
# --------------------------------------------------------------------------- #
def demo_pinned():
    print("=== 4. pinned memory + async copies ===")
    cpu_tensor = torch.randn(1024, 1024)
    if torch.cuda.is_available():
        pinned = cpu_tensor.pin_memory()         # page-locked → DMA-able
        gpu = pinned.to("cuda", non_blocking=True)  # overlaps with compute
        print(f"copied pinned tensor to GPU async, shape {tuple(gpu.shape)}")
    else:
        # The pin_memory() API exists on CPU-only builds too (may warn/no-op).
        try:
            pinned = cpu_tensor.pin_memory()
            print(f"pin_memory() ok, is_pinned={pinned.is_pinned()} "
                  f"(no GPU → non_blocking copy would be a no-op)")
        except Exception as e:
            print(f"pin_memory() needs a CUDA build ({type(e).__name__}); "
                  f"the pattern is: x.pin_memory().to('cuda', non_blocking=True)")
    print()


# --------------------------------------------------------------------------- #
# 5. memory_format — channels_last (NHWC) for convolutions.
# --------------------------------------------------------------------------- #
def demo_memory_format():
    print("=== 5. memory_format (channels_last / NHWC) ===")
    x = torch.randn(8, 16, 32, 32)               # NCHW
    x_cl = x.to(memory_format=torch.channels_last)
    conv = nn.Conv2d(16, 16, 3, padding=1).to(memory_format=torch.channels_last)
    y = conv(x_cl)
    print(f"input is_contiguous(channels_last): "
          f"{x_cl.is_contiguous(memory_format=torch.channels_last)}")
    print(f"conv output preserves channels_last: "
          f"{y.is_contiguous(memory_format=torch.channels_last)}")
    print("(NHWC lets cuDNN/Tensor-Cores avoid layout transposes on GPU)\n")


def main():
    demo_profiler()
    demo_autocast()
    demo_compile()
    demo_pinned()
    demo_memory_format()
    print("PASS")


if __name__ == "__main__":
    main()

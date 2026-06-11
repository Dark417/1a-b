"""
Accelerate Mixed Precision & Gradient Accumulation — Runnable Example
======================================================================
Docs: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
      https://huggingface.co/docs/accelerate/usage_guides/mixed_precision
      https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRADIENT ACCUMULATION — MATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Normally: optimizer.step() after every micro-batch of size B.
With gradient accumulation over K steps:
  - Run K forward+backward passes WITHOUT stepping
  - Sum gradients: g_eff = Σ_{k=1}^{K} g_k
  - Step once with accumulated gradients
  - Effective batch size = B * K  (same as physically using B*K batch)

Memory: still allocates activations for ONE micro-batch at a time → fits in
smaller VRAM while matching the gradient statistics of a large batch.

With Accelerate the accumulate() context manager handles the tricky part:
in DDP mode it calls model.no_sync() for the first K-1 steps (avoids
expensive all-reduce on every step), then syncs only on the K-th step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MIXED PRECISION ON CPU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- fp16 (IEEE half): NOT natively accelerated on x86 CPU → likely SLOWER +
  potential underflow without a loss scaler. Accelerate will warn.
- bf16 (bfloat16): supported on modern x86 (AVX512-BF16) and all Apple M1+
  CPUs; same exponent range as float32, just 7 mantissa bits.
  PyTorch ≥2.0 has native bf16 CPU kernels. Still slower than fp32 on CPU
  but at least numerically stable.
- "no" (fp32): the safe default for CPU training.

This script tries bf16 first; if not supported falls back to "no" (fp32)
and explains why. This matches real-world advice: use fp16/bf16 on GPU only.
"""

import sys
import os
import random

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import numpy as np
np.random.seed(0)
random.seed(0)

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

# ── Detect best precision for this CPU ──────────────────────────────────────
banner("0. Detect best mixed precision for this CPU")

def detect_precision():
    """Return the best mixed_precision string for the current CPU."""
    # Test bf16 matmul: if it errors → not supported
    try:
        a = torch.ones(4, 4, dtype=torch.bfloat16)
        _ = a @ a
        return "bf16"
    except Exception:
        return "no"

PRECISION = detect_precision()
print(f"  Chosen mixed_precision='{PRECISION}'")
if PRECISION == "bf16":
    print("  bf16 supported on this CPU — using bfloat16 weights/activations")
else:
    note_skip(
        "bf16 not supported on this CPU — using fp32 (mixed_precision='no')."
        " On a modern GPU or M-series Mac, bf16/fp16 give real speedups."
    )

# ── Dataset ──────────────────────────────────────────────────────────────────
banner("1. Synthetic classification dataset")

import torch.nn as nn

N, D, C = 128, 32, 4
X = torch.randn(N, D)
labels = torch.randint(0, C, (N,))

MICRO_BATCH = 8
GRAD_ACCUM_K = 4       # effective batch = 8 * 4 = 32

print(f"  N={N}, D={D}, C={C}")
print(f"  micro-batch={MICRO_BATCH}, grad_accum_steps={GRAD_ACCUM_K}")
print(f"  effective batch size = {MICRO_BATCH * GRAD_ACCUM_K}")

ds = torch.utils.data.TensorDataset(X, labels)
dl = torch.utils.data.DataLoader(ds, batch_size=MICRO_BATCH, shuffle=True)

# ── Model ─────────────────────────────────────────────────────────────────────
banner("2. Tiny classifier model")

class TinyClassifier(nn.Module):
    def __init__(self, d, hidden, c):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.bn  = nn.BatchNorm1d(hidden)   # tests norm layer handling in AMP
        self.fc2 = nn.Linear(hidden, c)

    def forward(self, x):
        x = torch.relu(self.bn(self.fc1(x)))
        return self.fc2(x)

model = TinyClassifier(D, 64, C)
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")

# ── Accelerator with mixed precision + gradient accumulation ─────────────────
banner("3. Accelerator(mixed_precision, gradient_accumulation_steps)")

from accelerate import Accelerator

accelerator = Accelerator(
    cpu=True,
    mixed_precision=PRECISION,
    gradient_accumulation_steps=GRAD_ACCUM_K,
)
print(f"  mixed_precision  : {accelerator.mixed_precision}")
print(f"  grad_accum_steps : {accelerator.gradient_accumulation_steps}")
print(f"  device           : {accelerator.device}")

# ── prepare() ────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

# ── Training loop ─────────────────────────────────────────────────────────────
banner("4. Training loop — 4 epochs showing loss decreasing")

CLIP_NORM = 1.0

for epoch in range(4):
    model.train()
    total_loss = 0.0
    n_batches = 0
    step_count = 0       # counts actual optimizer steps (not micro-batch)

    for batch_x, batch_y in dl:
        # ─── accumulate() context ─────────────────────────────────────────
        # With gradient_accumulation_steps=K, the context manager:
        #   • Tracks a micro-step counter internally
        #   • For steps 1..K-1: sets model.no_sync() (DDP) → no all-reduce
        #   • For step K: allows sync → gradient all-reduce → optimizer step
        # On CPU/single-GPU: just tracks when to call optimizer.step()
        with accelerator.accumulate(model):
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            accelerator.backward(loss)
            # ^ Replaces loss.backward(). With fp16: applies GradScaler.
            # With fp32/bf16: equivalent to loss.backward() + no-op scaler.

            # clip_grad_norm_ is a no-op inside accumulate() on non-sync steps
            # (Accelerate internally skips clip when accumulating)
            accelerator.clip_grad_norm_(model.parameters(), CLIP_NORM)

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

        total_loss += loss.detach().float().item()
        n_batches += 1

    accelerator.print(
        f"  Epoch {epoch+1}/4  "
        f"avg_loss={total_loss/n_batches:.5f}  "
        f"opt_steps={step_count}"
    )

# ── Explain the accumulate() context in detail ───────────────────────────────
banner("5. How accumulate() works internally")
print("""
The accelerate.accumulate(model) context manager:

  global_step = 0
  for micro_batch in dataloader:
      with accelerator.accumulate(model):
          # If NOT last micro-step in the accumulation window:
          #   model.no_sync()  (DDP: skip all-reduce)
          loss = model(micro_batch)
          accelerator.backward(loss)         # gradient += micro_grad
          optimizer.step()                   # NO-OP until last step
          optimizer.zero_grad()              # NO-OP until last step
      # After the K-th micro-step, optimizer.step() ACTUALLY fires
      global_step += 1

This is equivalent to manually:
    for i, micro in enumerate(micro_batches):
        if (i + 1) % K != 0:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
            optimizer.step(); optimizer.zero_grad()

Why bother with the context manager?  It handles edge cases:
  - Last batch may not be divisible by K → still steps
  - In DDP: controls no_sync() correctly per-rank
  - With GradScaler (fp16): integrates scale/unscale automatically

Effective batch size math:
  B_eff = micro_batch_size × grad_accum_steps × num_gpus
  e.g.   8               × 4                  × 1        = 32
""")

# ── Mixed precision detail ───────────────────────────────────────────────────
banner("6. Mixed precision internals (fp16 / bf16)")
print(f"""
On GPU (the real use case):
  fp16: weights in fp32 (master), forward/backward in fp16.
        GradScaler multiplies loss by a large scale factor S before backward
        to prevent fp16 underflow, then unscales before optimizer.step().
  bf16: no scaler needed (same exponent range as fp32); torch.autocast handles
        casting forward pass ops to bf16 automatically.

On CPU (this demo, mixed_precision='{PRECISION}'):
  {'bf16: ops cast to bfloat16 where PyTorch has CPU kernels (limited ops).' if PRECISION=='bf16' else 'fp32 (no-op): mixed_precision=no means standard fp32 training.'}
  fp16 on CPU: NOT recommended — no hardware support, likely slower.

Accelerate usage:
  accelerator = Accelerator(mixed_precision='bf16')  # or 'fp16', 'no'
  # Then just call accelerator.backward(loss) — scaling is automatic.
  # No need to touch GradScaler or autocast directly.
""")

print("\nDONE — exit 0")
sys.exit(0)

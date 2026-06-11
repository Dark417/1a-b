"""
Accelerate CPU Training Loop — Runnable Example
================================================
Docs: https://huggingface.co/docs/accelerate/basic_tutorials/overview
      https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation

Demonstrates:
- Accelerator() on CPU
- prepare(model, optimizer, dataloader)
- training loop with accelerator.backward()
- gradient accumulation via accumulate() context manager
- accelerator.print (only prints on main process in distributed)
- accelerator.gather_for_metrics for collecting scalars
- accelerator.save_state / load_state checkpoint
"""

import sys
import os
import random
import tempfile

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import numpy as np
np.random.seed(0)
random.seed(0)

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

# ── 1. Tiny synthetic regression dataset ────────────────────────────────────
banner("1. Build synthetic dataset (CPU, no downloads)")

N_SAMPLES = 64
IN_FEATURES = 16
OUT_FEATURES = 1
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2     # effective batch = 8 * 2 = 16

X = torch.randn(N_SAMPLES, IN_FEATURES)
# y = Xw + noise   (learnable signal)
w_true = torch.randn(IN_FEATURES, OUT_FEATURES)
y = X @ w_true + 0.05 * torch.randn(N_SAMPLES, OUT_FEATURES)

dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True
)
print(f"  Dataset: {N_SAMPLES} samples, features={IN_FEATURES}")
print(f"  DataLoader: batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}")

# ── 2. Tiny model ────────────────────────────────────────────────────────────
banner("2. Define tiny MLP (no downloads)")

import torch.nn as nn

class TinyMLP(nn.Module):
    """Two-layer MLP for regression."""
    def __init__(self, in_f, hidden, out_f):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_f),
        )

    def forward(self, x):
        return self.net(x)

model = TinyMLP(IN_FEATURES, 32, OUT_FEATURES)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params}")

# ── 3. Accelerator setup ─────────────────────────────────────────────────────
banner("3. Accelerator setup")

from accelerate import Accelerator

# gradient_accumulation_steps passed here so .accumulate() works correctly
accelerator = Accelerator(
    cpu=True,               # force CPU (also auto-selected when no GPU)
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    mixed_precision="no",   # CPU; fp16 AMP not supported on CPU
)
print(f"  Device           : {accelerator.device}")
print(f"  Mixed precision  : {accelerator.mixed_precision}")
print(f"  Num processes    : {accelerator.num_processes}")
print(f"  Distributed type : {accelerator.distributed_type}")

# ── 4. prepare() — the core Accelerate API call ──────────────────────────────
banner("4. accelerator.prepare(model, optimizer, dataloader)")

# NOTE: CALL prepare() BEFORE any .to(device) — Accelerate handles placement.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
loss_fn = nn.MSELoss()

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
# After prepare(), model lives on accelerator.device automatically.
print("  prepare() done — model/optimizer/dataloader wrapped")

# ── 5. Training loop ─────────────────────────────────────────────────────────
banner("5. Training loop — 3 epochs")

N_EPOCHS = 3

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in dataloader:
        # accumulate() context: synchronises gradient sync correctly in DDP;
        # on CPU/single-process it just tracks the step counter.
        with accelerator.accumulate(model):
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)

            # ALWAYS use accelerator.backward instead of loss.backward():
            # - Handles mixed-precision loss scaling
            # - Handles no_sync() in gradient-accumulation for DDP
            accelerator.backward(loss)

            # Gradient clipping AFTER backward, BEFORE optimizer step.
            # Only clips if not currently accumulating (handled internally).
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.detach().item()
        n_batches += 1

    # accelerator.print only prints on rank-0 in multi-process — safe everywhere
    avg = epoch_loss / n_batches
    accelerator.print(f"  Epoch {epoch+1}/{N_EPOCHS}  avg_loss={avg:.5f}")

# ── 6. accelerator.gather_for_metrics ────────────────────────────────────────
banner("6. gather_for_metrics — collecting tensors across processes")

# Simulate a per-batch metric tensor (would come from eval in practice)
sample_metric = torch.tensor([0.42])           # on accelerator.device already
gathered = accelerator.gather_for_metrics(sample_metric)
print(f"  gathered metric shape: {gathered.shape}, value: {gathered}")
print("  (In multi-GPU, this concatenates tensors from all ranks)")

# ── 7. save_state / load_state ───────────────────────────────────────────────
banner("7. save_state / load_state")

with tempfile.TemporaryDirectory() as ckpt_dir:
    accelerator.save_state(ckpt_dir)
    print(f"  Saved state to: {ckpt_dir}/")
    import os as _os
    files = _os.listdir(ckpt_dir)
    print(f"  Checkpoint files: {files}")

    # Load it back (round-trip)
    accelerator.load_state(ckpt_dir)
    print("  load_state() succeeded — weights/optimizer restored")

# ── 8. Key API summary ───────────────────────────────────────────────────────
banner("8. API Summary")
print("""
Key Accelerate calls used:
  accelerator = Accelerator(cpu=True, gradient_accumulation_steps=N)
  model, opt, dl = accelerator.prepare(model, opt, dl)
  with accelerator.accumulate(model):  # gradient-accum context
      accelerator.backward(loss)
      accelerator.clip_grad_norm_(model.parameters(), 1.0)
      opt.step(); opt.zero_grad()
  accelerator.print(...)               # rank-0 only in distributed
  accelerator.gather_for_metrics(t)    # all-gather across ranks
  accelerator.save_state(path)
  accelerator.load_state(path)

Docs: https://huggingface.co/docs/accelerate/package_reference/accelerator
""")

print("\nDONE — exit 0")
sys.exit(0)

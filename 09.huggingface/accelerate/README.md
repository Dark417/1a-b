# HuggingFace Accelerate — University-Grade Explainer

> Official docs: https://huggingface.co/docs/accelerate/index
> GitHub: https://github.com/huggingface/accelerate

---

## 1. What is Accelerate?

**Accelerate** is HuggingFace's answer to the question: *how do I run the same training loop on a laptop CPU, a single GPU, 8 GPUs with DDP, or a TPU pod — without rewriting it each time?*

The answer: wrap your training objects with `Accelerator.prepare()`, replace `loss.backward()` with `accelerator.backward(loss)`, and let the library handle:

- **Device placement** (`.to(device)` for you)
- **Distributed data-parallel (DDP)** wrapping
- **Mixed-precision** (`fp16` / `bf16`) casting and loss scaling
- **Gradient accumulation** synchronisation across processes
- **Gradient clipping** via `accelerator.clip_grad_norm_`
- **Checkpoint save / resume** with `save_state` / `load_state`
- **DeepSpeed** and **FSDP** (Fully-Sharded Data Parallel) as drop-in plugins

The core promise is *one codebase, every accelerator*.

---

## 2. The `Accelerator` Object

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",           # "no" | "fp16" | "bf16" | "fp8"
    gradient_accumulation_steps=4,    # virtual batch multiplier
    cpu=False,                        # force CPU even if GPU present
    log_with="tensorboard",           # optional experiment tracking
    project_dir="./logs",
)
```

On construction the object:
1. Detects the hardware (CPU / CUDA / MPS / XLA).
2. Reads `~/.cache/huggingface/accelerate/default_config.yaml` if it exists (written by `accelerate config`).
3. Sets up the distributed process group if running under `accelerate launch` or `torchrun`.

Useful attributes after construction:

| Attribute | Meaning |
|---|---|
| `accelerator.device` | The torch device for this process |
| `accelerator.num_processes` | Total world size |
| `accelerator.process_index` | Global rank of this process |
| `accelerator.is_main_process` | `True` only on rank 0 |
| `accelerator.mixed_precision` | Configured mixed-precision string |

---

## 3. `.prepare()` — the Central API

```python
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)
```

`prepare` does **all** of:
- Moves `model` to the right device.
- Wraps `model` in `DistributedDataParallel` (or FSDP / DeepSpeed module) when using multiple processes.
- Wraps `DataLoader` so each process gets its own shard (via `DistributedSampler`).
- Wraps the optimizer for mixed-precision loss scaling.

You pass any combination of (model, optimizer, scheduler, dataloader) in one call or across multiple calls.

---

## 4. Training Loop Pattern

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # gradient_accumulation_steps context — handles sync correctly
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            # ALWAYS use accelerator.backward instead of loss.backward()
            # This handles mixed-precision scaler + gradient sync
            accelerator.backward(loss)

            # clip_grad_norm_ works on the unwrapped params in all modes
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

    # Gather metric across all processes → tensor on each process
    all_preds = accelerator.gather(preds)
```

### Why `accelerator.backward()` instead of `loss.backward()`?

In fp16 mode the loss is *scaled* (multiplied by a large constant) before the backward pass to prevent gradient underflow. `accelerator.backward()` applies the scale before the pass and unscales after. Calling `loss.backward()` directly would bypass scaling.

---

## 5. Mixed Precision: fp16 vs bf16

| | fp16 | bf16 |
|---|---|---|
| Dynamic range | Smaller — needs loss scaling | Larger — same as fp32 range |
| Precision | 10-bit mantissa | 7-bit mantissa |
| Hardware support | All CUDA GPUs | Ampere+ GPUs, TPUs |
| Loss scaling needed | Yes (GradScaler) | No |
| Numerics | Can diverge on sensitive nets | More stable |

On CPU, mixed precision is effectively a **no-op** (torch ops run in fp32). Accelerate will emit a warning and continue. You can still write `Accelerator(mixed_precision="bf16")` in CPU-only code to test your config is wired up correctly.

---

## 6. Gradient Accumulation — The Math

Gradient accumulation simulates a larger batch without increasing memory.

Suppose `batch_size=4` and `gradient_accumulation_steps=8`.
- Effective batch size = `4 × 8 = 32`.
- Each micro-step accumulates gradients (sums them up) without calling `optimizer.step()`.
- Every 8 steps: `optimizer.step()` + `optimizer.zero_grad()`.

Mathematically, because gradients of a sum equal the sum of gradients:

```
∇L(B_eff) ≈ (1/K) * Σ_{k=1}^{K} ∇L(B_k)
```

The `accumulate(model)` context manager handles DDP gradient synchronisation: it suppresses the all-reduce sync on micro-steps and only syncs on the final step, which is a significant communication saving.

---

## 7. Saving and Loading State

```python
# Save full training state (model weights + optimizer + scheduler + RNG)
accelerator.save_state("./checkpoint_dir")

# Resume (restores everything including RNG state for reproducibility)
accelerator.load_state("./checkpoint_dir")

# Save just the model (unwrap DDP wrapper first)
unwrapped = accelerator.unwrap_model(model)
unwrapped.save_pretrained("./my_model")
```

`save_state` is process-aware: on multi-GPU runs only rank 0 writes to disk.

---

## 8. `accelerate config` and `accelerate launch`

### `accelerate config`

An interactive CLI wizard that asks you:
- Single machine or multi-node?
- How many GPUs? (or TPU)
- Do you want DeepSpeed / FSDP?
- Mixed precision?

It writes `~/.cache/huggingface/accelerate/default_config.yaml`. See `03.accelerate_config.md` for a full walkthrough.

### `accelerate launch`

```bash
# Single GPU
accelerate launch train.py

# Multi-GPU with 4 GPUs, bf16
accelerate launch --num_processes 4 --mixed_precision bf16 train.py

# Override config without editing the YAML
accelerate launch --config_file ./my_config.yaml train.py
```

`accelerate launch` handles spawning the right number of processes (like `torchrun`) and injects environment variables (`LOCAL_RANK`, `RANK`, `WORLD_SIZE`) so your script just calls `Accelerator()` and it all works.

---

## 9. DDP / FSDP / DeepSpeed Integration

### DDP (DistributedDataParallel)
The default multi-GPU mode. Each process holds a full model copy. Gradients are all-reduced after each backward pass.

### FSDP (Fully Sharded Data Parallel)
Parameters, gradients, and optimizer states are sharded across GPUs. Allows training models much larger than single-GPU memory. Enable via:
```yaml
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_sharding_strategy: FULL_SHARD
```

### DeepSpeed
Microsoft's optimisation library. Stages:
- **ZeRO-1**: shard optimizer states
- **ZeRO-2**: shard optimizer + gradients
- **ZeRO-3**: shard everything (params too)

Enable via `accelerate config` selecting DeepSpeed, or pass `DeepSpeedPlugin`.

---

## 10. `notebook_launcher`

For Jupyter notebooks, `accelerate launch` can't fork processes in the usual way. Use:

```python
from accelerate import notebook_launcher

def training_fn():
    accelerator = Accelerator()
    # ... your training loop

notebook_launcher(training_fn, args=(), num_processes=2)
```

---

## 11. Accelerate vs Raw torch DDP vs `Trainer`

| Concern | Raw torch DDP | Accelerate | `Trainer` |
|---|---|---|---|
| Boilerplate | Very high | Low | None |
| Flexibility | Complete | High | Medium |
| Mixed precision | Manual GradScaler | 1 kwarg | 1 kwarg |
| Gradient accumulation | Manual | `accumulate()` ctx | 1 kwarg |
| FSDP/DeepSpeed | Manual plugin | Config / plugin | Config |
| Custom loops | Yes | Yes | Limited |
| Non-HF models | Yes | Yes | Awkward |
| Best for | Power users | Custom loops + distributed | Standard HF workflows |

---

## Files in this directory

| File | Description |
|---|---|
| `01.cpu_training_loop.py` | End-to-end Accelerate training loop on CPU |
| `02.mixed_precision_grad_accum.py` | Mixed precision + gradient accumulation demo |
| `03.accelerate_config.md` | `accelerate config` walkthrough & YAML reference |
| `requirements.txt` | Python dependencies |

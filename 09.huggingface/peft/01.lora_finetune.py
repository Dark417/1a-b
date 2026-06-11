"""
PEFT LoRA Fine-Tuning — Runnable Example
=========================================
Docs: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
      https://huggingface.co/docs/peft/conceptual_guides/lora
      https://huggingface.co/docs/peft/package_reference/lora

Demonstrates:
- LoraConfig with r, lora_alpha, target_modules, task_type
- get_peft_model() wrapping a base CausalLM
- print_trainable_parameters() (shows tiny %)
- Training a few steps on an in-memory dataset
- Loss printing

Offline safety:
  Tries to download sshleifer/tiny-gpt2 for a realistic tokenizer.
  On any network/model failure: builds GPT2LMHeadModel from local
  GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=256) —
  training still runs with random integer "token" inputs.
"""

import sys
import os
import random
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import numpy as np
np.random.seed(0)
random.seed(0)

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

# ── 1. Load or build a tiny base model ──────────────────────────────────────
banner("1. Load base model (tiny-gpt2 or local fallback)")

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

VOCAB_SIZE = 256    # used only in fallback

def load_model_and_tokenizer():
    """Try to load tiny-gpt2 from HF Hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    return model, tokenizer, "hub"

def build_local_model():
    """Build a tiny GPT-2 from config — no download needed."""
    cfg = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=VOCAB_SIZE,
        n_positions=64,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    return model, None, "local"

ok, result = safe(load_model_and_tokenizer)
if ok:
    model, tokenizer, source = result
    VOCAB_SIZE = model.config.vocab_size
    print(f"  Loaded sshleifer/tiny-gpt2 from hub (vocab={VOCAB_SIZE})")
else:
    note_skip(f"needs network/model — demonstrating API shape with local fallback ({result})")
    model, tokenizer, source = build_local_model()
    print(f"  Built local GPT2 (vocab={VOCAB_SIZE}, n_embd=64, n_layer=2)")

n_base = sum(p.numel() for p in model.parameters())
print(f"  Base model parameters: {n_base:,}")

# ── 2. LoraConfig + get_peft_model ──────────────────────────────────────────
banner("2. LoraConfig + get_peft_model")

from peft import LoraConfig, TaskType, get_peft_model

# For GPT-2, the attention QKV is in a single Conv1D named 'c_attn'.
# For hub tiny-gpt2 and local GPT2Config this is the same.
# fan_in_fan_out=True is set automatically for Conv1D by PEFT.

lora_config = LoraConfig(
    r=4,                          # Low-rank dimension
    lora_alpha=8,                 # Scaling: effective Δ = (8/4) * B*A = 2 * B*A
    target_modules=["c_attn"],    # GPT-2 fused QKV projection
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

print(f"  r={lora_config.r}, alpha={lora_config.lora_alpha}")
print(f"  target_modules={lora_config.target_modules}")
print(f"  scale factor (alpha/r) = {lora_config.lora_alpha/lora_config.r:.2f}")
print(f"  => ΔW = (alpha/r) * B @ A  (B zero-init, A gaussian-init)")

peft_model = get_peft_model(model, lora_config)

# ── 3. print_trainable_parameters ───────────────────────────────────────────
banner("3. print_trainable_parameters — shows the tiny %")

peft_model.print_trainable_parameters()

n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
n_total     = sum(p.numel() for p in peft_model.parameters())
print(f"  Trainable : {n_trainable:,}")
print(f"  Total     : {n_total:,}")
print(f"  %         : {100.*n_trainable/n_total:.4f}%")

# ── 4. Build tiny in-memory dataset ─────────────────────────────────────────
banner("4. Tiny in-memory dataset (no hub download)")

import datasets as hf_datasets

# A handful of short sentences as training data
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning transforms the way we build software.",
    "Language models predict the next token given prior context.",
    "Fine-tuning adapts a pre-trained model to a specific domain.",
    "LoRA trains only low-rank weight deltas, keeping the base frozen.",
    "Gradient descent minimises the loss by following the slope downhill.",
    "Attention mechanisms allow models to relate distant tokens.",
    "The embedding layer maps token IDs to dense vectors.",
]

if tokenizer is not None:
    # Real tokenizer: encode the texts
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    MAX_LEN = 32
    encoded = tokenizer(
        TEXTS,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    print(f"  Tokenized {len(TEXTS)} sentences, max_len={MAX_LEN}")
else:
    # Local fallback: random integer tokens in [1, VOCAB_SIZE-1]
    MAX_LEN = 32
    input_ids = torch.randint(1, VOCAB_SIZE, (len(TEXTS), MAX_LEN))
    print(f"  Local fallback: {len(TEXTS)} random-token sequences, len={MAX_LEN}")

# For causal LM: labels = input_ids (standard next-token prediction)
labels = input_ids.clone()

print(f"  input_ids shape: {input_ids.shape}")

# ── 5. Training loop ─────────────────────────────────────────────────────────
banner("5. Training loop — 10 steps, showing loss going down")

DEVICE = torch.device("cpu")
peft_model.to(DEVICE)
peft_model.train()

optimizer = torch.optim.AdamW(
    [p for p in peft_model.parameters() if p.requires_grad],
    lr=5e-4
)

BATCH = 2
N_STEPS = 10

print(f"  Batch size: {BATCH}, steps: {N_STEPS}")
print(f"  Only {n_trainable:,} params receive gradients (LoRA A/B matrices)\n")

ds_input  = input_ids
ds_labels = labels

for step in range(N_STEPS):
    # Sample a mini-batch
    idx = torch.randperm(len(ds_input))[:BATCH]
    batch_ids = ds_input[idx].to(DEVICE)
    batch_lbl = ds_labels[idx].to(DEVICE)

    optimizer.zero_grad()
    out = peft_model(input_ids=batch_ids, labels=batch_lbl)
    loss = out.loss
    loss.backward()
    optimizer.step()

    if step == 0 or (step + 1) % 2 == 0:
        print(f"  step {step+1:3d}/{N_STEPS}  loss={loss.item():.4f}")

print("\n  Loss printed above. LoRA ΔW is being learned while base is frozen.")

# ── 6. Verify base model frozen ──────────────────────────────────────────────
banner("6. Verify base weights are frozen")

frozen = [n for n, p in peft_model.named_parameters() if not p.requires_grad]
trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
print(f"  Frozen params  : {len(frozen)} tensors")
print(f"  Trainable params: {len(trainable)} tensors")
print("  Trainable names:", trainable[:6])
print("  (All trainable names contain 'lora_' — only the LoRA matrices train)")

# ── 7. LoRA math recap ────────────────────────────────────────────────────────
banner("7. LoRA math recap")
print("""
For a weight W ∈ R^{d × k}:

  W_adapted(x) = x @ W.T  +  (alpha/r) * x @ A.T @ B.T

  A ∈ R^{r × k}    — initialised with N(0, σ²)
  B ∈ R^{d × r}    — initialised with zeros → ΔW = 0 at start

  Only A and B are trained.  W is frozen.

  With r=4, d=k=64:
    Full ΔW would need 64×64 = 4096 params
    LoRA needs 2 × 4×64 = 512 params   (8× reduction)

  With r=4, alpha=8:
    scale = alpha/r = 2.0
    This is equivalent to using a 2× higher learning rate for LoRA params.
""")

print("\nDONE — exit 0")
sys.exit(0)

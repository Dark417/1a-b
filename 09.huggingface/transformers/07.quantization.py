"""
07.quantization.py — Quantization Concepts and Demonstrations
===============================================================
Quantization reduces model precision (e.g., from float32 to int8/int4),
shrinking memory and speeding up inference.

Official docs:
  https://huggingface.co/docs/transformers/quantization/overview
  https://huggingface.co/docs/transformers/quantization/bitsandbytes
  https://huggingface.co/docs/transformers/quantization/gptq
  https://huggingface.co/docs/transformers/quantization/awq

This file:
  1. Explains each quantization method conceptually (in comments)
  2. Constructs BitsAndBytesConfig objects (no GPU required)
  3. Demonstrates torch dtype casting on a tiny local model (CPU-safe)
  4. Shows what quantized layer weights look like conceptually
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel,
    BertConfig, BertModel,
    GPT2Config, GPT2LMHeadModel,
)

# ─── 1. Why Quantization? ────────────────────────────────────────────────────
banner("1. Why Quantization?")
print("""
  float32 (fp32): 32 bits per parameter — the default
  float16 (fp16): 16 bits → 2× memory savings, hardware-accelerated on modern GPUs
  bfloat16 (bf16): 16 bits, wider exponent than fp16 → more numerically stable
  int8:  8 bits → 4× savings vs fp32 — bitsandbytes LLM.int8()
  int4:  4 bits → 8× savings vs fp32 — GPTQ, AWQ, bitsandbytes nf4

  A 7B param model:
    fp32  → 28 GB
    fp16  → 14 GB
    int8  →  7 GB
    int4  →  3.5 GB   (plus small overhead for scales/zeros)
""")

# ─── 2. BitsAndBytesConfig (int8) ────────────────────────────────────────────
banner("2. BitsAndBytesConfig — int8 (LLM.int8())")
# LLM.int8() (Dettmers et al. 2022) uses MIXED PRECISION:
#   - Finds "outlier" feature dimensions (large magnitude activations)
#   - Keeps those in fp16
#   - Quantizes the rest to int8
# Net effect: ~1.5× slowdown on inference but ~2× memory savings.
# REQUIRES bitsandbytes + CUDA GPU.

ok, BitsAndBytesConfig = safe(__import__, "transformers")
try:
    from transformers import BitsAndBytesConfig

    bnb_int8 = BitsAndBytesConfig(load_in_8bit=True)
    print(f"  BitsAndBytesConfig(load_in_8bit=True):")
    print(f"    load_in_8bit         : {bnb_int8.load_in_8bit}")
    print(f"    load_in_4bit         : {bnb_int8.load_in_4bit}")
    print()
    print("  Usage:")
    print("    model = AutoModelForCausalLM.from_pretrained(")
    print("        'meta-llama/Llama-2-7b-hf',")
    print("        quantization_config=bnb_int8,")
    print("        device_map='auto',   # requires a GPU")
    print("    )")
    print("  NOTE: Requires CUDA GPU and bitsandbytes package.")
except Exception as e:
    note_skip(str(e))
    print("  BitsAndBytesConfig is part of transformers; object construction is free.")
    print("  GPU is only required when actually loading a model with quantization.")

# ─── 3. BitsAndBytesConfig (int4 / NF4 / QLoRA) ─────────────────────────────
banner("3. BitsAndBytesConfig — int4 / NF4 (QLoRA)")
# NF4 (Normal Float 4) quantizes weights assuming normally-distributed values.
# QLoRA (Dettmers et al. 2023) adds:
#   - Double quantization: quantize the quantization constants themselves
#   - bfloat16 compute dtype for stable forward/backward passes
# This enables fine-tuning 65B models on a single GPU.

try:
    from transformers import BitsAndBytesConfig

    bnb_nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4 (better than uniform int4)
        bnb_4bit_use_double_quant=True,      # Double quant: saves ~0.37 bits/param extra
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 for numerics during compute
    )
    print(f"  BitsAndBytesConfig (NF4 / QLoRA):")
    print(f"    load_in_4bit              : {bnb_nf4.load_in_4bit}")
    print(f"    bnb_4bit_quant_type       : {bnb_nf4.bnb_4bit_quant_type}")
    print(f"    bnb_4bit_use_double_quant : {bnb_nf4.bnb_4bit_use_double_quant}")
    print(f"    bnb_4bit_compute_dtype    : {bnb_nf4.bnb_4bit_compute_dtype}")
    print()
    print("  Usage:")
    print("    model = AutoModelForCausalLM.from_pretrained(")
    print("        'meta-llama/Llama-2-13b-hf',")
    print("        quantization_config=bnb_nf4,")
    print("        device_map='auto',")
    print("    )")
    print("  THEN add LoRA adapters for fine-tuning (see PEFT library).")
except Exception as e:
    note_skip(str(e))

# ─── 4. GPTQ ─────────────────────────────────────────────────────────────────
banner("4. GPTQ — Post-Training Quantization via Hessians")
print("""
  GPTQ (Frantar et al. 2022):
    - PTQ (post-training quantization): quantize after training, no retraining
    - Uses the Hessian of the loss to minimize quantization error
    - Typical: 4-bit weights, float16 activations
    - Result: ~4× memory savings, ~3-4× speedup on GPU (with exllama/triton kernels)

  Config example (requires auto-gptq package):
    from transformers import GPTQConfig
    gptq_cfg = GPTQConfig(
        bits=4,                # quantize to 4 bits
        group_size=128,        # 128 weights share quantization params
        dataset="c4",          # calibration dataset
        tokenizer=tokenizer,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "model-name",
        quantization_config=gptq_cfg,
        device_map="auto",
    )

  Pre-quantized models are available on HuggingFace Hub:
    TheBloke/Llama-2-7B-GPTQ
    TheBloke/Mistral-7B-v0.1-GPTQ
""")

# Try to show GPTQConfig if available
try:
    from transformers import GPTQConfig
    cfg = GPTQConfig(bits=4, dataset="wikitext2", group_size=128)
    print(f"  GPTQConfig constructed: bits={cfg.bits}, group_size={cfg.group_size}")
except Exception as e:
    print(f"  GPTQConfig not available: {e}")
    print("  (Install auto-gptq to enable)")

# ─── 5. AWQ ──────────────────────────────────────────────────────────────────
banner("5. AWQ — Activation-Aware Weight Quantization")
print("""
  AWQ (Lin et al. 2023):
    - Observes that ~1% of weight channels are "salient" (high activation magnitude)
    - Scales those channels before quantization to preserve quality
    - Result: often better than GPTQ at 4-bit, faster inference with fused kernels

  Config example (requires autoawq package):
    from transformers import AwqConfig
    awq_cfg = AwqConfig(
        bits=4,
        fuse_max_seq_len=512,   # fuse ops for sequences up to this length
        do_fuse=True,           # fuse attention + MLP for speedup
    )
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-AWQ",
        quantization_config=awq_cfg,
        device_map="auto",
    )

  Pre-quantized:
    TheBloke/Llama-2-7B-AWQ
    TheBloke/Mistral-7B-v0.1-AWQ
""")

try:
    from transformers import AwqConfig
    awq = AwqConfig(bits=4)
    print(f"  AwqConfig constructed: bits={awq.bits}")
except Exception as e:
    print(f"  AwqConfig not available: {e}")
    print("  (Install autoawq to enable)")

# ─── 6. dtype casting on a tiny model (CPU-safe) ─────────────────────────────
banner("6. dtype Casting Demo (CPU-safe, tiny model)")
# We build a tiny BERT locally (no download) and show dtype casting.
# This is what from_pretrained(..., torch_dtype=torch.float16) does internally.

cfg = BertConfig(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
    intermediate_size=128, vocab_size=1000,
)
model_f32 = BertModel(cfg)

def dtype_of(m):
    return next(m.parameters()).dtype

def param_bytes(m):
    return sum(p.nelement() * p.element_size() for p in m.parameters())

print(f"  float32 model param bytes : {param_bytes(model_f32):,}")
print(f"  float32 dtype             : {dtype_of(model_f32)}")

# Cast to float16
model_f16 = BertModel(cfg).half()
print(f"\n  float16 model param bytes : {param_bytes(model_f16):,}")
print(f"  float16 dtype             : {dtype_of(model_f16)}")
print(f"  Savings ratio             : {param_bytes(model_f32) / param_bytes(model_f16):.1f}×")

# Cast to bfloat16
model_bf16 = BertModel(cfg).to(torch.bfloat16)
print(f"\n  bfloat16 model param bytes: {param_bytes(model_bf16):,}")
print(f"  bfloat16 dtype            : {dtype_of(model_bf16)}")

# Forward pass comparison (CPU, small input)
dummy_ids = torch.randint(0, 1000, (1, 8))
dummy_mask = torch.ones(1, 8, dtype=torch.long)

with torch.no_grad():
    out_f32 = model_f32(input_ids=dummy_ids, attention_mask=dummy_mask)
    out_f16 = model_f16(input_ids=dummy_ids, attention_mask=dummy_mask)
    out_bf16 = model_bf16(input_ids=dummy_ids, attention_mask=dummy_mask)

print(f"\n  Forward pass output dtypes:")
print(f"    fp32  last_hidden_state: {out_f32.last_hidden_state.dtype}")
print(f"    fp16  last_hidden_state: {out_f16.last_hidden_state.dtype}")
print(f"    bf16  last_hidden_state: {out_bf16.last_hidden_state.dtype}")

# Numerical difference between fp32 and fp16
diff = (out_f32.last_hidden_state - out_f16.last_hidden_state.float()).abs().mean().item()
print(f"\n  Mean |fp32 - fp16| output diff: {diff:.6f}")

# ─── 7. Simulated int8 quantization (educational) ────────────────────────────
banner("7. Manual int8 Quantization Simulation (educational)")
# Real int8 quantization uses specialized CUDA kernels.
# Here we simulate the math: scale → round → clamp → dequantize.
#
# Per-tensor symmetric quantization:
#   scale = max(|W|) / 127
#   W_int8 = round(W / scale).clamp(-128, 127)
#   W_approx = W_int8 * scale  (dequantized)

W = torch.randn(4, 4)  # toy weight matrix
print(f"  Original W (fp32):\n{W}")

scale = W.abs().max() / 127.0
W_int8 = (W / scale).round().clamp(-128, 127).to(torch.int8)
W_dequant = W_int8.float() * scale

print(f"\n  scale = {scale:.6f}")
print(f"  W_int8:\n{W_int8}")
print(f"\n  W_dequant (approx fp32):\n{W_dequant}")
print(f"\n  Quantization error (mean abs): {(W - W_dequant).abs().mean().item():.6f}")

# Practical note
print("""
  Note on hardware requirements:
    int8/int4 inference: requires CUDA GPU (bitsandbytes only supports CUDA)
    dtype casting (fp16/bf16): works on CPU and GPU
    GPTQ/AWQ inference: requires GPU for kernel-level speedup
    All CONFIG OBJECTS can be constructed on CPU for documentation purposes.
""")

print("\n[DONE] 07.quantization.py complete")

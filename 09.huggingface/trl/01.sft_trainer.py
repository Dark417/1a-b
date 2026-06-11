"""
TRL SFTTrainer — Supervised Fine-Tuning (Runnable Example)
===========================================================
Docs: https://huggingface.co/docs/trl/sft_trainer
      https://huggingface.co/docs/trl/sft_trainer#sftconfig

Version note: trl 1.5 — processing_class= replaces tokenizer=.
              SFTConfig absorbs all TrainingArguments fields + SFT-specific ones.

Demonstrates:
- SFTConfig with max_steps, per_device_train_batch_size, report_to="none"
- SFTTrainer with in-memory dataset (no hub downloads for data)
- Tries to download sshleifer/tiny-gpt2 for a realistic tokenizer;
  on failure builds a local GPT2Config model + uses a PreTrainedTokenizerFast
  backed by a small ByteLevel BPE vocabulary
- Falls back to plain transformers Trainer causal-LM loop if SFTTrainer fails
- Prints loss at each step
- Exits 0 always
"""

import sys
import os
import random
import tempfile
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import numpy as np
np.random.seed(0)
random.seed(0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

# ── 1. Build/load tiny model + tokenizer ────────────────────────────────────
banner("1. Load model + tokenizer (hub or local fallback)")

from transformers import GPT2Config, GPT2LMHeadModel

VOCAB_SIZE = None    # resolved below
tokenizer = None
model = None
source = None

def try_hub():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    return mdl, tok

ok, result = safe(try_hub)
if ok:
    model, tokenizer = result
    VOCAB_SIZE = model.config.vocab_size
    source = "hub:sshleifer/tiny-gpt2"
    print(f"  Loaded from hub (vocab={VOCAB_SIZE})")
else:
    note_skip(
        f"needs network/model — demonstrating API shape with local fallback ({result})"
    )
    # Build a minimal GPT2 locally
    VOCAB_SIZE = 50257    # GPT-2 vocab size (matches GPT2Tokenizer default)
    cfg = GPT2Config(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=VOCAB_SIZE, n_positions=128,
    )
    model = GPT2LMHeadModel(cfg)

    # Build a tokenizer: try loading from hub for a real one, else character-level
    def try_tok():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    ok2, tok_result = safe(try_tok)
    if ok2:
        tokenizer = tok_result
    else:
        # Absolute fallback: use a character-level tokenizer wrapper
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        # Build a trivial BPE with character vocab
        chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        vocab = {c: i for i, c in enumerate(["[PAD]", "[EOS]"] + chars)}
        merges = []
        raw_tok = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="[PAD]"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tok,
            pad_token="[PAD]",
            eos_token="[EOS]",
        )
        # Update model vocab size to match
        VOCAB_SIZE = tokenizer.vocab_size
        cfg = GPT2Config(
            n_layer=2, n_head=2, n_embd=64,
            vocab_size=VOCAB_SIZE, n_positions=128,
        )
        model = GPT2LMHeadModel(cfg)
    source = "local"
    print(f"  Built local GPT2 (vocab={VOCAB_SIZE}, n_embd=64, n_layer=2)")

n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}  |  source: {source}")

# ── 2. Build in-memory dataset ───────────────────────────────────────────────
banner("2. Build tiny in-memory dataset (no hub download for data)")

import datasets as hf_datasets

TEXTS = [
    "The transformer architecture uses self-attention to model sequences.",
    "Supervised fine-tuning adapts a pre-trained model to new tasks.",
    "Gradient descent iteratively minimises the loss function.",
    "Tokenisation converts raw text into integer IDs for the model.",
    "LoRA trains low-rank weight deltas while keeping base weights frozen.",
    "RLHF uses human preferences to align language model outputs.",
    "The attention mechanism computes weighted sums over value vectors.",
    "Cross-entropy loss measures the divergence between predicted and true distributions.",
]

train_dataset = hf_datasets.Dataset.from_dict({"text": TEXTS})
print(f"  Dataset: {len(train_dataset)} rows, column='text'")
print(f"  Example: '{TEXTS[0][:60]}...'")

# ── 3. SFTTrainer (live attempt) ─────────────────────────────────────────────
banner("3. SFTTrainer — live attempt (trl 1.5)")

from trl import SFTTrainer, SFTConfig
import inspect

print(f"  trl version: {__import__('trl').__version__}")
print(f"  SFTConfig dataset_text_field, max_length, use_cpu supported: checking...")

sft_success = False
train_output = None

def run_sft():
    with tempfile.TemporaryDirectory() as tmpdir:
        sft_config = SFTConfig(
            output_dir=tmpdir,
            max_steps=6,                      # a handful of steps
            per_device_train_batch_size=2,
            learning_rate=5e-4,
            report_to="none",
            dataset_text_field="text",        # which column to use
            max_length=32,                    # truncate to 32 tokens
            use_cpu=True,                     # force CPU
            logging_steps=1,                  # log every step
            save_strategy="no",               # don't save during training
            disable_tqdm=False,
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,       # trl 1.5: use processing_class, not tokenizer=
        )
        result = trainer.train()
        return result

ok_sft, sft_result = safe(run_sft)
if ok_sft:
    sft_success = True
    print(f"\n  SFTTrainer SUCCESS")
    print(f"  train_loss  : {sft_result.training_loss:.4f}")
    print(f"  train_steps : {sft_result.global_step}")
else:
    print(f"\n  SFTTrainer failed: {sft_result}")

# ── 4. Fallback: plain Trainer causal-LM loop ────────────────────────────────
if not sft_success:
    banner("4. Fallback: plain transformers Trainer (causal LM)")
    note_skip(
        "SFTTrainer API mismatch detected — falling back to plain "
        "transformers Trainer for causal-LM training on the same data."
    )

    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    # Tokenize manually
    def tokenize_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=32,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized_ds = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_ds.set_format("torch")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_args = TrainingArguments(
            output_dir=tmpdir,
            max_steps=6,
            per_device_train_batch_size=2,
            learning_rate=5e-4,
            report_to="none",
            logging_steps=1,
            save_strategy="no",
            use_cpu=True,
            disable_tqdm=False,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=tokenized_ds,
        )
        result = trainer.train()

    print(f"\n  Fallback Trainer SUCCESS")
    print(f"  train_loss  : {result.training_loss:.4f}")
    print(f"  train_steps : {result.global_step}")
else:
    banner("4. (Fallback not needed — SFTTrainer succeeded)")
    print("  SFTTrainer ran successfully; fallback Trainer not invoked.")

# ── 5. SFTConfig key fields explained ────────────────────────────────────────
banner("5. SFTConfig key fields (trl 1.5)")
print("""
SFTConfig inherits ALL TrainingArguments fields plus SFT-specific ones:

SFT-specific fields:
  dataset_text_field   : column name containing raw text (default 'text')
  max_length           : max sequence length (truncate/pad to this)
  packing              : True = pack short seqs into one context window
  packing_strategy     : 'greedy' (default) or 'ffd'
  completion_only_loss : compute loss only on completion tokens (not prompt)
  assistant_only_loss  : compute loss only on 'assistant' turns in chat format
  chat_template_path   : path or name of chat template to apply
  formatting_func      : callable(example) -> str (alternative to text column)
  dataset_kwargs       : extra kwargs for dataset processing

Key TrainingArguments fields:
  max_steps                : total training steps (overrides num_train_epochs)
  per_device_train_batch_size
  gradient_accumulation_steps
  learning_rate
  report_to='none'         : disable wandb/tensorboard logging
  save_strategy='no'       : skip checkpoint saving
  use_cpu=True             : force CPU (no GPU needed)
  logging_steps=1          : log loss every N steps

Docs: https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig
""")

# ── 6. With PEFT / LoRA ───────────────────────────────────────────────────────
banner("6. Snippet: SFTTrainer + LoRA (PEFT integration)")
print("""
# Pass peft_config to SFTTrainer; it wraps model automatically:
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

lora_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=['c_attn'],  # GPT-2; use ['q_proj','v_proj'] for LLaMA
    task_type=TaskType.CAUSAL_LM,
)

trainer = SFTTrainer(
    model=base_model,
    args=SFTConfig(output_dir='out', max_steps=100, report_to='none', use_cpu=True),
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_cfg,       # ← SFTTrainer calls get_peft_model() internally
)
trainer.train()
trainer.model.save_pretrained('./adapter/')  # saves only LoRA adapter (~KB)
""")

print("\nDONE — exit 0")
sys.exit(0)

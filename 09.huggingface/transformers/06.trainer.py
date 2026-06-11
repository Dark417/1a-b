"""
06.trainer.py — Fine-tuning with the Trainer API
=================================================
The Trainer handles:
  - The training loop (gradient accumulation, mixed precision, checkpointing)
  - Evaluation at regular intervals
  - Metric computation via compute_metrics callback
  - Distributed training (when using accelerate launch)

Official docs:
  https://huggingface.co/docs/transformers/main_classes/trainer
  https://huggingface.co/docs/transformers/training

This demo fine-tunes bert-tiny on a TINY in-memory classification dataset
(no internet needed for the data). The model download may need the network.
If download fails, a bert-tiny is built from scratch using BertConfig.

Dataset: 10 synthetic samples, 2 classes, binary sentiment.
Training: 2 epochs, batch_size=2, CPU, report_to="none".
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import tempfile

# ─── Imports ──────────────────────────────────────────────────────────────────
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BertConfig,
    BertForSequenceClassification,
    DataCollatorWithPadding,
)

# ─── 1. Build a tiny in-memory dataset (no download needed) ──────────────────
banner("1. Build tiny in-memory dataset")
# datasets.Dataset.from_dict() accepts a plain Python dict of lists.
# Each key is a column; all lists must have the same length.

raw_data = {
    "text": [
        "I love this product, it is fantastic!",
        "Terrible experience, very disappointing.",
        "Absolutely wonderful, highly recommend.",
        "Worst purchase I have ever made.",
        "Great quality and fast shipping.",
        "Broke after one day, waste of money.",
        "Amazing customer service and product.",
        "Do not buy this, complete garbage.",
        "Five stars, exceeded my expectations!",
        "One star, arrived damaged and useless.",
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],   # 1=positive, 0=negative
}

dataset = Dataset.from_dict(raw_data)
print(f"  Dataset size : {len(dataset)}")
print(f"  Columns      : {dataset.column_names}")
print(f"  Sample row   : {dataset[0]}")

# Split: 8 train / 2 eval
split = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]
print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

# ─── 2. Load tokenizer (with fallback) ────────────────────────────────────────
banner("2. Load tokenizer")

ok_tok, tok = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
if not ok_tok:
    note_skip(str(tok))
    # We'll use a simple word-splitting tokenizer as fallback
    # but the Trainer still needs a proper tokenizer for DataCollator.
    # So we fall back to BertTokenizer from a local config.
    from transformers import BertTokenizer
    # This won't work without files either; use a minimal hand-built approach
    tok = None

USE_LIVE_TOK = ok_tok and tok is not None

# ─── 3. Tokenize dataset ──────────────────────────────────────────────────────
banner("3. Tokenize dataset")

if USE_LIVE_TOK:
    def tokenize_fn(examples):
        return tok(
            examples["text"],
            truncation=True,
            max_length=64,
            padding=False,   # DataCollatorWithPadding handles dynamic padding
        )

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_eval  = eval_ds.map(tokenize_fn, batched=True)

    # Trainer expects a "labels" column (not "label")
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_eval  = tokenized_eval.rename_column("label", "labels")

    # Remove the raw text column (Trainer doesn't know what to do with it)
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_eval  = tokenized_eval.remove_columns(["text"])

    tokenized_train.set_format("torch")
    tokenized_eval.set_format("torch")

    print(f"  Tokenized train sample: {tokenized_train[0]}")
    data_collator = DataCollatorWithPadding(tok)
else:
    # Manual fallback: build integer-encoded tensors by hand
    # Simulate "tokenisation" with a simple char-hash vocab of 1000 tokens
    def manual_tokenize(text, max_length=32):
        """Hash each word to an int in [1, 999]."""
        words = text.lower().split()[:max_length - 2]
        ids = [101] + [abs(hash(w)) % 999 + 1 for w in words] + [102]
        ids = ids[:max_length]
        mask = [1] * len(ids)
        # Pad to max_length
        while len(ids) < max_length:
            ids.append(0)
            mask.append(0)
        return ids, mask

    def make_manual_dataset(data_dict):
        input_ids_list, mask_list, label_list = [], [], []
        for text, label in zip(data_dict["text"], data_dict["label"]):
            ids, mask = manual_tokenize(text)
            input_ids_list.append(ids)
            mask_list.append(mask)
            label_list.append(label)
        return Dataset.from_dict({
            "input_ids": input_ids_list,
            "attention_mask": mask_list,
            "labels": label_list,
        })

    tokenized_train = make_manual_dataset(train_ds.to_dict())
    tokenized_eval  = make_manual_dataset(eval_ds.to_dict())
    tokenized_train.set_format("torch")
    tokenized_eval.set_format("torch")
    data_collator = None  # default collator works for padded data
    print(f"  [fallback] Manual tokenized train sample keys: {list(tokenized_train[0].keys())}")

# ─── 4. Load / build model ────────────────────────────────────────────────────
banner("4. Load model (bert-tiny or local fallback)")

ok_mdl, model = safe(
    AutoModelForSequenceClassification.from_pretrained,
    "prajjwal1/bert-tiny",
    num_labels=2,
    ignore_mismatched_sizes=True,
)

if not ok_mdl:
    note_skip(str(model))
    # Build bert-tiny from scratch using BertConfig
    # (same architecture, random weights)
    vocab_size = 1000 if not USE_LIVE_TOK else 30522
    cfg = BertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        vocab_size=vocab_size,
        num_labels=2,
        max_position_embeddings=64,
    )
    model = BertForSequenceClassification(cfg)
    print(f"  [fallback] Built BertForSequenceClassification from scratch")
    print(f"             hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}")

param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {param_count:,}")

# ─── 5. compute_metrics ───────────────────────────────────────────────────────
banner("5. compute_metrics using numpy")

# Try to import evaluate; fall back to numpy
try:
    import evaluate as ev
    _metric = ev.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return _metric.compute(predictions=preds, references=labels)
    print("  Using evaluate library for accuracy")
except Exception as e:
    print(f"  evaluate not available ({e}), using numpy fallback")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float(np.mean(preds == labels))
        return {"accuracy": acc}

# ─── 6. TrainingArguments ─────────────────────────────────────────────────────
banner("6. TrainingArguments — all the knobs")

# Use a temp dir to avoid polluting the repo
output_dir = tempfile.mkdtemp(prefix="bert_tiny_ft_")
print(f"  Output dir: {output_dir}")

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,               # keep it short
    per_device_train_batch_size=2,    # tiny: 2 samples per step
    per_device_eval_batch_size=2,
    eval_strategy="epoch",            # evaluate at end of each epoch
    save_strategy="no",               # don't save checkpoints (speed)
    logging_steps=1,
    report_to="none",                 # no wandb / tensorboard
    seed=42,
    no_cuda=True,                     # force CPU
    dataloader_num_workers=0,         # single-process dataloader
)
print(f"  Epochs         : {training_args.num_train_epochs}")
print(f"  Train batch    : {training_args.per_device_train_batch_size}")
print(f"  Device         : {'cpu' if training_args.no_cuda else 'gpu'}")

# ─── 7. Create Trainer and train ─────────────────────────────────────────────
banner("7. Trainer — train + evaluate")

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)
if data_collator is not None:
    trainer_kwargs["data_collator"] = data_collator

trainer = Trainer(**trainer_kwargs)

print("  Starting training...")
train_result = trainer.train()

print(f"\n  Training complete!")
print(f"  Train runtime    : {train_result.metrics.get('train_runtime', 'N/A'):.2f}s")
print(f"  Train loss       : {train_result.metrics.get('train_loss', 'N/A'):.4f}")

# ─── 8. Evaluate ──────────────────────────────────────────────────────────────
banner("8. Evaluation on held-out set")

eval_results = trainer.evaluate()
print(f"  Eval results:")
for k, v in eval_results.items():
    print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

# ─── 9. Manual inference with trained model ──────────────────────────────────
banner("9. Manual inference after fine-tuning")

model.eval()
if USE_LIVE_TOK:
    test_texts = ["Absolutely love it!", "Complete disaster, avoid."]
    for text in test_texts:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = probs.argmax().item()
        print(f"  {text!r} → {'POSITIVE' if pred == 1 else 'NEGATIVE'} ({probs[pred]:.3f})")
else:
    for text in ["Absolutely love it!", "Complete disaster."]:
        ids, mask = manual_tokenize(text)
        dummy_inp = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([mask]),
        }
        with torch.no_grad():
            logits = model(**dummy_inp).logits
        pred = logits.argmax(dim=-1).item()
        print(f"  [fallback] {text!r} → label={pred}")

print("\n[DONE] 06.trainer.py complete")

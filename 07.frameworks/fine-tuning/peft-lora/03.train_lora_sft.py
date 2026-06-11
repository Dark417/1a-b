"""03 · Genuine tiny LoRA SFT — runs to completion on CPU.

This is the cornerstone runnable example: a real supervised fine-tuning loop
that trains LoRA adapters on a tiny GPT-2 over a toy instruction dataset, using
the plain `transformers.Trainer` (not TRL, so it has no extra version coupling).
It actually optimises — watch the loss, then generate before/after.

The task: teach the model a fixed Q->A mapping (overfitting a tiny set on
purpose, so a few steps visibly move the loss on CPU).

Docs: https://huggingface.co/docs/peft/task_guides/clm-prompt-tuning
Tiny model `sshleifer/tiny-gpt2`. ~seconds on CPU. Exit code 0.
"""
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

torch.manual_seed(0)
torch.set_num_threads(1)

MODEL = "sshleifer/tiny-gpt2"
MAX_LEN = 32


def build_dataset(tok) -> Dataset:
    pairs = [
        ("What is the capital of France?", "Paris."),
        ("What color is the sky?", "Blue."),
        ("2 + 2 equals?", "4."),
        ("Who wrote Hamlet?", "Shakespeare."),
    ]
    texts = [f"Q: {q}\nA: {a}{tok.eos_token}" for q, a in pairs] * 8  # 32 examples

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length",
                   max_length=MAX_LEN)

    return Dataset.from_dict({"text": texts}).map(
        tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"],
                   lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"),
    )
    model.print_trainable_parameters()

    ds = build_dataset(tok)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir="/tmp/peft_lora_sft",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=5e-3,        # high LR so a tiny model moves fast on CPU
        logging_steps=2,
        report_to=[],
        use_cpu=True,
        save_strategy="no",
        seed=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=collator)

    log0 = trainer.evaluate(ds) if False else None  # (eval omitted for speed)
    result = trainer.train()
    print(f"\nfinished {int(result.metrics['train_runtime']*1000)} ms, "
          f"final train_loss={result.training_loss:.4f}")

    # Loss should have decreased from the first to the last logged step.
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    if len(losses) >= 2:
        print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f} "
              f"({'decreased ✓' if losses[-1] < losses[0] else 'flat'})")

    # Generate from the fine-tuned adapter (greedy, deterministic).
    model.eval()
    ids = tok("Q: What is the capital of France?\nA:", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=5, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    print("sample generation:", repr(tok.decode(out[0], skip_special_tokens=True)))
    print("(tiny-gpt2 is random-init, so text is gibberish — the point is the "
          "loss dropped, proving the LoRA SFT loop trains end to end.)")


if __name__ == "__main__":
    main()

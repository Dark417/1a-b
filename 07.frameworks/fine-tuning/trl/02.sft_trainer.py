"""02 · SFTTrainer — a real tiny SFT run to completion.

`SFTTrainer` is TRL's supervised fine-tuning trainer. Hand it a model name and a
dataset with a `text` column and it tokenizes, batches, and trains. We overfit a
toy set so the loss visibly drops on CPU.

Docs: https://huggingface.co/docs/trl/sft_trainer
Tiny model `sshleifer/tiny-gpt2`. ~seconds on CPU. Exit code 0.
"""
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

torch.manual_seed(0)
torch.set_num_threads(1)

MODEL = "sshleifer/tiny-gpt2"


def main() -> None:
    # Prompt-completion shape; SFTTrainer joins and tokenizes automatically.
    rows = [
        {"prompt": "Q: capital of France?\nA:", "completion": " Paris."},
        {"prompt": "Q: 2+2?\nA:", "completion": " 4."},
        {"prompt": "Q: sky color?\nA:", "completion": " Blue."},
    ] * 8
    ds = Dataset.from_list(rows)

    cfg = SFTConfig(
        output_dir="/tmp/trl_sft",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        learning_rate=5e-3,
        logging_steps=4,
        max_length=32,
        report_to=[],
        use_cpu=True,
        save_strategy="no",
        seed=0,
    )
    trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=ds)

    result = trainer.train()
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"\nfinal train_loss={result.training_loss:.4f}")
    if len(losses) >= 2:
        print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f} "
              f"({'decreased ✓' if losses[-1] < losses[0] else 'flat'})")

    # Generate from the fine-tuned model.
    tok = trainer.processing_class
    ids = tok("Q: capital of France?\nA:", return_tensors="pt")
    with torch.no_grad():
        out = trainer.model.generate(**ids, max_new_tokens=4, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
    print("generation:", repr(tok.decode(out[0], skip_special_tokens=True)))
    print("(tiny random-init model -> gibberish text; the dropping loss is the "
          "proof the SFT loop works end to end.)")


if __name__ == "__main__":
    main()

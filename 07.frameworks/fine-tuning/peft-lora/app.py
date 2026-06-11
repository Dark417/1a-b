"""app.py · End-to-end LoRA workflow.

Ties the folder together: load base -> attach LoRA -> train a few steps ->
save adapter -> reload -> merge -> generate from the merged model. This is the
shape of a real fine-tuning pipeline, compressed to seconds on CPU.

Tiny model `sshleifer/tiny-gpt2`. CPU. Exit code 0.
"""
import os
import tempfile

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
from peft import LoraConfig, PeftModel, get_peft_model

torch.manual_seed(0)
torch.set_num_threads(1)

MODEL = "sshleifer/tiny-gpt2"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token

    # 1. base + LoRA
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL),
        LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"],
                   task_type="CAUSAL_LM"),
    )
    print("[1/5] attached LoRA"); model.print_trainable_parameters()

    # 2. tiny SFT
    texts = [f"Q: ping\nA: pong{tok.eos_token}"] * 16
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tok(b["text"], truncation=True, padding="max_length",
                      max_length=16),
        batched=True, remove_columns=["text"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/app_lora", num_train_epochs=3,
                               per_device_train_batch_size=8, learning_rate=5e-3,
                               logging_steps=5, report_to=[], use_cpu=True,
                               save_strategy="no", seed=0),
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    res = trainer.train()
    print(f"[2/5] trained, final loss={res.training_loss:.4f}")

    with tempfile.TemporaryDirectory() as d:
        # 3. save adapter only
        model.save_pretrained(d)
        print(f"[3/5] saved adapter ({len(os.listdir(d))} files, KB-sized)")

        # 4. reload onto a fresh base, then merge
        reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL), d)
        merged = reloaded.merge_and_unload()
        print(f"[4/5] reloaded + merged -> {type(merged).__name__}")

    # 5. generate from the merged (plain) model
    merged.eval()
    ids = tok("Q: ping\nA:", return_tensors="pt")
    with torch.no_grad():
        out = merged.generate(**ids, max_new_tokens=4, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    print("[5/5] generation:", repr(tok.decode(out[0], skip_special_tokens=True)))
    print("\npipeline complete: load -> LoRA -> train -> save -> merge -> serve ✓")


if __name__ == "__main__":
    main()

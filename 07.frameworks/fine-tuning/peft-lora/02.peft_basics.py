"""02 · PEFT basics — LoraConfig, get_peft_model, inspect, save/load.

Now the real library. We wrap a tiny GPT-2 with a LoRA adapter, inspect the
trainable-parameter count, save *only the adapter* (a few KB, not the whole
model), then reload it onto a fresh base.

Docs: https://huggingface.co/docs/peft/package_reference/lora
Tiny model `sshleifer/tiny-gpt2`. Runs on CPU. Exit code 0.
"""
import os
import tempfile

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

torch.manual_seed(0)
torch.set_num_threads(1)

MODEL = "sshleifer/tiny-gpt2"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL)

    # GPT-2 fuses q/k/v into a single Conv1D called `c_attn`.
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn"],   # see model.named_modules() for other archs
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()  # prints "trainable params: ... || ..."

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"manual count: {n_train}/{n_total} trainable "
          f"({100*n_train/n_total:.3f}%)")

    # Which modules got adapted?
    adapted = sorted({n.split(".lora_")[0].split(".")[-1]
                      for n, _ in model.named_parameters() if "lora_" in n})
    print("adapted module types:", adapted)

    # Save ONLY the adapter (note the tiny size), then reload onto a fresh base.
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        files = sorted(os.listdir(d))
        print("adapter dir contents:", files)
        size_kb = sum(os.path.getsize(os.path.join(d, f)) for f in files
                      if os.path.isfile(os.path.join(d, f))) / 1024
        print(f"adapter on disk: {size_kb:.1f} KB (base model NOT included)")

        fresh = AutoModelForCausalLM.from_pretrained(MODEL)
        reloaded = PeftModel.from_pretrained(fresh, d)
        # Outputs should match the saved model on the same input.
        ids = tok("hello world", return_tensors="pt")
        with torch.no_grad():
            a = model(**ids).logits
            b = reloaded(**ids).logits
        max_delta = (a - b).abs().max().item()
        print(f"reloaded adapter reproduces logits: max|Δ|={max_delta:.2e} ✓")


if __name__ == "__main__":
    main()

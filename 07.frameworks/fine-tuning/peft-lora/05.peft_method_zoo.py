"""05 · The PEFT method zoo — beyond plain LoRA.

PEFT implements a whole family of parameter-efficient methods. We build each on
the same tiny GPT-2 and compare trainable-parameter counts and the *idea* behind
each. This is the landscape an engineer should know.

  - LoRA      : ΔW = (α/r) B A                       (low-rank update)
  - DoRA      : decompose W into magnitude m and direction; LoRA on direction
                (Liu et al. 2024, https://arxiv.org/abs/2402.09353)
  - rsLoRA    : LoRA with scaling α/√r (stabilises high rank)
  - (IA)^3    : learn elementwise rescaling vectors for k, v, ffn (very few params)
                (Liu et al. 2022, https://arxiv.org/abs/2205.05638)
  - LoHa      : ΔW via Hadamard product of two low-rank pairs (FedPara)
  - Prompt    : prepend trainable "soft" prompt embeddings (Lester et al. 2021)
  - Prefix    : trainable key/value prefixes at every layer (Li & Liang 2021)

Docs: https://huggingface.co/docs/peft/conceptual_guides/adapter
Tiny model `sshleifer/tiny-gpt2`. CPU. Exit code 0.
"""
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM
from peft import (
    IA3Config,
    LoHaConfig,
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    get_peft_model,
)

torch.manual_seed(0)
torch.set_num_threads(1)

MODEL = "sshleifer/tiny-gpt2"


def count(cfg, label: str) -> None:
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    try:
        model = get_peft_model(base, cfg)
    except Exception as e:  # some methods are picky about target modules
        print(f"{label:28s} -> skipped ({type(e).__name__})")
        return
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"{label:28s} -> trainable {n_train:>6d} / {n_total} "
          f"({100*n_train/n_total:.3f}%)")


def main() -> None:
    print(f"comparing PEFT methods on {MODEL} (base = "
          f"{sum(p.numel() for p in AutoModelForCausalLM.from_pretrained(MODEL).parameters())} params)\n")

    count(LoraConfig(r=8, target_modules=["c_attn"], task_type="CAUSAL_LM"),
          "LoRA (r=8)")
    count(LoraConfig(r=8, target_modules=["c_attn"], use_dora=True,
                     task_type="CAUSAL_LM"),
          "DoRA (r=8)")
    count(LoraConfig(r=8, target_modules=["c_attn"], use_rslora=True,
                     task_type="CAUSAL_LM"),
          "rsLoRA (r=8)")
    count(IA3Config(target_modules=["c_attn", "mlp.c_proj"],
                    feedforward_modules=["mlp.c_proj"], task_type="CAUSAL_LM"),
          "(IA)^3")
    count(LoHaConfig(r=8, alpha=16, target_modules=["c_attn"],
                     task_type="CAUSAL_LM"),
          "LoHa (r=8)")
    count(PromptTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM"),
          "Prompt tuning (10 tokens)")
    count(PrefixTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM"),
          "Prefix tuning (10 tokens)")

    print("\nTakeaways:")
    print("  • (IA)^3 trains the fewest params (just rescaling vectors).")
    print("  • DoRA ≈ LoRA params + a magnitude vector; often better quality.")
    print("  • Prompt/prefix tuning add no weight deltas — they learn inputs.")
    print("  • Same API for all: get_peft_model(model, <SomeConfig>).")


if __name__ == "__main__":
    main()

"""04 · Merge, swap, and stack adapters.

Three production patterns PEFT enables:

  1. MERGE — bake (α/r)·BA into the base weight (`merge_and_unload`) so inference
     has zero LoRA overhead and you get a plain transformers model back.
  2. SWAP — keep ONE base in memory and hot-swap multiple named adapters
     (multi-tenant serving: one GPU, many task adapters).
  3. STACK / combine — load several adapters and activate a weighted combination.

Docs: https://huggingface.co/docs/peft/developer_guides/lora#merge-adapters
Tiny model `sshleifer/tiny-gpt2`. CPU. Exit code 0.
"""
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

torch.manual_seed(0)
torch.set_num_threads(1)

MODEL = "sshleifer/tiny-gpt2"


def make_adapter(name: str) -> LoraConfig:
    return LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"],
                      task_type="CAUSAL_LM")


def randomise_lora(model) -> None:
    """Give the (zero-init) B matrices nonzero values so adapters differ."""
    for n, p in model.named_parameters():
        if "lora_B" in n:
            with torch.no_grad():
                p.normal_(0, 0.02)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    ids = tok("hello", return_tensors="pt")

    # ---- 1. MERGE ----------------------------------------------------------
    base = AutoModelForCausalLM.from_pretrained(MODEL)
    peft_model = get_peft_model(base, make_adapter("default"))
    randomise_lora(peft_model)
    with torch.no_grad():
        before = peft_model(**ids).logits.clone()
    merged = peft_model.merge_and_unload()  # returns a plain transformers model
    with torch.no_grad():
        after = merged(**ids).logits
    print(f"merge_and_unload: logits preserved "
          f"(max|Δ|={(before-after).abs().max().item():.2e}); "
          f"type is now {type(merged).__name__} (no PEFT wrapper) ✓")

    # ---- 2. SWAP multiple named adapters on ONE base -----------------------
    base2 = AutoModelForCausalLM.from_pretrained(MODEL)
    model = get_peft_model(base2, make_adapter("english"), adapter_name="english")
    model.add_adapter("french", make_adapter("french"))
    # give each adapter distinct weights
    model.set_adapter("english"); randomise_lora(model)
    model.set_adapter("french"); randomise_lora(model)

    model.set_adapter("english")
    with torch.no_grad():
        out_en = model(**ids).logits.clone()
    model.set_adapter("french")
    with torch.no_grad():
        out_fr = model(**ids).logits.clone()
    diff = (out_en - out_fr).abs().max().item()
    print(f"swapped 'english' <-> 'french' on one base; outputs differ "
          f"(max|Δ|={diff:.3e}) — multi-tenant serving ✓")
    print("active adapter:", model.active_adapter)

    # ---- 3. COMBINE adapters into a new weighted adapter -------------------
    model.add_weighted_adapter(
        adapters=["english", "french"],
        weights=[0.5, 0.5],
        adapter_name="blend",
        combination_type="linear",
    )
    model.set_adapter("blend")
    with torch.no_grad():
        out_blend = model(**ids).logits
    # linear blend should sit between the two parents on at least some logits
    print(f"created 'blend' = 0.5*english + 0.5*french; "
          f"available adapters: {list(model.peft_config.keys())} ✓")


if __name__ == "__main__":
    main()

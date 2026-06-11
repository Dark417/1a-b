# PEFT + LoRA — the adapter library

[PEFT](https://huggingface.co/docs/peft) ("Parameter-Efficient Fine-Tuning") is
Hugging Face's library of **adapter methods**. It wraps any `transformers` model
so that only a tiny set of new parameters trains while the base stays frozen.
LoRA is the headline method; PEFT also implements DoRA, AdaLoRA, (IA)³,
prompt/prefix tuning, LoHa/LoKr, VeRA and more.

> The basics of PEFT live in
> [`09.huggingface/peft/`](../../../09.huggingface/peft/). This folder is the
> **deep dive**: the LoRA math made concrete, the full method zoo, adapter
> merging/swapping/stacking, and a genuinely runnable tiny LoRA SFT on CPU.

## Why PEFT exists (the mental model)

A `transformers` model is a tree of `nn.Linear`/`Conv1D` layers. PEFT
**surgically replaces** chosen layers with a wrapper that keeps the original
`weight` frozen and adds a trainable low-rank branch:

```
        ┌──────────── frozen ────────────┐
  x ───►│  h = W0 · x                     │──► h
        └─────────────────────────────────┘
              │ (same x)
              ▼   trainable, init B=0 so ΔW=0 at step 0
        ┌─ A (r×k) ─┐   ┌─ B (d×r) ─┐
  x ───►│  z = A·x  │──►│ Δ = B·z   │──► (α/r)·Δ  ──► h += (α/r)·Δ
        └───────────┘   └───────────┘
```

`get_peft_model(model, LoraConfig(...))` walks the module tree, finds the
modules named in `target_modules`, and swaps each for a `lora.Linear` that owns
`lora_A`, `lora_B` and the frozen base. Forward becomes
`W0·x + (α/r)·B·A·x`. Backprop only touches `A`, `B`.

### The LoRA equation

$$ W = W_0 + \frac{\alpha}{r} B A,\quad B\!\in\!\mathbb{R}^{d\times r},\,
   A\!\in\!\mathbb{R}^{r\times k},\, r \ll \min(d,k),\quad B=\mathbf{0}\text{ at init.} $$

Trainable params per adapted layer: $r(d+k)$ instead of $dk$. The scale
$\alpha/r$ lets you change rank without re-tuning the learning rate.

## Install

```bash
pip install -r requirements.txt
```

## Config knobs — `LoraConfig`

| Knob | Meaning | Typical |
|---|---|---|
| `r` | rank of the update | 8–64 (higher = more capacity + params) |
| `lora_alpha` | scale numerator; effective scale `alpha/r` | often `2*r` |
| `target_modules` | which layers get adapters | attn proj (`q_proj,v_proj`, …) or `"all-linear"` |
| `lora_dropout` | dropout on the LoRA input | 0.0–0.1 |
| `bias` | train biases? `none`/`all`/`lora_only` | `none` |
| `use_dora` | enable **DoRA** (magnitude+direction decomp) | `False` |
| `use_rslora` | rank-stabilised scaling `alpha/√r` | `False` |
| `task_type` | `CAUSAL_LM`, `SEQ_CLS`, `SEQ_2_SEQ_LM`, … | match your head |
| `modules_to_save` | extra modules to fully train (e.g. new `lm_head`) | e.g. embeddings when adding tokens |

**Picking `target_modules`:** more modules (e.g. `"all-linear"`, or adding the
MLP `gate/up/down_proj`) = more capacity and usually better quality, at more
params. Attention-only is the classic minimal choice.

## The files

| File | Feature | Runs on CPU |
|---|---|---|
| `01.lora_from_scratch.py` | LoRA in ~40 lines of pure PyTorch — see the math run | **live** |
| `02.peft_basics.py` | `LoraConfig` + `get_peft_model`, inspect trainable params, save/load adapter | **live** |
| `03.train_lora_sft.py` | **genuine tiny LoRA SFT to completion** (Trainer + PEFT) | **live** |
| `04.merge_and_swap.py` | merge adapter into base; swap/stack multiple adapters | **live** |
| `05.peft_method_zoo.py` | DoRA, (IA)³, prompt/prefix tuning, LoHa — build each, count params | **live** |
| `app.py` | end-to-end: train a LoRA, save it, reload, merge, generate | **live** |

Run any of them:

```bash
python 03.train_lora_sft.py
```

## Gotchas

- **`target_modules` names must match your architecture.** GPT-2 uses fused
  `c_attn` (a `Conv1D`); LLaMA uses `q_proj/k_proj/v_proj/o_proj`. Wrong names →
  "Target modules not found". Use `model.named_modules()` or `"all-linear"`.
- **`Conv1D` warning.** For GPT-2, PEFT flips `fan_in_fan_out=True` automatically
  and warns — harmless.
- **Merging is lossy in low precision.** `merge_and_unload()` bakes `α/r·BA` into
  the base; if the base is 4-bit you must dequantise first (you can't merge into
  a 4-bit tensor). Merge in fp16/bf16.
- **Saving.** `model.save_pretrained(dir)` writes *only* the adapter (a few MB),
  not the base. To share a standalone model, merge then save.
- **Set `pad_token`.** GPT-2 has no pad token; set `tokenizer.pad_token =
  tokenizer.eos_token` before batching or collation fails.
- **`modules_to_save`** is required if you resize embeddings / add tokens, else
  the new rows never train.

## References
- LoRA — Hu et al. 2021, https://arxiv.org/abs/2106.09685
- DoRA — Liu et al. 2024, https://arxiv.org/abs/2402.09353
- PEFT docs — https://huggingface.co/docs/peft
- PEFT conceptual guides — https://huggingface.co/docs/peft/conceptual_guides/adapter

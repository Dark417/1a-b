# TRL — Transformer Reinforcement Learning (SFT / DPO / GRPO)

[TRL](https://huggingface.co/docs/trl) is Hugging Face's library of **post-training
trainers**. It wraps the gnarly bookkeeping of supervised fine-tuning and
preference/RL objectives into a handful of `Trainer` subclasses that share the
`transformers.Trainer` API: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`,
`PPOTrainer`, `RewardTrainer`, `KTOTrainer`, `ORPOTrainer`.

> **Cross-link.** The *basics* of TRL — what `SFTTrainer` is, the chat-template
> dataset format — live in [`09.huggingface/trl/`](../../../09.huggingface/trl/).
> This folder is the **deep dive into the objectives**: the math of SFT vs DPO vs
> GRPO, when to use each, the full config surface, and combining TRL with PEFT.
> It also pairs with [`../peft-lora/`](../peft-lora/) (adapters) and
> [`../qlora/`](../qlora/) (4-bit).

## Where TRL sits

```
  raw text ──► [SFT] ──► instruct model ──► [DPO/ORPO/KTO] ──► aligned model
                                        └──► [GRPO/PPO + reward] ──► reasoning/RLHF
```

Most modern recipes are: **SFT first** (teach the format/skill), then a
**preference stage** (DPO/ORPO) or **RL stage** (GRPO) to align behaviour.

## The objectives, briefly

**SFT** — supervised fine-tuning. Standard next-token cross-entropy on
target completions. Optionally mask the *prompt* tokens so only the
*completion* contributes to the loss (`completion_only_loss` / a collator).

$$ \mathcal{L}_{\text{SFT}} = -\sum_{t} \log p_\theta(y_t \mid y_{<t}, x). $$

**DPO** — Direct Preference Optimization. Given pairs *(chosen, rejected)*, train
the policy to prefer chosen *without* a separate reward model, using a reference
model $\pi_{\text{ref}}$ (the frozen SFT model):

$$ \mathcal{L}_{\text{DPO}} = -\log \sigma\!\Big(
   \beta \log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
   - \beta \log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \Big). $$

$\beta$ controls how far the policy may drift from the reference.
([DPO paper](https://arxiv.org/abs/2305.18290))

**ORPO** — odds-ratio preference optimization. Folds preference into the SFT loss
so you need **no reference model and no separate SFT stage**.
([ORPO](https://arxiv.org/abs/2403.07691))

**KTO** — Kahneman-Tversky Optimization. Learns from **unpaired** binary
good/bad labels (cheaper data than pairs). ([KTO](https://arxiv.org/abs/2402.01306))

**GRPO** — Group Relative Policy Optimization (the DeepSeek-R1 reasoning recipe).
Sample a *group* of completions per prompt, score each with a reward function,
and push up above-average completions — no value/critic network.
([DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300))

## Install

```bash
pip install -r requirements.txt
```

## Config knobs — `SFTConfig` (subclasses `TrainingArguments`)

| Knob | Meaning |
|---|---|
| `max_length` | truncate/pack sequences to this length |
| `packing` | concatenate short examples to fill `max_length` (throughput) |
| `completion_only_loss` | mask prompt tokens from the loss |
| `dataset_text_field` | column holding raw text (default `"text"`) |
| `assistant_only_loss` | with chat templates, train on assistant turns only |
| `learning_rate`, `num_train_epochs`, `per_device_train_batch_size` | inherited from `TrainingArguments` |
| `peft_config` (trainer arg) | pass a `LoraConfig` to train LoRA instead of full FT |

`DPOConfig` adds `beta`, `loss_type` (`sigmoid`/`ipo`/`hinge`),
`reference_free`. `GRPOConfig` adds `num_generations`, `reward_funcs` (callables
or reward models), generation params.

## The files

| File | Objective | Runs on CPU |
|---|---|---|
| `01.datasets_and_chat.py` | dataset formats: text / prompt-completion / chat; apply chat templates | **live** |
| `02.sft_trainer.py` | **real tiny SFT to completion** with `SFTTrainer` (+ optional PEFT) | **live** |
| `03.sft_with_peft.py` | LoRA SFT via TRL — pass a `LoraConfig`, train, save adapter | **live** |
| `04.dpo_trainer.py` | DPO on a toy preference set; inspect the reward margin | **live** |
| `05.grpo_overview.py` | GRPO explained with a runnable reward function on tiny generations | **live** |
| `app.py` | SFT then DPO on the same tiny model (the standard two-stage recipe) | **live** |

```bash
python 02.sft_trainer.py
```

## Gotchas

- **Version coupling.** TRL tracks `transformers`/`accelerate` closely and the
  `SFTConfig` surface changes between releases (e.g. `max_seq_length` was renamed
  to `max_length`; `dataset_text_field` moved into the config). Pin versions per
  `requirements.txt` and check the changelog when upgrading.
- **DPO needs a reference model** unless `reference_free=True`; by default TRL
  makes one by copying the policy — that doubles memory. With PEFT, TRL uses the
  base (adapters disabled) as the reference for free.
- **Chat templates.** For chat data, the tokenizer's `chat_template` decides the
  exact string; a wrong/absent template silently mistrains. Use
  `apply_chat_template` and verify the rendered text.
- **`packing=True`** boosts throughput but mixes documents in one sequence; keep
  EOS tokens so the model still learns to stop.
- **CPU is for smoke tests only.** Real runs need a GPU; tiny-gpt2 here exists to
  prove the loop trains, not to produce a good model.

## References
- TRL docs — https://huggingface.co/docs/trl
- DPO — https://arxiv.org/abs/2305.18290
- ORPO — https://arxiv.org/abs/2403.07691
- KTO — https://arxiv.org/abs/2402.01306
- GRPO / DeepSeekMath — https://arxiv.org/abs/2402.03300

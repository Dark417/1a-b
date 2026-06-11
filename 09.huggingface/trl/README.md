# TRL — Transformer Reinforcement Learning (University-Grade Explainer)

> Official docs: https://huggingface.co/docs/trl/index  
> GitHub: https://github.com/huggingface/trl  
> Version note: This folder targets **trl 1.5** (installed). TRL's API evolves quickly; always pin the version.  

---

## 1. What is TRL?

TRL is HuggingFace's alignment-training library that sits on top of `transformers` + `peft` + `accelerate`. It implements the full RLHF (Reinforcement Learning from Human Feedback) pipeline and modern alternatives:

```
Stage 1: Supervised Fine-Tuning (SFT)    ←  SFTTrainer
Stage 2: Reward Modelling (optional)     ←  RewardTrainer
Stage 3: RL / Preference Optimisation   ←  PPOTrainer / GRPOTrainer / DPOTrainer
```

Modern alignment stacks often skip Stage 3 with PPO and go directly to:
- **DPO** (Direct Preference Optimisation) — no reward model needed
- **GRPO** (Group Relative Policy Optimisation) — used in DeepSeek-R1

---

## 2. SFTTrainer — Supervised Fine-Tuning

### 2.1 Purpose

Supervised fine-tuning trains the model to produce desired outputs given prompts — the "imitation learning" stage. It is usually the first step before DPO/PPO.

### 2.2 Dataset Formats

SFTTrainer accepts multiple dataset formats:

**Plain text column** (simplest):
```python
{"text": "### Instruction: Summarise this. ### Response: ..."}
```

**Prompt + completion** (handled via `formatting_func`):
```python
{"prompt": "Translate to French: hello", "completion": "bonjour"}
```

**Chat template** (messages list):
```python
{"messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."}
]}
```

### 2.3 Key SFTConfig Parameters

```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="./sft-output",
    max_steps=100,                    # or num_train_epochs=3
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    gradient_accumulation_steps=4,
    max_length=512,                   # truncate/pad sequences to this length
    dataset_text_field="text",        # which column to use as text
    packing=True,                     # pack short sequences into one context window
    report_to="none",
)
```

**Packing** (`packing=True`): concatenates multiple short examples with EOS tokens and splits into fixed-length chunks. Dramatically improves GPU utilisation when examples are shorter than `max_seq_length`.

### 2.4 With PEFT (LoRA)

```python
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
                          task_type="CAUSAL_LM")

trainer = SFTTrainer(
    model=base_model,
    args=SFTConfig(output_dir="out", max_steps=100, report_to="none"),
    train_dataset=dataset,
    peft_config=lora_config,      # ← SFTTrainer wraps model automatically
)
trainer.train()
trainer.model.save_pretrained("./adapter/")  # saves LoRA adapter only
```

### 2.5 Chat Templates

```python
config = SFTConfig(
    ...,
    chat_template_path="chatml",    # apply ChatML template to messages column
)
```

---

## 3. DPOTrainer — Direct Preference Optimisation

### 3.1 Motivation: Why not PPO?

PPO (Proximal Policy Optimisation) for RLHF requires:
1. A separately trained reward model R(x, y)
2. Four models in memory simultaneously: policy π, reference π_ref, reward R, value V
3. Complex RL training loop (rollouts, GAE, clipping, KL penalty)
4. Very sensitive hyperparameters

**DPO** (Rafailov et al. 2023 — https://arxiv.org/abs/2305.18290) derives a *direct* supervised objective from the RLHF optimisation problem. No reward model, no rollouts.

### 3.2 DPO Derivation (Sketch)

Starting from the RLHF objective:
```
max_π  E_{x~D, y~π} [R(x,y)] − β·KL[π(·|x) || π_ref(·|x)]
```

The optimal policy (in closed form) is:
```
π*(y|x) ∝ π_ref(y|x) · exp(R(x,y) / β)
```

This lets us express R(x,y) as a function of the policy:
```
R(x,y) = β · log[π*(y|x) / π_ref(y|x)] + β · log Z(x)
```

Plugging into the Bradley-Terry preference model P(y_w ≻ y_l) = σ(R(y_w) - R(y_l)):

### DPO Loss (the key formula):

```
L_DPO(π_θ; π_ref) = −E_{(x, y_w, y_l) ~ D} [
    log σ(
        β · log[π_θ(y_w|x) / π_ref(y_w|x)]
      − β · log[π_θ(y_l|x) / π_ref(y_l|x)]
    )
]
```

Where:
- `y_w` = chosen (preferred) response
- `y_l` = rejected response
- `β` = temperature controlling deviation from reference
- `σ` = sigmoid function
- `π_ref` = frozen reference model (the SFT checkpoint)

**Intuition**: The loss encourages the policy to assign *relatively higher* probability to the chosen response vs the rejected one, compared to what the reference model would assign. It's a ranking loss expressed in log-probability space.

### 3.3 Implementation with DPOTrainer

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    output_dir="./dpo-output",
    max_steps=100,
    per_device_train_batch_size=2,
    beta=0.1,                    # KL penalty coefficient
    report_to="none",
)

# Dataset must have columns: prompt, chosen, rejected
dataset = Dataset.from_dict({
    "prompt":   ["Explain gravity:"] * N,
    "chosen":   ["Gravity is..."]  * N,
    "rejected": ["I don't know."]  * N,
})

trainer = DPOTrainer(
    model=sft_model,              # the policy being trained
    ref_model=ref_model,          # frozen SFT checkpoint
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

If `ref_model=None`, DPOTrainer creates a copy of `model` and freezes it (appropriate when using PEFT — the base is the ref, adapters are the policy).

### 3.4 DPO vs RLHF/PPO Comparison

| Aspect | PPO (RLHF) | DPO |
|---|---|---|
| Reward model | Required (separate training) | Not needed |
| Models in memory | 4 (policy, ref, reward, value) | 2 (policy + ref) |
| RL rollouts | Yes (slow, noisy) | No |
| Objective | RL (high variance) | Supervised (stable) |
| Hyperparameters | Many (clip, GAE-λ, etc.) | Few (β, lr) |
| Quality | Strong but complex | Competitive, simpler |
| Best for | Online/iterative feedback | Offline preference data |

---

## 4. GRPO — Group Relative Policy Optimisation

Used in DeepSeek-R1. Instead of a learned reward model, uses a verifiable
reward function (e.g., math answer correctness):

```
A_i = (r_i − mean(r_group)) / std(r_group)   # advantage via group normalisation
L_GRPO = −E[A_i · log π_θ(y_i|x)] + β·KL[π_θ||π_ref]
```

```python
from trl import GRPOTrainer, GRPOConfig

def reward_fn(completions, **kwargs):
    return [1.0 if "correct" in c else 0.0 for c in completions]

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=GRPOConfig(output_dir="out", max_steps=50),
    train_dataset=dataset,
)
```

---

## 5. RewardTrainer

Trains a reward model from preference pairs for use with PPO:

```python
from trl import RewardTrainer, RewardConfig

# Dataset: {"input_ids_chosen": ..., "input_ids_rejected": ...}
trainer = RewardTrainer(
    model=model,              # must have a scalar reward head
    args=RewardConfig(output_dir="out"),
    train_dataset=dataset,
)
```

---

## 6. API Evolution Warning

TRL moves fast. In trl 1.x:
- `SFTTrainer` no longer takes `tokenizer=` — use `processing_class=`
- `SFTConfig` absorbs many `TrainingArguments` fields
- `peft_config` is passed directly to `SFTTrainer`/`DPOTrainer`
- Some older tutorials use `SFTTrainer(tokenizer=..., dataset_text_field=...)`
  which may error on trl 1.5 — check the changelog.

Always do:
```python
import trl; print(trl.__version__)  # verify: 1.5.x
```

---

*Generated for trl 1.5.1 — docs: https://huggingface.co/docs/trl*

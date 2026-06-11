# fine-tuning — adapting pretrained LLMs to your task

Pretraining gives a model broad competence; **fine-tuning** specialises it —
teaching a format (chat, JSON), a domain (legal, medical), a behaviour (refuse
X, prefer Y), or a style. This area covers the *toolchain* for doing that
efficiently, from the math of **LoRA** to config-driven trainers like Axolotl
and LLaMA-Factory.

> **Cross-link.** [`../../../09.huggingface/peft/`](../../../09.huggingface/peft/)
> and [`../../../09.huggingface/trl/`](../../../09.huggingface/trl/) cover the
> *basics* of PEFT and TRL (the SFT/DPO trainer abstractions). This area goes
> **deeper and broader**: the LoRA/QLoRA math, the full method landscape, and the
> config-driven ecosystem (Unsloth, Axolotl, torchtune, LLaMA-Factory) that wraps
> those libraries for production runs. We avoid duplicating the 09 basics and
> cross-reference them instead.

## The core idea: don't move all the weights

A 7B model has 7e9 parameters. Full fine-tuning updates all of them — expensive
(needs optimizer state ≈ 2× params in fp32) and you get a full-size checkpoint
per task. **Parameter-Efficient Fine-Tuning (PEFT)** freezes the base model and
trains a tiny number of new parameters.

### LoRA — Low-Rank Adaptation

For a frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA learns a
low-rank update instead of editing $W_0$:

$$ W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A,\qquad
   B \in \mathbb{R}^{d\times r},\; A \in \mathbb{R}^{r\times k},\; r \ll \min(d,k). $$

Only $A$ and $B$ train. With $r=8$, $d=k=4096$ you train $2 \cdot 4096 \cdot 8 =
65{,}536$ params instead of $16.7$M per matrix — a ~250× reduction. $A$ is init
to a small Gaussian, $B$ to zero, so $\Delta W = 0$ at step 0 (training starts
from the base model). The scale $\alpha/r$ decouples learning-rate-like scaling
from the rank. ([LoRA paper, Hu et al. 2021](https://arxiv.org/abs/2106.09685))

At inference you can **merge** $W_0 + \frac{\alpha}{r}BA$ back into one matrix —
zero added latency. Adapters are also *swappable*: one base model, many task
adapters loaded on demand.

### QLoRA — quantise the frozen base, train LoRA on top

QLoRA makes LoRA fit a single consumer GPU by storing the *frozen* base weights
in **4-bit NormalFloat (NF4)** and back-propagating through them into fp16/bf16
LoRA adapters. Three tricks: **NF4** (an info-theoretically optimal 4-bit type
for normally-distributed weights), **double quantisation** (quantise the
quantisation constants), and **paged optimizers** (offload optimizer state to
CPU on memory spikes). Result: fine-tune a 65B model on one 48GB GPU with no
quality loss vs 16-bit LoRA.
([QLoRA paper, Dettmers et al. 2023](https://arxiv.org/abs/2305.14314))

### The PEFT method landscape (covered in `peft-lora/`)

LoRA is one of many adapters PEFT supports: **DoRA** (weight-decomposed LoRA),
**LoHa/LoKr** (Hadamard/Kronecker products), **AdaLoRA** (adaptive rank
allocation), **(IA)³** (learned rescaling vectors), **prompt/prefix/P-tuning**
(learn soft tokens), **VeRA** (shared random bases + tiny scaling). We tour them.

## The training objectives

| Objective | What it does | Tool |
|---|---|---|
| **SFT** (supervised fine-tuning) | imitate target completions (next-token CE) | TRL `SFTTrainer`, all configs |
| **DPO** | prefer chosen over rejected without a reward model | TRL `DPOTrainer` |
| **ORPO / KTO / SimPO** | preference tuning variants (odds-ratio, unpaired, …) | TRL |
| **GRPO / PPO** | RL with a reward (reasoning, RLHF) | TRL |
| **continued pretraining** | more next-token on raw domain text | all |

## Which tool? — the ecosystem map

These tools stack: the bottom is libraries, the top is config-driven launchers
that *use* those libraries.

```
config launchers   Axolotl · LLaMA-Factory · torchtune     (YAML/CLI -> training)
speed layer        Unsloth                                  (fused kernels, 2x faster)
trainer abstractions  TRL (SFT/DPO/GRPO) · transformers Trainer
adapter library    PEFT (LoRA/QLoRA/DoRA/...)
quantisation       bitsandbytes (NF4/int8)
base               transformers + accelerate + datasets + PyTorch
```

| Tool | What it is | Interface | When to reach for it |
|---|---|---|---|
| **PEFT** | the adapter library | Python | you want control / custom loops |
| **TRL** | SFT/DPO/GRPO trainers | Python | standard objectives, less boilerplate |
| **QLoRA** | technique (PEFT+bnb) | Python | big model, one GPU |
| **Unsloth** | optimized kernels | Python (drop-in) | 2× faster, less VRAM, single GPU |
| **Axolotl** | YAML-driven trainer | YAML config | reproducible runs, many recipes |
| **torchtune** | PyTorch-native recipes | config + CLI | hackable recipes, no HF Trainer |
| **LLaMA-Factory** | broad trainer + WebUI | YAML / GUI | 100+ models, SFT→DPO→PPO menu |

## Offline / CPU contract

Real fine-tuning needs a GPU, but **every `.py` here exits 0 on CPU** using tiny
models (`sshleifer/tiny-gpt2`, `hf-internal-testing/tiny-random-*`):

- `peft-lora/03.train_lora_sft.py` and `trl/02.sft_trainer.py` run a **genuine
  tiny LoRA SFT to completion on CPU**.
- QLoRA / Unsloth need a CUDA GPU + bitsandbytes: code is correct and idiomatic
  but **guarded** behind `torch.cuda.is_available()` with a CPU-LoRA fallback.
- Axolotl / torchtune / LLaMA-Factory are **config-driven**: we ship real configs
  and a Python walkthrough that *parses and explains* them (runs on CPU); the
  actual `accelerate launch` is shown but guarded.

## Sub-folders

| Folder | Topic | Live on CPU? |
|---|---|---|
| [`peft-lora/`](peft-lora/) | LoRA + the PEFT method zoo, merge/swap | **yes (real tiny SFT)** |
| [`qlora/`](qlora/) | 4-bit NF4 + LoRA, bitsandbytes | guarded (GPU), CPU fallback |
| [`trl/`](trl/) | SFT/DPO/GRPO trainers (deep dive) | **yes (real tiny SFT)** |
| [`unsloth/`](unsloth/) | fused-kernel speedups | guarded (GPU), explains |
| [`axolotl/`](axolotl/) | YAML configs, walkthrough | config walkthrough (CPU) |
| [`torchtune/`](torchtune/) | recipes + configs | config walkthrough (CPU) |
| [`llama-factory/`](llama-factory/) | config/WebUI trainer | config walkthrough (CPU) |

## References
- LoRA — https://arxiv.org/abs/2106.09685
- QLoRA — https://arxiv.org/abs/2305.14314
- PEFT docs — https://huggingface.co/docs/peft
- TRL docs — https://huggingface.co/docs/trl
- Unsloth — https://docs.unsloth.ai/
- Axolotl — https://docs.axolotl.ai/
- torchtune — https://docs.pytorch.org/torchtune/
- LLaMA-Factory — https://github.com/hiyouga/LLaMA-Factory

# PEFT — Parameter-Efficient Fine-Tuning (University-Grade Explainer)

> Official docs: https://huggingface.co/docs/peft/index  
> Paper index: https://huggingface.co/docs/peft/conceptual_guides/lora  
> GitHub: https://github.com/huggingface/peft  

---

## 1. Why PEFT?

Modern LLMs have billions of parameters. Full fine-tuning requires:
- Storing and updating every parameter → huge GPU memory
- Separate checkpoint per downstream task (7B × 3 tasks = 21B params on disk)
- Slow convergence on small datasets (catastrophic forgetting risk)

**PEFT principle**: *Freeze* the pre-trained base weights and *train only a tiny set of new parameters* that adapt the model to the new task.

```
Full fine-tune:  update ALL W (e.g. 7B params)
PEFT / LoRA:     freeze W, train  ΔW  (e.g. 0.1% of params = 7M)
```

Benefits:
- 10–1000× fewer trainable parameters
- Much lower GPU memory (no optimizer state for frozen params)
- Adapter files are tiny (~MB not GB) → easy multi-task switching
- Often matches full fine-tune quality on downstream tasks

---

## 2. Methods Catalogue

### 2.1 LoRA — Low-Rank Adaptation *(most popular)*

**Paper**: Hu et al. 2021 — https://arxiv.org/abs/2106.09685

**Core idea**: For a weight matrix W ∈ ℝ^{d×k}, instead of learning ΔW (d×k params), learn two low-rank matrices:

```
   W_adapted = W  +  (α/r) · B · A
               ^^^^  ^^^^^^^^^^^^^^^^
               frozen   ΔW, rank r

   A ∈ ℝ^{r×k}   (random Gaussian init)
   B ∈ ℝ^{d×r}   (zero init → ΔW = 0 at start)
   r << min(d, k)   e.g. r=4, r=8, r=16
   α = lora_alpha   scaling hyperparameter
```

During forward pass:
```python
# Standard linear:   out = x @ W.T
# LoRA linear:       out = x @ W.T  +  (alpha/r) * x @ A.T @ B.T
```

The (α/r) factor normalises the learning rate across different rank choices.
With α=r you get an unscaled delta (equivalent to lr=1×); with α=2r you
double the effective lr for the LoRA parameters.

**Parameter count**: r×k + d×r = r(d+k) vs d×k for full ΔW.
Example: d=k=4096, r=8 → 8×8192 = 65K vs 4096² = 16.8M. **~256× smaller**.

**Typical targets**: attention projection matrices (q_proj, v_proj, k_proj, o_proj).

### 2.2 QLoRA — Quantized LoRA

**Paper**: Dettmers et al. 2023 — https://arxiv.org/abs/2305.14314

Combines LoRA with 4-bit NF4 quantization of the frozen base weights:
- Base weights: 4-bit (NF4 or FP4) → ~4× VRAM reduction vs fp16
- LoRA adapters: kept in bf16/fp16
- "Double quantization": quantize the quantization constants too
- Requires `bitsandbytes` library and `load_in_4bit=True` in `BitsAndBytesConfig`

```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config)
# Then apply LoRA normally via get_peft_model()
```

### 2.3 AdaLoRA — Adaptive Budget Allocation

**Paper**: Zhang et al. 2023 — https://arxiv.org/abs/2303.10512

Instead of fixed rank r for all layers, AdaLoRA uses SVD-based importance
scoring to adaptively prune singular values and allocate rank budget where
it matters most. Uses `AdaLoraConfig` with `init_r` (initial rank), `target_r`
(final avg rank), and `tinit`/`tfinal` (warmup/cooldown steps).

### 2.4 Prefix Tuning

**Paper**: Li & Liang 2021 — https://arxiv.org/abs/2101.00190

Prepend a sequence of *trainable* "virtual tokens" to every Transformer layer's
key-value pairs. The base model weights are frozen; only the prefix embeddings
are trained. No architectural change to forward pass — just longer KV sequence.

```python
from peft import PrefixTuningConfig, TaskType
config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
```

### 2.5 Prompt Tuning

**Paper**: Lester et al. 2021 — https://arxiv.org/abs/2104.08691

Simpler than prefix tuning: prepend trainable tokens only at the *input embedding* layer (not all layers). Very few parameters (num_virtual_tokens × embed_dim).
Effective at scale (large models) but weaker than prefix for small models.

```python
from peft import PromptTuningConfig, TaskType
config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=8,
                            prompt_tuning_init="TEXT",
                            prompt_tuning_init_text="Classify sentiment:")
```

### 2.6 P-Tuning v2

**Paper**: Liu et al. 2022 — https://arxiv.org/abs/2110.07602

Prefix tuning generalised to all NLP tasks (not just generation). Applies
learnable prefix to every layer (like prefix tuning) but also works for
encoder-only / encoder-decoder models on classification.

### 2.7 IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations

**Paper**: Liu et al. 2022 — https://arxiv.org/abs/2205.05638

Learn three per-layer scalar vectors (l_k, l_v, l_ff) that rescale key,
value, and FFN activations:
```
  K' = l_k ⊙ K,   V' = l_v ⊙ V,   FFN' = l_ff ⊙ FFN
```
Even fewer parameters than LoRA (just 3×d per layer). Good for few-shot.

```python
from peft import IA3Config
config = IA3Config(task_type=TaskType.CAUSAL_LM,
                   target_modules=["k_proj","v_proj","down_proj"],
                   feedforward_modules=["down_proj"])
```

### 2.8 LoHa / LoKr (Hadamard / Kronecker LoRA)

Used primarily for diffusion model fine-tuning (Stable Diffusion):
- **LoHa**: ΔW = (A1 ⊗ A2) ⊙ (B1 ⊗ B2) — Hadamard product of two low-rank matrices
- **LoKr**: ΔW = A ⊗ B — Kronecker product factorization

---

## 3. LoraConfig Fields — Complete Reference

```python
from peft import LoraConfig, TaskType

config = LoraConfig(
    # ── Core rank/scaling ──────────────────────────────────────
    r=8,                          # Rank of ΔW. Higher r = more params = more capacity.
                                  # Typical: 4, 8, 16, 32, 64.
    lora_alpha=16,                # Scaling factor α. Effective LR scale = α/r.
                                  # Common: set α = 2×r for "standard" scaling.
    lora_dropout=0.05,            # Dropout on LoRA activations (regularisation).

    # ── Which layers to adapt ─────────────────────────────────
    target_modules=["q_proj", "v_proj"],
    # For GPT-2: ["c_attn"] (QKV fused); for LLaMA: ["q_proj","v_proj","k_proj","o_proj"]
    # Can also use regex: target_modules=r".*attn.*"
    # Or "all-linear" to target all Linear layers.

    # ── Bias handling ─────────────────────────────────────────
    bias="none",                  # "none": no bias params trained (recommended)
                                  # "all": train all biases
                                  # "lora_only": train only LoRA layer biases

    # ── Task type ─────────────────────────────────────────────
    task_type=TaskType.CAUSAL_LM, # CAUSAL_LM, SEQ_CLS, SEQ_2_SEQ_LM, TOKEN_CLS, etc.

    # ── Advanced ──────────────────────────────────────────────
    use_rslora=False,             # rsLoRA: scale by 1/sqrt(r) instead of 1/r
    modules_to_save=None,         # List of modules to unfreeze fully (e.g. "lm_head")
    init_lora_weights=True,       # True = Gaussian A, zero B. "loftq" = LoftQ init.
)
```

### Choosing `target_modules`

```python
# Print all module names to find what to target:
for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        print(name, module.__class__.__name__, module.weight.shape)
```

---

## 4. Core PEFT Workflow

```python
from peft import get_peft_model, PeftModel

# 1. Load base model (frozen)
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Wrap with PEFT
peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

# 3. Train (only LoRA params have gradients)
# ... training loop / Trainer ...

# 4. Save adapter only (~10MB, not 13GB)
peft_model.save_pretrained("./my-lora-adapter/")
# Saves: adapter_config.json + adapter_model.safetensors

# 5. Load for inference
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base, "./my-lora-adapter/")
model.eval()

# 6. Merge adapters into base weights (for deployment)
merged_model = model.merge_and_unload()
# merged_model is now a plain transformers model with W + BA baked in.
merged_model.save_pretrained("./merged-model/")
```

---

## 5. Multiple Adapters & Switching

```python
# Add a second adapter (base weights stay frozen)
peft_model.add_adapter("task_b", LoraConfig(r=4, ...))

# Switch active adapter
peft_model.set_adapter("task_a")
out_a = peft_model(inputs)

peft_model.set_adapter("task_b")
out_b = peft_model(inputs)

# Disable adapters (run base model only)
with peft_model.disable_adapter():
    out_base = peft_model(inputs)

# List adapters
print(peft_model.peft_config.keys())  # dict_keys(['task_a', 'task_b'])
```

---

## 6. Combining PEFT with Trainer / TRL

```python
from transformers import Trainer, TrainingArguments
from trl import SFTTrainer, SFTConfig

# Trainer sees only the trainable LoRA params → fast, low-memory
trainer = Trainer(model=peft_model, ...)

# TRL SFTTrainer: pass peft_config directly (it wraps automatically)
trainer = SFTTrainer(model=base_model, peft_config=lora_config, ...)
```

---

*Generated for peft 0.19 — docs: https://huggingface.co/docs/peft*

"""
TRL DPOTrainer + DPO Loss from Scratch
========================================
Docs: https://huggingface.co/docs/trl/dpo_trainer
      https://huggingface.co/docs/trl/dpo_trainer#expected-dataset-format
      DPO paper: https://arxiv.org/abs/2305.18290

Demonstrates:
1. DPO loss from scratch in NumPy/PyTorch — teaches the math
2. DPOTrainer live run with tiny in-memory preference dataset
   (explicit ref_model required in trl 1.5 when base_model_name_or_path is empty)

DPO Loss formula:
  L_DPO = -E[ log σ( β * (log π_θ(y_w|x)/π_ref(y_w|x)
                         - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]

Where:
  y_w = chosen (preferred) response
  y_l = rejected response
  β   = temperature (controls deviation from reference)
  σ   = sigmoid
  π_ref = frozen reference (SFT) model
"""

import sys
import os
import random
import copy
import tempfile
import warnings
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import numpy as np
np.random.seed(0)
random.seed(0)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

# ═══════════════════════════════════════════════════════════════════════════
# PART A — DPO LOSS FROM SCRATCH
# Always runs — teaches the math on toy logprobs regardless of trainer API.
# ═══════════════════════════════════════════════════════════════════════════

banner("A. DPO Loss from scratch — toy logprobs (always runs)")
print("""
DPO derives a supervised objective directly from the RLHF optimisation problem.

Starting from:  max_π  E[R(x,y)] − β·KL[π||π_ref]
The closed-form optimal policy is:
    π*(y|x) ∝ π_ref(y|x) · exp(R(x,y)/β)

Rearranging: R(x,y) = β·log[π*(y|x)/π_ref(y|x)] + β·log Z(x)

With Bradley-Terry preference model P(y_w≻y_l) = σ(R(y_w) − R(y_l)):
Plugging in and Z(x) cancels:

  L_DPO = −E[ log σ( β · (log_ratio_chosen − log_ratio_rejected) ) ]

where:
  log_ratio_chosen  = log π_θ(y_w|x) − log π_ref(y_w|x)
  log_ratio_rejected = log π_θ(y_l|x) − log π_ref(y_l|x)
""")

# Toy scenario: 4 preference pairs, each with scalar log-probs
BETA = 0.1
N_PAIRS = 6

# Simulate log-probs under policy and reference model
# (in practice these are sum of per-token log-probs)
torch.manual_seed(42)
logp_policy_chosen   = torch.randn(N_PAIRS) * 2 - 3   # e.g. around -5 to -1
logp_policy_rejected = torch.randn(N_PAIRS) * 2 - 4   # slightly lower

# Reference model: slightly different distribution
logp_ref_chosen   = logp_policy_chosen   + torch.randn(N_PAIRS) * 0.3
logp_ref_rejected = logp_policy_rejected + torch.randn(N_PAIRS) * 0.3

# Compute log-ratios: how much more/less likely than reference
log_ratio_chosen   = logp_policy_chosen   - logp_ref_chosen
log_ratio_rejected = logp_policy_rejected - logp_ref_rejected

print(f"  log_ratio_chosen   : {log_ratio_chosen.numpy().round(3)}")
print(f"  log_ratio_rejected : {log_ratio_rejected.numpy().round(3)}")

# DPO loss per pair
delta = BETA * (log_ratio_chosen - log_ratio_rejected)
dpo_loss_per_pair = -torch.nn.functional.logsigmoid(delta)
dpo_loss = dpo_loss_per_pair.mean()

print(f"\n  β*(ratio_chosen − ratio_rejected): {delta.numpy().round(3)}")
print(f"  −log σ(δ) per pair              : {dpo_loss_per_pair.numpy().round(3)}")
print(f"  DPO loss (mean)                 : {dpo_loss.item():.4f}")
print(f"  (log(2)≈0.693 is the 'random' baseline — lower is better)")

# Show gradient direction
print("""
  Gradient intuition:
    When policy assigns relatively MORE probability to chosen vs rejected
    (compared to reference), δ > 0 → σ(δ) > 0.5 → loss < log(2).
    The gradient nudges policy toward chosen, away from rejected.
""")

# NumPy implementation (equivalent)
banner("A2. NumPy DPO loss — explicit sigmoid formula")

def dpo_loss_numpy(logp_pi_w, logp_pi_l, logp_ref_w, logp_ref_l, beta=0.1):
    """
    DPO loss from scratch in NumPy.

    Args:
        logp_pi_w  : log π_θ(y_w|x)  — policy logprob on chosen
        logp_pi_l  : log π_θ(y_l|x)  — policy logprob on rejected
        logp_ref_w : log π_ref(y_w|x) — reference logprob on chosen
        logp_ref_l : log π_ref(y_l|x) — reference logprob on rejected
        beta       : KL temperature
    Returns:
        scalar loss (lower = better alignment)
    """
    ratio_w = logp_pi_w - logp_ref_w     # log-ratio for chosen
    ratio_l = logp_pi_l - logp_ref_l     # log-ratio for rejected
    delta   = beta * (ratio_w - ratio_l) # preference signal
    # sigmoid(x) = 1 / (1 + e^{-x}),  log sigmoid(x) = -log(1+e^{-x})
    log_sigmoid = -np.log1p(np.exp(-delta))   # numerically stable
    return -np.mean(log_sigmoid)

# Use same toy values
loss_np = dpo_loss_numpy(
    logp_policy_chosen.numpy(),
    logp_policy_rejected.numpy(),
    logp_ref_chosen.numpy(),
    logp_ref_rejected.numpy(),
    beta=BETA,
)
print(f"  NumPy DPO loss: {loss_np:.4f}")
print(f"  PyTorch loss  : {dpo_loss.item():.4f}")
print(f"  Match: {abs(loss_np - dpo_loss.item()) < 1e-5}")

# Show how loss changes as policy improves
banner("A3. Loss vs. preference strength (sweep)")
print("  Simulating improving policy (chosen gets higher, rejected lower):\n")
print(f"  {'gap':>6}  {'dpo_loss':>10}  {'interpretation'}")
print(f"  {'-'*50}")
for gap in [-0.5, 0.0, 0.5, 1.0, 2.0, 4.0]:
    # Larger gap = policy more confidently prefers chosen over rejected
    lp_w = np.full(4, -3.0 + gap / 2)
    lp_l = np.full(4, -3.0 - gap / 2)
    lref = np.full(4, -3.0)   # ref is neutral
    loss_v = dpo_loss_numpy(lp_w, lp_l, lref, lref, beta=BETA)
    interp = "random" if abs(gap) < 0.1 else ("better" if gap > 0 else "worse")
    print(f"  {gap:>6.1f}  {loss_v:>10.4f}  {interp}")

# ═══════════════════════════════════════════════════════════════════════════
# PART B — DPOTrainer LIVE RUN
# ═══════════════════════════════════════════════════════════════════════════

banner("B. DPOTrainer — live run (trl 1.5)")

from transformers import GPT2Config, GPT2LMHeadModel
import datasets as hf_datasets

VOCAB_SIZE = 50257

def try_hub_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    return mdl, tok

def build_local_model(vocab):
    cfg = GPT2Config(
        n_layer=2, n_head=2, n_embd=64,
        vocab_size=vocab, n_positions=128,
    )
    return GPT2LMHeadModel(cfg)

tokenizer = None
policy_model = None

ok, hub_result = safe(try_hub_model)
if ok:
    policy_model, tokenizer = hub_result
    VOCAB_SIZE = policy_model.config.vocab_size
    print(f"  Loaded sshleifer/tiny-gpt2 from hub (vocab={VOCAB_SIZE})")
else:
    note_skip(
        f"needs network/model — building local GPT2 fallback ({hub_result})"
    )
    policy_model = build_local_model(VOCAB_SIZE)
    # Try getting tokenizer separately (needed for DPOTrainer)
    ok2, tok_result = safe(lambda: __import__('transformers').AutoTokenizer.from_pretrained(
        "sshleifer/tiny-gpt2"
    ))
    if ok2:
        tokenizer = tok_result
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = None
    print(f"  Built local GPT2 (vocab={VOCAB_SIZE})")

# Reference model: frozen copy of policy (or separate load)
ref_model = copy.deepcopy(policy_model)
for p in ref_model.parameters():
    p.requires_grad_(False)
ref_model.eval()
print("  Reference model: frozen copy of policy model")

# Build preference dataset
PREFERENCE_DATA = {
    "prompt": [
        "Explain machine learning:",
        "What is a neural network?",
        "Describe gradient descent:",
        "What is attention in transformers?",
        "Explain overfitting:",
        "What is backpropagation?",
    ],
    "chosen": [
        "Machine learning is a field of AI that enables computers to learn patterns from data.",
        "A neural network is a computational model inspired by the human brain's structure.",
        "Gradient descent is an optimisation algorithm that iteratively reduces the loss.",
        "Attention allows models to weigh the importance of different tokens in a sequence.",
        "Overfitting occurs when a model memorises training data and fails to generalise.",
        "Backpropagation computes gradients via the chain rule for training neural networks.",
    ],
    "rejected": [
        "Machine learning is when computers do stuff automatically.",
        "A neural network is a network that thinks like a brain kind of.",
        "Gradient descent goes down to find the minimum somehow.",
        "Attention is when the model pays attention to things.",
        "Overfitting is bad for models and should be avoided.",
        "Backpropagation is how neural nets learn things.",
    ],
}

pref_dataset = hf_datasets.Dataset.from_dict(PREFERENCE_DATA)
print(f"\n  Preference dataset: {len(pref_dataset)} rows")
print(f"  Columns: {pref_dataset.column_names}")
print(f"  Example prompt  : '{PREFERENCE_DATA['prompt'][0]}'")
print(f"  Example chosen  : '{PREFERENCE_DATA['chosen'][0][:50]}...'")
print(f"  Example rejected: '{PREFERENCE_DATA['rejected'][0][:50]}...'")

# Run DPOTrainer
dpo_live = False

if tokenizer is not None:
    def run_dpo():
        from trl import DPOTrainer, DPOConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            dpo_cfg = DPOConfig(
                output_dir=tmpdir,
                max_steps=4,                    # tiny run
                per_device_train_batch_size=2,
                learning_rate=1e-4,
                beta=0.1,                       # KL penalty coefficient
                report_to="none",
                use_cpu=True,
                logging_steps=1,
                save_strategy="no",
                disable_tqdm=False,
            )
            trainer = DPOTrainer(
                model=policy_model,
                ref_model=ref_model,            # explicit ref (needed in trl 1.5)
                args=dpo_cfg,
                train_dataset=pref_dataset,
                processing_class=tokenizer,
            )
            result = trainer.train()
            return result

    banner("B1. DPOTrainer training run")
    ok_dpo, dpo_result = safe(run_dpo)
    if ok_dpo:
        dpo_live = True
        print(f"\n  DPOTrainer SUCCESS")
        print(f"  train_loss    : {dpo_result.training_loss:.4f}")
        print(f"  global_step   : {dpo_result.global_step}")
        print("  (loss ≈ log(2) ≈ 0.693 at initialisation — random policy)")
    else:
        note_skip(f"DPOTrainer failed: {dpo_result}")
        print("  The DPO loss from scratch (Part A) still demonstrates the math.")
else:
    note_skip("No tokenizer available — skipping DPOTrainer live run (Part A still shows math).")

# ── DPO config explanation ────────────────────────────────────────────────────
banner("C. DPOConfig key fields (trl 1.5)")
print("""
DPOConfig (inherits TrainingArguments + DPO-specific fields):

  beta           : KL penalty (default 0.1). Higher β → stay closer to ref.
                   Typical range: 0.01 – 1.0
  loss_type      : 'sigmoid' (standard DPO), 'hinge', 'ipo', 'kto_pair'
                   'ipo' = Identity Preference Optimisation (no log sigmoid)
  label_smoothing: smooths the logistic loss (0.0 = standard DPO)
  max_length     : max total sequence length (prompt + response)
  max_prompt_length : max prompt length (rest is response)
  use_cpu        : force CPU training (for testing)
  report_to='none' : disable experiment tracking

Dataset format (required columns):
  prompt   : the input prompt string
  chosen   : the preferred response string
  rejected : the dispreferred response string

DPOTrainer automatically tokenises, prepends prompt to each response,
computes log-probs under policy and reference, and applies the DPO loss.

Docs: https://huggingface.co/docs/trl/dpo_trainer#trl.DPOConfig
""")

# ── Summary ──────────────────────────────────────────────────────────────────
banner("D. Summary")
print(f"""
Part A: DPO loss from scratch — ALWAYS RUNS (numpy + torch)
Part B: DPOTrainer live run   — {'LIVE (succeeded)' if dpo_live else 'SKIPPED (no tokenizer or trainer error)'}

DPO in one line:
  "Prefer responses that are more likely under the policy
   relative to the reference, while penalising KL divergence."

The key insight: DPO converts an RL problem into a supervised one,
eliminating the need for an explicit reward model or RL rollouts.
""")

print("\nDONE — exit 0")
sys.exit(0)

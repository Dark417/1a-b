"""
03.generation.py — Decoding Strategies for Text Generation
===========================================================
Autoregressive generation: at each step, the model produces logits over the
entire vocabulary; a *decoding strategy* picks the next token from those logits.

Math intuition (brief):

  - **Greedy**: next = argmax(logits)  — deterministic, can be repetitive
  - **Beam search**: keep top-B hypotheses; score = sum of log-probs
  - **Temperature**: divide logits by T before softmax
        P(token) = softmax(logits / T)
        T→0 : greedy; T→∞ : uniform distribution; T<1 sharpens, T>1 flattens
  - **Top-k**: zero out all but the top-k logits, then sample
  - **Top-p (nucleus)**: keep the smallest set of tokens whose cumulative
        probability ≥ p, then sample; adapts vocabulary size per step
  - **Repetition penalty**: logits[token] /= penalty if token was seen before
  - **no_repeat_ngram_size**: block any n-gram that has already appeared

Official docs:
  - https://huggingface.co/docs/transformers/generation_strategies
  - https://huggingface.co/docs/transformers/main_classes/text_generation

Tiny model used: sshleifer/tiny-gpt2
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

PROMPT = "Once upon a time"
MAX_NEW = 30

# ─── Load model once (shared across all examples) ────────────────────────────
banner("Loading sshleifer/tiny-gpt2")

ok_tok, tok = safe(AutoTokenizer.from_pretrained, "sshleifer/tiny-gpt2")
ok_mdl, model = safe(AutoModelForCausalLM.from_pretrained, "sshleifer/tiny-gpt2")

if ok_tok and ok_mdl:
    tok.pad_token = tok.eos_token           # GPT-2 has no pad token
    model.eval()
    inputs = tok(PROMPT, return_tensors="pt")
    MODEL_AVAILABLE = True
    print(f"  Model loaded. vocab_size={model.config.vocab_size}")
else:
    note_skip(f"tok={ok_tok} model={ok_mdl}")
    MODEL_AVAILABLE = False
    print("  Showing GenerationConfig API shape only (all strategies still constructed).")


def decode(ids):
    """Decode token ids → string."""
    return tok.decode(ids[0], skip_special_tokens=True)


def gen(**kwargs):
    """Run model.generate with shared inputs; return decoded string."""
    if not MODEL_AVAILABLE:
        return "[skip]"
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            pad_token_id=tok.eos_token_id,
            **kwargs,
        )
    return decode(out)


# ─── 1. Greedy decoding ──────────────────────────────────────────────────────
banner("1. Greedy Decoding (do_sample=False, num_beams=1)")
# At each step: next_token = argmax(logits)
# Fast and deterministic. Tends to produce repetitive, "safe" text.
text = gen(do_sample=False)
print(f"  Greedy: {text!r}")

# ─── 2. Beam Search ───────────────────────────────────────────────────────────
banner("2. Beam Search (num_beams=4)")
# Maintains B=4 partial hypotheses simultaneously.
# Each step: expand each hypothesis → pick top-B by cumulative log-prob.
# Higher beam count → better quality but slower (quadratic memory in beam×vocab).
text = gen(num_beams=4, do_sample=False, early_stopping=True)
print(f"  Beam-4: {text!r}")

# Multiple sequences via beam search
if MODEL_AVAILABLE:
    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            num_beams=4,
            num_return_sequences=2,   # return 2 of the 4 beams
            pad_token_id=tok.eos_token_id,
        )
    for i, seq in enumerate(outs):
        print(f"  Beam seq {i}: {decode(seq.unsqueeze(0))!r}")

# ─── 3. Sampling (do_sample=True) ────────────────────────────────────────────
banner("3. Pure Sampling (do_sample=True, temperature=1.0)")
# Sample from the full vocabulary distribution at each step.
# Stochastic: different seed → different output.
torch.manual_seed(42)
text = gen(do_sample=True, temperature=1.0)
print(f"  Sample T=1.0: {text!r}")

# ─── 4. Temperature scaling ───────────────────────────────────────────────────
banner("4. Temperature (T=0.3 vs T=1.5)")
# T < 1 sharpens distribution → more predictable (closer to greedy)
# T > 1 flattens distribution → more random / creative
torch.manual_seed(42)
text_low = gen(do_sample=True, temperature=0.3)
torch.manual_seed(42)
text_high = gen(do_sample=True, temperature=1.5)
print(f"  T=0.3  (sharp)  : {text_low!r}")
print(f"  T=1.5  (flat)   : {text_high!r}")

# ─── 5. Top-k Sampling ────────────────────────────────────────────────────────
banner("5. Top-k Sampling (top_k=50)")
# Zero out all but the top-50 logits by probability, then sample.
# Prevents drawing from very low-probability tokens but uses fixed k.
torch.manual_seed(42)
text = gen(do_sample=True, top_k=50, temperature=1.0)
print(f"  top_k=50: {text!r}")

# ─── 6. Top-p (Nucleus) Sampling ─────────────────────────────────────────────
banner("6. Top-p / Nucleus Sampling (top_p=0.92)")
# Sort tokens by descending probability; keep the minimum set whose CDF ≥ p=0.92.
# Adapts: when the model is confident, k is small; when uncertain, k is large.
# Often combined with temperature for best results.
torch.manual_seed(42)
text = gen(do_sample=True, top_p=0.92, temperature=0.9)
print(f"  top_p=0.92: {text!r}")

# Combined top_k + top_p (common production setting)
torch.manual_seed(42)
text = gen(do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
print(f"  top_k=50 + top_p=0.95: {text!r}")

# ─── 7. Repetition Penalty ────────────────────────────────────────────────────
banner("7. Repetition Penalty (repetition_penalty=1.3)")
# For each token that has already appeared: logit /= penalty (if logit > 0)
#                                            logit *= penalty (if logit < 0)
# penalty=1.0 = no effect; penalty>1.0 discourages repeats.
torch.manual_seed(42)
text = gen(do_sample=False, repetition_penalty=1.3)
print(f"  rep_penalty=1.3: {text!r}")

# ─── 8. no_repeat_ngram_size ─────────────────────────────────────────────────
banner("8. no_repeat_ngram_size=3")
# Hard constraint: never produce an n-gram that has already appeared.
# n=3 means no trigram repeats; n=2 no bigram repeats.
# Stronger than repetition_penalty; works by setting banned n-gram logits = -inf.
torch.manual_seed(42)
text = gen(do_sample=False, no_repeat_ngram_size=3)
print(f"  no_repeat_ngram=3: {text!r}")

# ─── 9. num_return_sequences with sampling ────────────────────────────────────
banner("9. num_return_sequences=3 (sampling)")
# Generate 3 independent samples in a single forward pass (batch expansion)
if MODEL_AVAILABLE:
    torch.manual_seed(42)
    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            num_return_sequences=3,
            pad_token_id=tok.eos_token_id,
        )
    for i, seq in enumerate(outs):
        print(f"  Seq {i}: {decode(seq.unsqueeze(0))!r}")
else:
    print("  [skip] showing shape: outs.shape = (num_return_sequences, gen_len)")

# ─── 10. GenerationConfig ─────────────────────────────────────────────────────
banner("10. GenerationConfig — serialisable strategy config")
# GenerationConfig bundles all generation kwargs into a saveable config object.
# model.generation_config is the model's default; override per call.

# Build a config for creative writing
creative_cfg = GenerationConfig(
    do_sample=True,
    temperature=0.85,
    top_k=60,
    top_p=0.92,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    max_new_tokens=40,
    num_return_sequences=1,
)
print(f"  creative_cfg.do_sample    : {creative_cfg.do_sample}")
print(f"  creative_cfg.temperature  : {creative_cfg.temperature}")
print(f"  creative_cfg.top_k        : {creative_cfg.top_k}")

if MODEL_AVAILABLE:
    torch.manual_seed(42)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            generation_config=creative_cfg,
            pad_token_id=tok.eos_token_id,
        )
    print(f"  Generated with config: {decode(out)!r}")
else:
    print("  [skip] model unavailable; GenerationConfig object constructed OK")

# Show model's built-in default GenerationConfig
if MODEL_AVAILABLE:
    print(f"\n  model.generation_config keys: {list(model.generation_config.to_dict().keys())[:8]} …")

# ─── 11. Length control kwargs ────────────────────────────────────────────────
banner("11. Length control: min_length, max_new_tokens, eos_token_id")
# max_new_tokens: how many NEW tokens to generate (does not count prompt)
# max_length: total length including prompt (deprecated for max_new_tokens)
# min_new_tokens: minimum NEW tokens before EOS is allowed
# forced_eos_token_id: always end with this token id

if MODEL_AVAILABLE:
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=15,
            min_new_tokens=5,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    n_new = out.shape[1] - inputs['input_ids'].shape[1]
    print(f"  Prompt tokens: {inputs['input_ids'].shape[1]}")
    print(f"  New tokens generated: {n_new}")
    print(f"  Full output: {decode(out)!r}")
else:
    print("  [skip] demonstrating kwargs: max_new_tokens=15, min_new_tokens=5")

print("\n[DONE] 03.generation.py complete")

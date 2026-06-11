"""
03.generation.py — Decoding Strategies for Text Generation
===========================================================
HuggingFace provides a rich generate() API that wraps many decoding algorithms.
This module demonstrates each strategy with tiny-gpt2 and explains the math.

Official docs:
  https://huggingface.co/docs/transformers/generation_strategies
  https://huggingface.co/docs/transformers/main_classes/text_generation

Tiny model: sshleifer/tiny-gpt2 (fast on CPU, ~1M params)

Math intuitions inline in comments.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, GPT2Config, GPT2LMHeadModel

# ─── Load model once; reuse throughout ────────────────────────────────────────
banner("Loading tiny-gpt2 for all generation experiments")

ok_tok, tok = safe(AutoTokenizer.from_pretrained, "sshleifer/tiny-gpt2")
ok_mdl, model = safe(AutoModelForCausalLM.from_pretrained, "sshleifer/tiny-gpt2")

USING_LIVE = ok_tok and ok_mdl

if USING_LIVE:
    tok.pad_token = tok.eos_token     # GPT-2 has no pad token
    model.eval()
    VOCAB = tok.vocab_size
    prompt = "The future of artificial intelligence is"
    inputs = tok(prompt, return_tensors="pt")
    print(f"  Model loaded! Vocab={VOCAB}, prompt tokens={inputs['input_ids'].shape[1]}")
else:
    note_skip(str(model if not ok_mdl else tok))
    # Build a tiny GPT-2 from scratch for shape demonstrations
    cfg = GPT2Config(n_embd=64, n_head=2, n_layer=2, vocab_size=1000,
                     n_positions=64, n_ctx=64)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    VOCAB = 1000
    # Create simple tokenizer shim
    class TinyTok:
        pad_token_id = 0
        eos_token_id = 0
        bos_token_id = 0
        def __call__(self, text, return_tensors=None):
            # Produce 5 fake tokens
            ids = torch.randint(1, VOCAB, (1, 5))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        def decode(self, ids, skip_special_tokens=True):
            return f"<decoded {len(ids) if hasattr(ids, '__len__') else '?'} tokens>"
    tok = TinyTok()
    inputs = tok("fake prompt", return_tensors="pt")
    print("  [fallback] Using tiny GPT-2 built from scratch (random weights)")


def gen(extra_label="", **kwargs):
    """Helper: generate and decode, return text."""
    torch.manual_seed(42)
    # Always set pad_token_id to avoid warning
    kwargs.setdefault("pad_token_id", tok.eos_token_id if hasattr(tok, "eos_token_id") else 0)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, **kwargs)
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"  [{extra_label}] {text!r}")
    return text


# ─── 1. Greedy Decoding ───────────────────────────────────────────────────────
banner("1. Greedy Decoding")
# Algorithm: at each step, pick the token with highest probability.
#   next_token = argmax P(token | context)
# Properties:
#   + Deterministic, fast
#   - Repetitive, can miss the globally optimal sequence
#   - do_sample=False (default) enables greedy

gen("greedy", do_sample=False)

# ─── 2. Beam Search ──────────────────────────────────────────────────────────
banner("2. Beam Search")
# Algorithm: keep top-k (= num_beams) candidate sequences at each step.
#   Score = sum log P(token_i | context)
# Properties:
#   + Better than greedy (explores more paths)
#   - More memory: stores num_beams × seq_len tokens
#   - Can still be repetitive for open-ended generation
# num_beams=1 degenerates to greedy.

gen("beam-5", do_sample=False, num_beams=5, early_stopping=True)
gen("beam-3", do_sample=False, num_beams=3)

# ─── 3. Sampling ─────────────────────────────────────────────────────────────
banner("3. Random Sampling (do_sample=True)")
# Algorithm: sample from the full vocabulary distribution each step.
#   next_token ~ Categorical(softmax(logits))
# Properties:
#   + Diverse, creative outputs
#   - Can produce incoherent low-probability tokens

torch.manual_seed(0)
gen("sample", do_sample=True)

# ─── 4. Temperature Scaling ───────────────────────────────────────────────────
banner("4. Temperature Scaling")
# Modifies the logit distribution BEFORE softmax:
#   P(token) = softmax(logits / T)
#
#   T → 0  : sharper distribution, converges to greedy
#   T = 1  : original distribution
#   T → ∞  : uniform distribution, pure random
#
# Low temperature = conservative / focused
# High temperature = creative / risky

torch.manual_seed(0)
gen("temp=0.3 (sharp)", do_sample=True, temperature=0.3)
torch.manual_seed(0)
gen("temp=1.0 (neutral)", do_sample=True, temperature=1.0)
torch.manual_seed(0)
gen("temp=1.5 (creative)", do_sample=True, temperature=1.5)

# ─── 5. Top-k Sampling ───────────────────────────────────────────────────────
banner("5. Top-k Sampling")
# Algorithm: at each step, keep only the k most likely tokens, re-normalize,
#   then sample from that reduced distribution.
#   top_k = 1 → greedy; top_k = vocab_size → full sampling
#
# Properties:
#   + Prevents very low-probability tokens
#   - k is fixed regardless of distribution shape

torch.manual_seed(0)
gen("top-k=10", do_sample=True, top_k=10)
torch.manual_seed(0)
gen("top-k=50", do_sample=True, top_k=50)

# ─── 6. Top-p / Nucleus Sampling ─────────────────────────────────────────────
banner("6. Top-p (Nucleus) Sampling")
# Algorithm: sort tokens by probability descending, keep the smallest set
#   whose cumulative probability ≥ p, then sample from that set.
#   Formally: nucleus = {v : Σ P(v') >= p, sorted by P desc}
#
# Properties:
#   + Adapts to distribution shape (more tokens when distribution is flat)
#   + Avoids low-prob garbage
#   top_p=1.0 → no restriction; top_p=0.9 is a common default

torch.manual_seed(0)
gen("top-p=0.9", do_sample=True, top_p=0.9)
torch.manual_seed(0)
gen("top-p=0.95+top-k=50", do_sample=True, top_p=0.95, top_k=50)

# ─── 7. Repetition Penalty ───────────────────────────────────────────────────
banner("7. Repetition Penalty")
# Divides the logit of any token that already appeared in context:
#   logit_i /= repetition_penalty  (if logit_i > 0)
#   logit_i *= repetition_penalty  (if logit_i < 0)
# Values > 1.0 discourage repeating; 1.0 = no effect.

torch.manual_seed(0)
gen("rep=1.0 (off)", do_sample=False, repetition_penalty=1.0)
torch.manual_seed(0)
gen("rep=1.3", do_sample=False, repetition_penalty=1.3)

# ─── 8. No-repeat n-gram ─────────────────────────────────────────────────────
banner("8. no_repeat_ngram_size")
# Hard constraint: if an n-gram has already appeared in the output,
# forbid completing it again.
# no_repeat_ngram_size=3 → no 3-gram can repeat.

torch.manual_seed(0)
gen("no_repeat_ngram=3", do_sample=False, no_repeat_ngram_size=3)

# ─── 9. num_return_sequences ─────────────────────────────────────────────────
banner("9. num_return_sequences")
# Return multiple completions in one call.
# Requires do_sample=True OR num_beams >= num_return_sequences.

torch.manual_seed(0)
pad_id = tok.eos_token_id if hasattr(tok, "eos_token_id") else 0
with torch.no_grad():
    outs = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=True,
        temperature=1.0,
        num_return_sequences=3,
        pad_token_id=pad_id,
    )
for i, o in enumerate(outs):
    print(f"  seq {i}: {tok.decode(o, skip_special_tokens=True)!r}")

# ─── 10. GenerationConfig ────────────────────────────────────────────────────
banner("10. GenerationConfig — bundle all gen params")
# GenerationConfig stores all generation hyperparameters.
# Can be saved to disk alongside the model checkpoint.
# model.generation_config is the default for that model.

gen_cfg = GenerationConfig(
    max_new_tokens=15,
    do_sample=True,
    temperature=0.8,
    top_k=40,
    top_p=0.92,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
    num_return_sequences=1,
)
print(f"  GenerationConfig:\n  {gen_cfg}")

torch.manual_seed(42)
with torch.no_grad():
    out = model.generate(**inputs, generation_config=gen_cfg,
                         pad_token_id=pad_id)
print(f"  Output: {tok.decode(out[0], skip_special_tokens=True)!r}")

print("\n[DONE] 03.generation.py complete")

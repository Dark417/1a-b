"""
02.automodel.py — Auto* Classes and Manual Forward Pass
========================================================
The Auto* family reads config.json from any checkpoint and returns the correct
class without you knowing the architecture in advance.

Official docs:
  - https://huggingface.co/docs/transformers/model_doc/auto
  - https://huggingface.co/docs/transformers/main_classes/configuration

Key insight: config.json → model_type → concrete class
  e.g. model_type="bert" → BertConfig, BertModel, BertForMaskedLM, …

Tiny models used:
  - prajjwal1/bert-tiny   (encoder)
  - sshleifer/tiny-gpt2   (causal LM decoder)
  - hf-internal-testing/tiny-random-t5  (seq2seq)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    BertConfig,
    BertModel,
)

# ─── 1. AutoConfig: read config.json without downloading weights ───────────────
banner("1. AutoConfig — reading config.json")
# AutoConfig.from_pretrained only downloads config.json (~few KB), not weights.
# It returns the concrete config class (BertConfig, GPT2Config, …)

ok, config = safe(AutoConfig.from_pretrained, "prajjwal1/bert-tiny")
if ok:
    print(f"  model_type       : {config.model_type}")
    print(f"  hidden_size      : {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  vocab_size       : {config.vocab_size}")
    print(f"  Config class     : {type(config).__name__}")
else:
    note_skip(str(config))
    # Build a BertConfig from scratch — same object you'd get from from_pretrained
    config = BertConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        vocab_size=30522,
    )
    print(f"  [fallback] model_type: {config.model_type}")
    print(f"  [fallback] hidden_size: {config.hidden_size}")

# ─── 2. Overriding config fields (no download) ────────────────────────────────
banner("2. Config from scratch (no download needed)")
# You can always create a config programmatically — useful for custom architectures
tiny_cfg = BertConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=128,
    vocab_size=1000,
)
print(f"  Custom config hidden_size: {tiny_cfg.hidden_size}")
print(f"  Custom config to_dict keys: {list(tiny_cfg.to_dict().keys())[:6]} …")

# ─── 3. AutoTokenizer ─────────────────────────────────────────────────────────
banner("3. AutoTokenizer — tokenize → input_ids")
# Returns the fast (Rust-backed) tokenizer when available
# Encodes text to token ids, handles special tokens automatically

ok, tok = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
if ok:
    text = "Hello, Hugging Face!"
    enc = tok(text, return_tensors="pt")
    print(f"  Text         : {text!r}")
    print(f"  input_ids    : {enc['input_ids']}")
    print(f"  token_type_ids: {enc.get('token_type_ids')}")
    print(f"  attention_mask: {enc['attention_mask']}")
    # Decode back to string
    decoded = tok.decode(enc['input_ids'][0], skip_special_tokens=True)
    print(f"  Decoded      : {decoded!r}")
    # Batch tokenization with padding
    batch = tok(
        ["Short sentence.", "This is a longer sentence with more tokens."],
        padding=True, truncation=True, max_length=32, return_tensors="pt"
    )
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")  # (2, padded_len)
else:
    note_skip(str(tok))
    print("  API shape: input_ids tensor (batch, seq_len), attention_mask same shape")

# ─── 4. AutoModel — base encoder (no task head) ──────────────────────────────
banner("4. AutoModel — raw hidden states")
# AutoModel returns the bare transformer stack: just hidden states, no task head.
# Useful for feature extraction / embeddings.

ok_tok, tok = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, model = safe(AutoModel.from_pretrained, "prajjwal1/bert-tiny")

if ok_tok and ok_mdl:
    inputs = tok("Feature extraction example", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # last_hidden_state: (batch, seq_len, hidden_size)
    # pooler_output (BERT): (batch, hidden_size) — [CLS] through a linear+tanh
    print(f"  last_hidden_state shape: {outputs.last_hidden_state.shape}")
    print(f"  pooler_output shape    : {outputs.pooler_output.shape}")
    # CLS embedding = first token of last_hidden_state
    cls_emb = outputs.last_hidden_state[:, 0, :]
    print(f"  CLS embedding shape    : {cls_emb.shape}")
    # Mean pool (ignore padding via attention_mask)
    mask = inputs['attention_mask'].unsqueeze(-1).float()
    mean_emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    print(f"  Mean-pool emb shape    : {mean_emb.shape}")
else:
    note_skip(f"tok={ok_tok}, mdl={ok_mdl}")
    # Fallback: build tiny model from scratch
    cfg = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, vocab_size=1000)
    fb_model = BertModel(cfg)
    dummy_ids = torch.randint(0, 1000, (1, 8))
    dummy_mask = torch.ones(1, 8, dtype=torch.long)
    with torch.no_grad():
        out = fb_model(input_ids=dummy_ids, attention_mask=dummy_mask)
    print(f"  [fallback] last_hidden_state: {out.last_hidden_state.shape}")
    print(f"  [fallback] pooler_output    : {out.pooler_output.shape}")

# ─── 5. AutoModelForSequenceClassification ────────────────────────────────────
banner("5. AutoModelForSequenceClassification — logits + class head")
# Adds a Linear(hidden_size, num_labels) on top of [CLS] token

ok_tok, tok = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, clf = safe(
    AutoModelForSequenceClassification.from_pretrained,
    "prajjwal1/bert-tiny",
    num_labels=2,         # override; random head weights
    ignore_mismatched_sizes=True,
)
if ok_tok and ok_mdl:
    inputs = tok("I love transformers!", return_tensors="pt")
    with torch.no_grad():
        out = clf(**inputs)
    print(f"  logits shape : {out.logits.shape}")          # (1, 2)
    probs = torch.softmax(out.logits, dim=-1)
    print(f"  probabilities: {probs}")
else:
    note_skip(f"clf init failed")
    print("  API shape: out.logits = (batch, num_labels)")

# ─── 6. AutoModelForCausalLM — GPT-style next-token prediction ──────────────
banner("6. AutoModelForCausalLM — causal LM logits")
# For generation: the last token's logits → next token distribution

ok_tok, tok2 = safe(AutoTokenizer.from_pretrained, "sshleifer/tiny-gpt2")
ok_mdl, gpt = safe(AutoModelForCausalLM.from_pretrained, "sshleifer/tiny-gpt2")

if ok_tok and ok_mdl:
    # GPT-2 has no pad token; set it to eos
    tok2.pad_token = tok2.eos_token
    inputs = tok2("The quick brown fox", return_tensors="pt")
    with torch.no_grad():
        out = gpt(**inputs)
    # logits: (batch, seq_len, vocab_size)
    print(f"  logits shape: {out.logits.shape}")
    # Next-token prediction: take last position, argmax
    next_tok_id = out.logits[0, -1, :].argmax()
    print(f"  Greedy next token: {tok2.decode([next_tok_id.item()])!r}")
    # Loss is computed if labels are provided (same as input_ids for LM)
    inputs_with_labels = {**inputs, "labels": inputs["input_ids"]}
    with torch.no_grad():
        out_with_loss = gpt(**inputs_with_labels)
    print(f"  LM loss (teacher-forcing): {out_with_loss.loss.item():.4f}")
else:
    note_skip("CausalLM init failed")
    print("  API shape: out.logits = (batch, seq_len, vocab_size); out.loss = scalar")

# ─── 7. AutoModelForMaskedLM ──────────────────────────────────────────────────
banner("7. AutoModelForMaskedLM — predict [MASK] tokens")
# Outputs logits for every position; at [MASK] positions, pick the most likely token

ok_tok, tok3 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, mlm = safe(AutoModelForMaskedLM.from_pretrained, "prajjwal1/bert-tiny")

if ok_tok and ok_mdl:
    text = "Paris is the [MASK] of France."
    inputs = tok3(text, return_tensors="pt")
    # Find position of [MASK] token
    mask_pos = (inputs['input_ids'] == tok3.mask_token_id).nonzero(as_tuple=True)[1]
    with torch.no_grad():
        out = mlm(**inputs)
    print(f"  logits shape: {out.logits.shape}")
    top5_ids = out.logits[0, mask_pos[0], :].topk(5).indices
    print(f"  Top-5 for [MASK]: {tok3.convert_ids_to_tokens(top5_ids.tolist())}")
else:
    note_skip("MLM init failed")
    print("  API shape: out.logits = (batch, seq_len, vocab_size)")

# ─── 8. AutoModelForQuestionAnswering ────────────────────────────────────────
banner("8. AutoModelForQuestionAnswering — span extraction")
# Returns start_logits and end_logits over all tokens
# argmax(start_logits) .. argmax(end_logits) → answer span

ok_tok, tok4 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, qa = safe(AutoModelForQuestionAnswering.from_pretrained, "prajjwal1/bert-tiny")

if ok_tok and ok_mdl:
    question = "Who founded Hugging Face?"
    context = "Hugging Face was co-founded by Clément Delangue and Julien Chaumond."
    inputs = tok4(question, context, return_tensors="pt")
    with torch.no_grad():
        out = qa(**inputs)
    print(f"  start_logits shape: {out.start_logits.shape}")
    print(f"  end_logits shape  : {out.end_logits.shape}")
    start = out.start_logits.argmax()
    end = out.end_logits.argmax() + 1
    ans_ids = inputs['input_ids'][0][start:end]
    print(f"  Predicted answer: {tok4.decode(ans_ids)!r}")
else:
    note_skip("QA init failed")
    print("  API shape: out.start_logits = out.end_logits = (batch, seq_len)")

# ─── 9. AutoModelForSeq2SeqLM ─────────────────────────────────────────────────
banner("9. AutoModelForSeq2SeqLM — encoder-decoder (T5 style)")
# Encoder processes the source; decoder generates the target token by token
# generate() handles the autoregressive loop automatically

ok_tok, tok5 = safe(AutoTokenizer.from_pretrained, "hf-internal-testing/tiny-random-t5")
ok_mdl, t5 = safe(AutoModelForSeq2SeqLM.from_pretrained, "hf-internal-testing/tiny-random-t5")

if ok_tok and ok_mdl:
    inputs = tok5("translate English to French: Hello world", return_tensors="pt")
    with torch.no_grad():
        # encoder_outputs can be cached; here we just do a single generate()
        generated = t5.generate(**inputs, max_new_tokens=10)
    decoded = tok5.decode(generated[0], skip_special_tokens=True)
    print(f"  Encoder input shape  : {inputs['input_ids'].shape}")
    print(f"  Generated token ids  : {generated[0].tolist()}")
    print(f"  Decoded output       : {decoded!r}")
else:
    note_skip("Seq2Seq init failed")
    print("  API shape: model.generate() returns (batch, gen_len) token ids")

# ─── 10. dtype and device placement ─────────────────────────────────────────
banner("10. dtype and device control")
# from_pretrained accepts torch_dtype and device_map
# fp32 (default) → bf16 cuts memory in half; requires supporting hardware
# device_map="auto" uses accelerate to split across GPUs / CPU / disk

ok_tok, tok6 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, m16 = safe(
    AutoModel.from_pretrained,
    "prajjwal1/bert-tiny",
    torch_dtype=torch.float16,   # half-precision on CPU (demonstration only)
)
if ok_tok and ok_mdl:
    print(f"  Model dtype: {next(m16.parameters()).dtype}")
    print(f"  Model device: {next(m16.parameters()).device}")
    # Cast back to fp32 for numerical safety on CPU
    m16 = m16.to(torch.float32)
    print(f"  After .to(float32): {next(m16.parameters()).dtype}")
else:
    note_skip("dtype demo skipped")

print("\n[DONE] 02.automodel.py complete")

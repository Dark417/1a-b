"""
04.attention_hidden_states.py — Inspecting Internals: Attentions & Hidden States
==================================================================================
output_attentions=True and output_hidden_states=True unlock the full internal
state of the transformer for interpretability, probing, and embedding extraction.

Official docs:
  https://huggingface.co/docs/transformers/main_classes/output
  https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel

Tiny model: prajjwal1/bert-tiny (2 layers, 2 heads, hidden=128)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel

# ─── Load bert-tiny ───────────────────────────────────────────────────────────
banner("Loading bert-tiny")

ok_tok, tok = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, model = safe(AutoModel.from_pretrained, "prajjwal1/bert-tiny")

LIVE = ok_tok and ok_mdl

if LIVE:
    model.eval()
    text = "The cat sat on the mat."
    inputs = tok(text, return_tensors="pt")
    NUM_LAYERS = model.config.num_hidden_layers
    NUM_HEADS = model.config.num_attention_heads
    HIDDEN = model.config.hidden_size
    print(f"  bert-tiny: layers={NUM_LAYERS}, heads={NUM_HEADS}, hidden={HIDDEN}")
    print(f"  Input tokens: {tok.convert_ids_to_tokens(inputs['input_ids'][0].tolist())}")
else:
    note_skip(str(model if not ok_mdl else tok))
    # Build from scratch
    cfg = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, vocab_size=1000)
    model = BertModel(cfg)
    model.eval()
    NUM_LAYERS = cfg.num_hidden_layers
    NUM_HEADS = cfg.num_attention_heads
    HIDDEN = cfg.hidden_size
    # Fake tokenizer
    class FakeTok:
        def __call__(self, text, return_tensors=None):
            ids = torch.randint(1, 1000, (1, 7))
            return type("Enc", (), {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "__getitem__": lambda self, k: getattr(self, k),
            })()
        def convert_ids_to_tokens(self, ids):
            return [f"tok{i}" for i in ids]
    tok = FakeTok()
    inputs = {"input_ids": torch.randint(1, 1000, (1, 7)),
              "attention_mask": torch.ones(1, 7, dtype=torch.long)}
    print(f"  [fallback] layers={NUM_LAYERS}, heads={NUM_HEADS}, hidden={HIDDEN}")

SEQ_LEN = inputs["input_ids"].shape[1] if isinstance(inputs, dict) else inputs['input_ids'].shape[1]

# ─── 1. output_hidden_states=True ────────────────────────────────────────────
banner("1. output_hidden_states=True — all layer representations")
# Returns hidden_states: tuple of (num_layers + 1) tensors
# hidden_states[0]  = embedding layer output (before any transformer block)
# hidden_states[1]  = output of block 1
# hidden_states[-1] = output of final block (== last_hidden_state)
# Shape of each: (batch, seq_len, hidden_size)

with torch.no_grad():
    inp = dict(inputs) if isinstance(inputs, dict) else dict(inputs)
    outputs = model(**inp, output_hidden_states=True)

hs = outputs.hidden_states   # tuple len = num_layers + 1
print(f"  Number of hidden state tensors: {len(hs)} (embedding + {NUM_LAYERS} layers)")
for i, h in enumerate(hs):
    label = "embedding" if i == 0 else f"layer {i}"
    print(f"  hidden_states[{i}] ({label:12}): {tuple(h.shape)}")

# Embedding layer vs final layer difference:
embedding_norm = hs[0].norm(dim=-1).mean().item()
final_norm = hs[-1].norm(dim=-1).mean().item()
print(f"\n  Avg L2 norm — embedding layer: {embedding_norm:.3f}, final layer: {final_norm:.3f}")

# ─── 2. output_attentions=True ───────────────────────────────────────────────
banner("2. output_attentions=True — attention weight matrices")
# Returns attentions: tuple of num_layers tensors
# Shape of each: (batch, num_heads, seq_len, seq_len)
# attentions[l][b, h, i, j] = how much token i attends to token j in layer l, head h
# Note: values are AFTER softmax, so they sum to 1 along the last dim.

with torch.no_grad():
    outputs_att = model(**inp, output_attentions=True)

atts = outputs_att.attentions   # tuple len = num_layers
print(f"  Number of attention tensors: {len(atts)} (one per layer)")
for l, att in enumerate(atts):
    print(f"  attentions[{l}] shape: {tuple(att.shape)}  "
          f"# (batch={att.shape[0]}, heads={att.shape[1]}, "
          f"seq={att.shape[2]}, seq={att.shape[3]})")
    # Verify attention weights sum to 1 along last dim
    att_sum = att[0, 0].sum(dim=-1)  # (seq_len,)
    print(f"    Row sums (batch=0, head=0): {att_sum.tolist()}")

# ─── 3. Both flags together ───────────────────────────────────────────────────
banner("3. Both output_hidden_states + output_attentions")

with torch.no_grad():
    full_out = model(**inp, output_hidden_states=True, output_attentions=True)

print(f"  hidden_states count : {len(full_out.hidden_states)}")
print(f"  attentions count    : {len(full_out.attentions)}")
print(f"  last_hidden_state   : {full_out.last_hidden_state.shape}")

# ─── 4. CLS embedding extraction ─────────────────────────────────────────────
banner("4. CLS Token Embedding — sentence representation")
# BERT's [CLS] token is trained to aggregate sentence-level information.
# Useful as a sentence embedding for classification, clustering, or similarity.

cls_emb = full_out.last_hidden_state[:, 0, :]   # (batch, hidden_size)
print(f"  CLS embedding shape  : {cls_emb.shape}")
print(f"  CLS embedding norm   : {cls_emb.norm(dim=-1).item():.4f}")

# Compare CLS across all layers
print(f"\n  CLS norm per layer:")
for i, h in enumerate(full_out.hidden_states):
    cls_norm = h[:, 0, :].norm(dim=-1).item()
    label = "emb" if i == 0 else f"L{i:2d}"
    print(f"    [{label}] {cls_norm:.4f}")

# ─── 5. Mean-pool embedding ───────────────────────────────────────────────────
banner("5. Mean-Pool Embedding — averaging non-padding tokens")
# More robust than CLS for some tasks (e.g., sentence-transformers use this).
# Must mask padding tokens to avoid including their zeros in the average.

if isinstance(inputs, dict):
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (batch, seq, 1)
else:
    mask = inputs["attention_mask"].unsqueeze(-1).float()

last_hs = full_out.last_hidden_state       # (batch, seq, hidden)
summed = (last_hs * mask).sum(dim=1)       # (batch, hidden)
counts = mask.sum(dim=1)                   # (batch, 1)
mean_emb = summed / counts                 # (batch, hidden)

print(f"  Mean-pool embedding shape: {mean_emb.shape}")
print(f"  Mean-pool embedding norm : {mean_emb.norm(dim=-1).item():.4f}")

# Cosine similarity between CLS and mean-pool
cos_sim = torch.nn.functional.cosine_similarity(cls_emb, mean_emb).item()
print(f"  Cosine(CLS, mean_pool)   : {cos_sim:.4f}")

# ─── 6. Per-head attention pattern ───────────────────────────────────────────
banner("6. Per-Head Attention Analysis")
# Different heads attend to different linguistic patterns.
# We can inspect which tokens each head focuses on.

layer0_att = full_out.attentions[0]   # (batch, heads, seq, seq)
print(f"  Layer 0 attention shape: {tuple(layer0_att.shape)}")

for h in range(NUM_HEADS):
    # Average attention FROM all positions TO each token (column sum)
    attn_to = layer0_att[0, h].mean(dim=0)   # (seq,)
    max_attended_pos = attn_to.argmax().item()
    print(f"  Head {h}: most attended position = {max_attended_pos}, "
          f"attention = {attn_to[max_attended_pos]:.3f}")

# ─── 7. Hidden state evolution across layers ─────────────────────────────────
banner("7. Hidden State Evolution Across Layers")
# The residual stream grows richer as we go deeper.
# We can measure how much each layer changes the representation.

print("  Layer-to-layer L2 distance (how much each layer changes the rep):")
for i in range(1, len(full_out.hidden_states)):
    prev = full_out.hidden_states[i - 1]
    curr = full_out.hidden_states[i]
    delta = (curr - prev).norm(dim=-1).mean().item()
    label = "emb→L1" if i == 1 else f"L{i-1}→L{i}"
    print(f"  {label:8}: {delta:.4f}")

print("\n[DONE] 04.attention_hidden_states.py complete")

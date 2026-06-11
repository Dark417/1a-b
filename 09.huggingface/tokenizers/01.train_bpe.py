"""
01.train_bpe.py — Train a BPE tokenizer from scratch on a local corpus.

Covers:
- Tokenizer(BPE()) construction
- BpeTrainer with special tokens and vocab_size
- train_from_iterator (no file needed, no download)
- encode / decode
- Encoding attributes: tokens, ids, offsets
- save to JSON, reload from JSON

Docs: https://huggingface.co/docs/tokenizers/training_from_memory
      https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE
      https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer
"""

import random
import os
import sys
import tempfile

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.normalizers import Lowercase, Sequence as NormSequence
from tokenizers.decoders import BPEDecoder

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Corpus (entirely in-memory, no download)
# ---------------------------------------------------------------------------
CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning is a subset of artificial intelligence",
    "natural language processing enables computers to understand text",
    "deep learning models require large amounts of training data",
    "transformers revolutionized the field of natural language processing",
    "bert uses bidirectional attention for language understanding",
    "gpt models generate text autoregressively one token at a time",
    "fine tuning adapts pretrained models to downstream tasks",
    "tokenization splits text into subword units for the model",
    "embeddings map tokens to dense vector representations in space",
    "the attention mechanism computes weighted sums over all positions",
    "layer normalization stabilizes training in deep neural networks",
    "dropout regularization prevents overfitting in machine learning",
    "gradient descent optimizes model parameters using backpropagation",
    "residual connections allow gradients to flow through deep layers",
    "byte pair encoding merges the most frequent character pairs first",
    "wordpiece maximizes likelihood when selecting subword merge operations",
    "unigram language model prunes tokens that increase corpus loss least",
    "special tokens mark the beginning and end of sequences for models",
    "padding ensures all sequences in a batch have the same length",
]


# ---------------------------------------------------------------------------
# 1. Build and train a BPE tokenizer
# ---------------------------------------------------------------------------
banner("1. Tokenizer(BPE()) — construction")

# BPE(unk_token=...) sets the unknown token for words not in vocab
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

print("Model type     :", type(tokenizer.model).__name__)
print("Tokenizer type :", type(tokenizer).__name__)


# ---------------------------------------------------------------------------
# 2. Normalizer + Pre-tokenizer
# ---------------------------------------------------------------------------
banner("2. Normalizer + Pre-tokenizer")

tokenizer.normalizer = NormSequence([Lowercase()])
tokenizer.pre_tokenizer = Whitespace()

print("Normalizer   :", tokenizer.normalizer)
print("Pre-tokenizer:", tokenizer.pre_tokenizer)
print("Pre-tok test :", tokenizer.pre_tokenizer.pre_tokenize_str("Hello World, it's NLP!"))


# ---------------------------------------------------------------------------
# 3. BpeTrainer
# ---------------------------------------------------------------------------
banner("3. BpeTrainer — configuration")

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

trainer = BpeTrainer(
    vocab_size=200,           # small for our tiny corpus
    min_frequency=1,          # include all pairs seen at least once
    special_tokens=SPECIAL_TOKENS,
    show_progress=False,
)
print("Vocab size target:", trainer.vocab_size)
print("Min frequency    :", trainer.min_frequency)
print("Special tokens   :", trainer.special_tokens)


# ---------------------------------------------------------------------------
# 4. train_from_iterator
# ---------------------------------------------------------------------------
banner("4. train_from_iterator(corpus)")

tokenizer.train_from_iterator(CORPUS, trainer=trainer)

actual_vocab_size = tokenizer.get_vocab_size()
print("Actual vocab size after training:", actual_vocab_size)

# Check special token IDs
for tok in SPECIAL_TOKENS:
    print(f"  {tok:<8} → id {tokenizer.token_to_id(tok)}")


# ---------------------------------------------------------------------------
# 5. encode — single string
# ---------------------------------------------------------------------------
banner("5. encode() — single string")

test_sentence = "natural language processing enables understanding"
enc = tokenizer.encode(test_sentence)

print("Input    :", test_sentence)
print("tokens   :", enc.tokens)
print("ids      :", enc.ids)
print("offsets  :", enc.offsets)
print("n_tokens :", len(enc.tokens))

# Verify offsets align with original string
print("\nOffset verification:")
for token, (start, end) in zip(enc.tokens, enc.offsets):
    original = test_sentence[start:end]
    print(f"  token={token!r:15} offsets=({start},{end}) original={original!r}")


# ---------------------------------------------------------------------------
# 6. encode a pair (text_a, text_b) — type_ids
# ---------------------------------------------------------------------------
banner("6. encode() — sentence pair")

text_a = "deep learning"
text_b = "machine learning"
enc_pair = tokenizer.encode(text_a, text_b)

print("Pair encode:")
print("  tokens  :", enc_pair.tokens)
print("  ids     :", enc_pair.ids)
print("  type_ids:", enc_pair.type_ids)  # 0 = first seq, 1 = second seq


# ---------------------------------------------------------------------------
# 7. decode
# ---------------------------------------------------------------------------
banner("7. decode() — ids back to string")

decoded = tokenizer.decode(enc.ids)
print("Decoded:", decoded)
print("Original:", test_sentence)

# Batch decode
batch_decoded = tokenizer.decode_batch([enc.ids, enc_pair.ids])
print("Batch decoded[0]:", batch_decoded[0])
print("Batch decoded[1]:", batch_decoded[1])


# ---------------------------------------------------------------------------
# 8. Decoder
# ---------------------------------------------------------------------------
banner("8. BPEDecoder — explicit decoder")

tokenizer.decoder = BPEDecoder()
enc2 = tokenizer.encode("tokenization splits text")
decoded2 = tokenizer.decode(enc2.ids)
print("tokens :", enc2.tokens)
print("decoded:", decoded2)


# ---------------------------------------------------------------------------
# 9. Vocabulary inspection
# ---------------------------------------------------------------------------
banner("9. Vocabulary inspection")

vocab = tokenizer.get_vocab()
print(f"Total vocab entries: {len(vocab)}")

# Show first 20 entries sorted by id
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
print("First 20 tokens:")
for tok, idx in sorted_vocab[:20]:
    print(f"  {idx:4d} : {tok!r}")

# Look up specific tokens
sample_word = "learning"
tok_id = tokenizer.token_to_id(sample_word)
print(f"\ntoken_to_id({sample_word!r}) = {tok_id}")
if tok_id is not None:
    print(f"id_to_token({tok_id}) = {tokenizer.id_to_token(tok_id)!r}")


# ---------------------------------------------------------------------------
# 10. Save to JSON and reload
# ---------------------------------------------------------------------------
banner("10. save() / from_file()")

with tempfile.TemporaryDirectory() as tmpdir:
    json_path = os.path.join(tmpdir, "bpe_tokenizer.json")

    tokenizer.save(json_path)
    print("Saved to:", json_path)
    print("File size:", os.path.getsize(json_path), "bytes")

    # Peek at the JSON structure
    import json
    with open(json_path) as f:
        data = json.load(f)
    print("JSON top-level keys:", list(data.keys()))
    print("model.type:", data["model"]["type"])
    print("Vocab entries in JSON:", len(data["model"]["vocab"]))

    # Reload
    tok_reloaded = Tokenizer.from_file(json_path)
    enc_orig     = tokenizer.encode("deep learning models")
    enc_reload   = tok_reloaded.encode("deep learning models")
    print("\nOriginal ids :", enc_orig.ids)
    print("Reloaded ids :", enc_reload.ids)
    assert enc_orig.ids == enc_reload.ids, "Save/reload mismatch!"
    print("Save/reload: MATCH OK")

print("\nAll done — 01.train_bpe.py EXIT 0")

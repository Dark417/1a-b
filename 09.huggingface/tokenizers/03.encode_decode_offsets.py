"""
03.encode_decode_offsets.py — Deep dive on the Encoding object.

Covers:
- Encoding attributes: ids, tokens, attention_mask, type_ids, offsets, word_ids
- char↔token alignment: char_to_token, token_to_chars, word_to_tokens, token_to_word
- enable_padding / enable_truncation
- batch encode (encode_batch)
- Loading a tokenizer: try bert-base-uncased tokenizer.json from transformers cache;
  on failure, train a tiny local WordPiece tokenizer so all examples still run.

Docs: https://huggingface.co/docs/tokenizers/api/encoding
      https://huggingface.co/docs/tokenizers/api/tokenizer#tokenizers.Tokenizer.enable_padding
      https://huggingface.co/docs/tokenizers/api/tokenizer#tokenizers.Tokenizer.enable_truncation
      https://huggingface.co/docs/tokenizers/components#post-processors
"""

import random
import os
import sys
import tempfile
import json

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import WordPiece as WordPieceDecoder

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

random.seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# Helper: train a minimal local WordPiece tokenizer
# ---------------------------------------------------------------------------
CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world this is a test sentence for tokenization",
    "machine learning natural language processing deep learning",
    "bert uses bidirectional attention for language understanding",
    "transformers revolutionized the field of nlp research",
    "tokenization splits text into subword units for models",
    "embeddings map tokens to dense vector representations space",
    "the quick brown fox is a common english pangram sentence",
    "deep neural networks learn hierarchical feature representations",
    "pretrained language models are fine tuned on downstream tasks",
    "special tokens mark boundaries of sentences in bert models",
    "padding and truncation handle variable length input sequences",
    "offsets track character positions of each token in source text",
    "word ids group subword tokens back to their original words",
    "char to token and token to chars enable span extraction tasks",
]


def build_local_tokenizer():
    """Train a tiny WordPiece tokenizer from CORPUS. Always succeeds."""
    SPECIAL = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.normalizer = BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=False, lowercase=True
    )
    tok.pre_tokenizer = BertPreTokenizer()
    tok.decoder = WordPieceDecoder(prefix="##", cleanup=True)

    trainer = WordPieceTrainer(
        vocab_size=500, min_frequency=1, special_tokens=SPECIAL, show_progress=False
    )
    tok.train_from_iterator(CORPUS, trainer=trainer)

    CLS_ID = tok.token_to_id("[CLS]")
    SEP_ID = tok.token_to_id("[SEP]")
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", CLS_ID), ("[SEP]", SEP_ID)],
    )
    return tok


# ---------------------------------------------------------------------------
# 1. Get a tokenizer — try transformers cache first, fall back to local
# ---------------------------------------------------------------------------
banner("1. Load tokenizer — try transformers / fallback to local")

def try_load_bert_tokenizer():
    """Try to get bert-base-uncased tokenizer via transformers (cached or hub)."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("bert-base-uncased")
    return t.backend_tokenizer   # raw tokenizers.Tokenizer object


ok, result = safe(try_load_bert_tokenizer)
if ok:
    tokenizer = result
    print("LIVE: Loaded bert-base-uncased from transformers cache/hub")
    VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Vocab size: {VOCAB_SIZE}")
else:
    note_skip(f"needs network — using local fallback ({result})")
    tokenizer = build_local_tokenizer()
    VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"LOCAL: Trained tiny WordPiece, vocab size: {VOCAB_SIZE}")


# ---------------------------------------------------------------------------
# 2. Encoding attributes — the full picture
# ---------------------------------------------------------------------------
banner("2. Encoding attributes — full tour")

sentence = "The quick brown fox jumps over the lazy dog"
enc = tokenizer.encode(sentence)

print(f"Input sentence : {sentence!r}")
print(f"n_tokens       : {len(enc.ids)}")
print()
print(f"ids            : {enc.ids}")
print(f"tokens         : {enc.tokens}")
print(f"attention_mask : {enc.attention_mask}")
print(f"type_ids       : {enc.type_ids}")
print(f"offsets        : {enc.offsets}")
print(f"word_ids       : {enc.word_ids}")
print()

# Table view
print(f"{'idx':<5} {'token':<15} {'id':<6} {'attn':<5} {'type':<5} {'offset':<12} {'word_id'}")
print("-" * 65)
for i, (token, tid, attn, ttype, offset, wid) in enumerate(zip(
    enc.tokens, enc.ids, enc.attention_mask,
    enc.type_ids, enc.offsets, enc.word_ids
)):
    wid_str = str(wid) if wid is not None else "None"
    print(f"{i:<5} {token:<15} {tid:<6} {attn:<5} {ttype:<5} {str(offset):<12} {wid_str}")


# ---------------------------------------------------------------------------
# 3. word_ids — grouping subword tokens back to words
# ---------------------------------------------------------------------------
banner("3. word_ids — subword → word grouping")

# Use a sentence with a word likely to be split
split_sentence = "tokenization and subword representations"
enc2 = tokenizer.encode(split_sentence)

print("Sentence:", split_sentence)
print("Tokens  :", enc2.tokens)
print("word_ids:", enc2.word_ids)
print()

# Group tokens by word_id
from collections import defaultdict
word_groups = defaultdict(list)
for i, wid in enumerate(enc2.word_ids):
    if wid is not None:
        word_groups[wid].append(enc2.tokens[i])

print("Word → subword tokens:")
for wid in sorted(word_groups):
    subtokens = word_groups[wid]
    print(f"  Word {wid}: {subtokens}")


# ---------------------------------------------------------------------------
# 4. char_to_token — character index → token index
# ---------------------------------------------------------------------------
banner("4. char_to_token(sequence_index, char_pos)")

reference = "Hello world"
enc3 = tokenizer.encode(reference)
print("Sentence:", reference)
print("Tokens  :", enc3.tokens)
print("Offsets :", enc3.offsets)
print()

# For every character, find which token contains it
print("Character → token mapping:")
for char_idx in range(len(reference)):
    char = reference[char_idx]
    tok_idx = enc3.char_to_token(0, char_idx)
    tok_str = enc3.tokens[tok_idx] if tok_idx is not None else "N/A"
    print(f"  char[{char_idx}]={char!r:3}  → token_idx={tok_idx}  token={tok_str!r}")


# ---------------------------------------------------------------------------
# 5. token_to_chars — token index → (start_char, end_char)
# ---------------------------------------------------------------------------
banner("5. token_to_chars(sequence_index, token_index)")

print("Sentence:", reference)
print("Tokens  :", enc3.tokens)
print()
print("Token → character span:")
for tok_idx, token in enumerate(enc3.tokens):
    char_span = enc3.token_to_chars(0, tok_idx)
    if char_span is not None:
        s, e = char_span
        original = reference[s:e]
        print(f"  token[{tok_idx}]={token!r:15}  → chars[{s}:{e}] = {original!r}")
    else:
        print(f"  token[{tok_idx}]={token!r:15}  → chars=None (special token)")


# ---------------------------------------------------------------------------
# 6. word_to_tokens / token_to_word
# ---------------------------------------------------------------------------
banner("6. word_to_tokens / token_to_word")

words = "deep learning models"
enc4 = tokenizer.encode(words)
print("Sentence:", words)
print("Tokens  :", enc4.tokens)
print("word_ids:", enc4.word_ids)
print()

# Count unique non-None word_ids
n_words = len(set(wid for wid in enc4.word_ids if wid is not None))
print(f"Number of source words: {n_words}")

print("word_to_tokens(seq=0, word_idx):")
for word_idx in range(n_words):
    span = enc4.word_to_tokens(0, word_idx)
    if span:
        token_start, token_end = span
        word_tokens = enc4.tokens[token_start:token_end]
        print(f"  word {word_idx} → tokens[{token_start}:{token_end}] = {word_tokens}")

print("\ntoken_to_word(seq=0, token_idx):")
for tok_idx, token in enumerate(enc4.tokens):
    wid = enc4.token_to_word(0, tok_idx)
    print(f"  token[{tok_idx}]={token!r:15} → word_id={wid}")


# ---------------------------------------------------------------------------
# 7. enable_padding
# ---------------------------------------------------------------------------
banner("7. enable_padding()")

tokenizer.enable_padding(
    direction="right",
    pad_id=tokenizer.token_to_id("[PAD]") or 0,
    pad_type_id=0,
    pad_token="[PAD]",
    length=20,       # pad all sequences to length 20
)

short_sentences = [
    "hello",
    "machine learning is great",
    "tokenization",
]
enc_padded = tokenizer.encode_batch(short_sentences)

print("Padding to length 20:")
for i, (sent, enc) in enumerate(zip(short_sentences, enc_padded)):
    print(f"\n  [{i}] {sent!r}")
    print(f"       tokens: {enc.tokens}")
    print(f"       ids   : {enc.ids}")
    print(f"       attn  : {enc.attention_mask}")
    assert len(enc.ids) == 20, f"Expected 20, got {len(enc.ids)}"
print("All padded to length 20: OK")

# Remove padding setting
tokenizer.no_padding()


# ---------------------------------------------------------------------------
# 8. enable_truncation
# ---------------------------------------------------------------------------
banner("8. enable_truncation(max_length=10)")

tokenizer.enable_truncation(max_length=10)

long_sentences = [
    "the quick brown fox jumps over the lazy dog and then some",
    "natural language processing is a fascinating subfield of artificial intelligence research",
]
print("Truncating to max_length=10:")
for sent in long_sentences:
    enc_t = tokenizer.encode(sent)
    print(f"\n  Input  : {sent!r}")
    print(f"  tokens : {enc_t.tokens}")
    print(f"  n_toks : {len(enc_t.tokens)}")
    assert len(enc_t.ids) <= 10

tokenizer.no_truncation()


# ---------------------------------------------------------------------------
# 9. Padding + Truncation together (batch encode)
# ---------------------------------------------------------------------------
banner("9. Padding + Truncation together (batch_encode)")

tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("[PAD]") or 0,
    pad_token="[PAD]",
)
tokenizer.enable_truncation(max_length=12)

batch = [
    "hi",
    "deep learning requires lots of data and compute resources",
    "tokenization",
    "transformers are powerful models for nlp tasks and more",
]
encs = tokenizer.encode_batch(batch)

print("Batch encode with truncation (max=12) and dynamic padding:")
for i, (sent, enc) in enumerate(zip(batch, encs)):
    print(f"  [{i}] len={len(enc.ids):3}  tokens={enc.tokens}")

tokenizer.no_padding()
tokenizer.no_truncation()


# ---------------------------------------------------------------------------
# 10. Pair encoding — offsets and type_ids
# ---------------------------------------------------------------------------
banner("10. Pair encoding — type_ids and offsets")

text_a = "what is bert"
text_b = "bert is a language model"

enc_pair = tokenizer.encode(text_a, text_b)

print("Text A:", text_a)
print("Text B:", text_b)
print()
print(f"{'idx':<5} {'token':<15} {'type_id':<9} {'offset':<14} {'word_id'}")
print("-" * 60)
for i, (token, ttype, offset, wid) in enumerate(zip(
    enc_pair.tokens, enc_pair.type_ids, enc_pair.offsets, enc_pair.word_ids
)):
    wid_str = str(wid) if wid is not None else "None"
    print(f"{i:<5} {token:<15} {ttype:<9} {str(offset):<14} {wid_str}")

print()
print("type_id 0 = tokens from text_a (+ [CLS] and first [SEP])")
print("type_id 1 = tokens from text_b (+ last [SEP])")


# ---------------------------------------------------------------------------
# 11. Decode
# ---------------------------------------------------------------------------
banner("11. decode() / decode_batch()")

sample_enc = tokenizer.encode("machine learning is fascinating")
decoded = tokenizer.decode(sample_enc.ids)
print("Encoded tokens :", sample_enc.tokens)
print("Encoded ids    :", sample_enc.ids)
print("Decoded string :", decoded)

# Batch decode
ids_batch = [tokenizer.encode(s).ids for s in ["hello world", "deep learning"]]
decoded_batch = tokenizer.decode_batch(ids_batch)
print("\nBatch decoded:")
for d in decoded_batch:
    print(" ", d)

print("\nAll done — 03.encode_decode_offsets.py EXIT 0")

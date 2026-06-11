"""
02.train_wordpiece.py — Train a BERT-style WordPiece tokenizer from scratch.

Covers:
- WordPiece model with WordPieceTrainer
- BertNormalizer (accent strip, lowercasing, Chinese char splitting)
- BertPreTokenizer (whitespace + punctuation)
- TemplateProcessing post-processor for [CLS] / [SEP] injection
- WordPiece decoder (handles ## continuation prefix)
- encode of (text, pair) and special token handling

Docs: https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.WordPiece
      https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.WordPieceTrainer
      https://huggingface.co/docs/tokenizers/api/post-processors#tokenizers.processors.TemplateProcessing
      https://huggingface.co/course/chapter6/8
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
# Corpus — in-memory, no download
# ---------------------------------------------------------------------------
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand text.",
    "Deep learning models require large amounts of training data.",
    "Transformers revolutionized the field of NLP in 2017.",
    "BERT uses bidirectional attention for language understanding.",
    "GPT models generate text autoregressively one token at a time.",
    "Fine-tuning adapts pretrained models to downstream tasks.",
    "Tokenization splits text into subword units for the model.",
    "Embeddings map tokens to dense vector representations.",
    "The attention mechanism computes weighted sums over positions.",
    "Layer normalization stabilizes training in deep neural networks.",
    "Dropout regularization prevents overfitting in machine learning.",
    "Gradient descent optimizes model parameters via backpropagation.",
    "Residual connections allow gradients to flow through deep layers.",
    "WordPiece maximizes likelihood when selecting subword merges.",
    "Unigram language models prune tokens that increase corpus loss.",
    "Special tokens mark the beginning and end of sequences.",
    "Padding ensures all sequences in a batch have the same length.",
    "Byte-pair encoding merges the most frequent character pairs first.",
    "Pre-training on large corpora learns general language representations.",
    "Question answering requires understanding context and extracting spans.",
    "Named entity recognition labels tokens with entity type tags.",
    "Sentiment analysis classifies text as positive, negative, or neutral.",
    "Text summarization condenses long documents into shorter versions.",
    "Machine translation converts text from one language to another.",
    "Semantic similarity measures how close two sentences are in meaning.",
    "Word embeddings capture semantic relationships between vocabulary items.",
    "Subword tokenization balances vocabulary size and coverage of text.",
    "Positional encodings provide the model with sequence order information.",
]


# ---------------------------------------------------------------------------
# 1. Build the tokenizer skeleton
# ---------------------------------------------------------------------------
banner("1. Tokenizer(WordPiece()) — construction")

# WordPiece(unk_token=...) handles OOV tokens
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
print("Model type:", type(tokenizer.model).__name__)


# ---------------------------------------------------------------------------
# 2. BertNormalizer
# ---------------------------------------------------------------------------
banner("2. BertNormalizer")

tokenizer.normalizer = BertNormalizer(
    clean_text=True,          # remove control chars, normalize whitespace
    handle_chinese_chars=True,# add spaces around CJK chars
    strip_accents=True,       # é → e (good for multilingual)
    lowercase=True,           # fold to lowercase
)
print("Normalizer:", tokenizer.normalizer)

# Demonstrate normalization
test_strings = [
    "Héllo Wörld",
    "NLP is GREAT!!!",
    "中文字符 tokenization",
    "café au lait",
]
print("\nNormalization examples:")
for s in test_strings:
    normalized = tokenizer.normalizer.normalize_str(s)
    print(f"  {s!r:30} → {normalized!r}")


# ---------------------------------------------------------------------------
# 3. BertPreTokenizer
# ---------------------------------------------------------------------------
banner("3. BertPreTokenizer")

tokenizer.pre_tokenizer = BertPreTokenizer()
print("Pre-tokenizer:", tokenizer.pre_tokenizer)

# BertPreTokenizer splits on whitespace AND punctuation
demo_texts = [
    "don't stop me now",
    "it's a fine-tuned model",
    "score: 0.95 (great!)",
]
print("\nPre-tokenization examples:")
for t in demo_texts:
    split = tokenizer.pre_tokenizer.pre_tokenize_str(t)
    print(f"  {t!r}")
    print(f"    → {split}")


# ---------------------------------------------------------------------------
# 4. WordPieceTrainer
# ---------------------------------------------------------------------------
banner("4. WordPieceTrainer — configuration + training")

trainer = WordPieceTrainer(
    vocab_size=300,
    min_frequency=1,
    special_tokens=SPECIAL_TOKENS,
    continuing_subword_prefix="##",   # BERT convention
    show_progress=False,
)
print("Vocab size target  :", trainer.vocab_size)
print("Continuing prefix  :", trainer.continuing_subword_prefix)
print("Special tokens     :", trainer.special_tokens)

tokenizer.train_from_iterator(CORPUS, trainer=trainer)

actual_vocab = tokenizer.get_vocab_size()
print("Actual vocab size  :", actual_vocab)
print("Special token IDs  :")
for tok in SPECIAL_TOKENS:
    print(f"  {tok:<8} → id {tokenizer.token_to_id(tok)}")


# ---------------------------------------------------------------------------
# 5. WordPiece decoder
# ---------------------------------------------------------------------------
banner("5. WordPiece decoder — ## continuation handling")

tokenizer.decoder = WordPieceDecoder(prefix="##", cleanup=True)
print("Decoder:", tokenizer.decoder)

enc_test = tokenizer.encode("tokenization is important")
print("\nEncode 'tokenization is important':")
print("  tokens:", enc_test.tokens)
decoded_test = tokenizer.decode(enc_test.ids)
print("  decoded:", decoded_test)


# ---------------------------------------------------------------------------
# 6. TemplateProcessing post-processor (BERT-style)
# ---------------------------------------------------------------------------
banner("6. TemplateProcessing — [CLS] sentence [SEP]")

CLS_ID = tokenizer.token_to_id("[CLS]")
SEP_ID = tokenizer.token_to_id("[SEP]")
print(f"[CLS] id = {CLS_ID},  [SEP] id = {SEP_ID}")

tokenizer.post_processor = TemplateProcessing(
    # Single sentence: [CLS] A [SEP]
    single="[CLS] $A [SEP]",
    # Sentence pair: [CLS] A [SEP] B [SEP]
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", CLS_ID),
        ("[SEP]", SEP_ID),
    ],
)
print("Post-processor set.")


# ---------------------------------------------------------------------------
# 7. encode — single sentence
# ---------------------------------------------------------------------------
banner("7. encode() — single sentence")

sentence = "deep learning requires large training data"
enc = tokenizer.encode(sentence)

print("Input   :", sentence)
print("tokens  :", enc.tokens)
print("ids     :", enc.ids)
print("offsets :", enc.offsets)
print("type_ids:", enc.type_ids)
print("attn_mask:", enc.attention_mask)
print(f"[CLS] at position 0: {enc.tokens[0] == '[CLS]'}")
print(f"[SEP] at last position: {enc.tokens[-1] == '[SEP]'}")


# ---------------------------------------------------------------------------
# 8. encode pair — (text_a, text_b)
# ---------------------------------------------------------------------------
banner("8. encode() — sentence pair (text_a, text_b)")

text_a = "what is natural language processing"
text_b = "it is a branch of artificial intelligence"

enc_pair = tokenizer.encode(text_a, text_b)

print("Sentence A:", text_a)
print("Sentence B:", text_b)
print()
print("tokens   :", enc_pair.tokens)
print("ids      :", enc_pair.ids)
print("type_ids :", enc_pair.type_ids)   # 0 for A tokens, 1 for B tokens
print("attn_mask:", enc_pair.attention_mask)
print()

# Verify: segment 0 ends at first [SEP], segment 1 starts after
cls_sep_positions = [i for i, t in enumerate(enc_pair.tokens) if t in ("[CLS]", "[SEP]")]
print("Special token positions (CLS/SEP):", cls_sep_positions)
print("Type IDs at these positions:", [enc_pair.type_ids[i] for i in cls_sep_positions])


# ---------------------------------------------------------------------------
# 9. ## continuation tokens
# ---------------------------------------------------------------------------
banner("9. WordPiece ## continuation tokens")

# Words that are likely to be split
long_words = [
    "tokenization",
    "representations",
    "understanding",
    "backpropagation",
    "regularization",
]
print("WordPiece subword splits (no [CLS]/[SEP] here for clarity):")
# Temporarily remove post-processor to see raw splits
saved_pp = tokenizer.post_processor
tokenizer.post_processor = None

for word in long_words:
    enc_w = tokenizer.encode(word)
    print(f"  {word!r:20} → {enc_w.tokens}")

tokenizer.post_processor = saved_pp


# ---------------------------------------------------------------------------
# 10. Vocabulary inspection
# ---------------------------------------------------------------------------
banner("10. Vocabulary — ## tokens vs whole-word tokens")

vocab = tokenizer.get_vocab()
whole_words = [t for t in vocab if not t.startswith("##") and t not in SPECIAL_TOKENS]
cont_subwords = [t for t in vocab if t.startswith("##")]

print(f"Total vocab     : {len(vocab)}")
print(f"Special tokens  : {len(SPECIAL_TOKENS)}")
print(f"Whole-word tokens: {len(whole_words)}")
print(f"## continuation : {len(cont_subwords)}")

print("\nSample ## continuation tokens:")
print(" ", sorted(cont_subwords)[:20])


# ---------------------------------------------------------------------------
# 11. Save and reload
# ---------------------------------------------------------------------------
banner("11. save() / from_file() round-trip")

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "wordpiece_tokenizer.json")
    tokenizer.save(path)
    print("Saved to:", path)

    with open(path) as f:
        data = json.load(f)
    print("JSON keys    :", list(data.keys()))
    print("model type   :", data["model"]["type"])
    print("unk_token    :", data["model"]["unk_token"])
    print("vocab size   :", len(data["model"]["vocab"]))
    print("post_processor type:", data.get("post_processor", {}).get("type", "None"))

    tok2 = Tokenizer.from_file(path)
    enc_a = tokenizer.encode(sentence)
    enc_b = tok2.encode(sentence)
    assert enc_a.ids == enc_b.ids, "Save/reload mismatch!"
    print("Save/reload: MATCH OK")

print("\nAll done — 02.train_wordpiece.py EXIT 0")

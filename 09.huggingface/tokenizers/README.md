# HuggingFace `tokenizers` Library

> Official docs: https://huggingface.co/docs/tokenizers/index

---

## What is `tokenizers`?

`tokenizers` is HuggingFace's Rust-backed library for fast, production-grade tokenization.
It exposes Python bindings over a Rust core, giving **10–100× speedups** versus pure-Python
tokenizers while remaining fully flexible.

Key capabilities:
- Train tokenizers from scratch (BPE, WordPiece, Unigram/SentencePiece, WordLevel).
- Serialize to / load from a single JSON file (portable across languages).
- Rich `Encoding` objects with tokens, ids, offsets, type_ids, word_ids.
- Character ↔ token alignment for span extraction (NER, QA, highlighting).
- Batch encoding with automatic padding and truncation.
- Used internally by `transformers` for every "fast" tokenizer variant.

---

## The Tokenizer Pipeline

```
Raw string
    │
    ▼
┌─────────────┐
│ Normalizer  │  Unicode normalization, lowercasing, accent stripping, …
└─────────────┘
    │
    ▼
┌──────────────────┐
│ Pre-Tokenizer    │  Split into "words" before subword algo runs
│ (Whitespace,     │  e.g. "don't" → ["don", "'", "t"]
│  ByteLevel, …)   │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Model            │  Core algorithm: BPE / WordPiece / Unigram / WordLevel
│                  │  Maps word → subword token IDs
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Post-Processor   │  Add special tokens, segment IDs, etc.
│ (TemplateProc.,  │  e.g. [CLS] … [SEP] for BERT
│  RobertaProc.)   │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│ Decoder          │  Reconstruct original string from tokens
│ (BPE, WP, …)     │  Handles merge artifacts (Ġ, ##, ▁)
└──────────────────┘
    │
    ▼
Tokens / IDs / Offsets
```

### Normalizers

| Normalizer | Effect |
|---|---|
| `NFD`, `NFC`, `NFKD`, `NFKC` | Unicode normalization forms |
| `Lowercase` | Fold to lowercase |
| `StripAccents` | Remove diacritics (é → e) |
| `BertNormalizer` | Chinese char splitting, accent strip, lowercase |
| `Sequence([...])` | Chain multiple normalizers |

### Pre-Tokenizers

| Pre-Tokenizer | Splits on |
|---|---|
| `Whitespace` | `\w+` and non-`\w` groups |
| `WhitespaceSplit` | Only whitespace |
| `ByteLevel` | Every byte → Unicode escape; used by GPT-2 |
| `BertPreTokenizer` | Whitespace + punctuation, handles Chinese |
| `Punctuation` | Individual punctuation chars |
| `Metaspace` | Space as prefix `▁` (SentencePiece-style) |

---

## Subword Algorithms Compared

### BPE — Byte-Pair Encoding

**Origin**: Originally a data compression algorithm; popularized for NLP by Sennrich et al. (2015).

**Algorithm**:
1. Initialize vocab = all characters (or bytes) in training corpus.
2. Count all adjacent pair frequencies.
3. Merge the most frequent pair into a new token.
4. Repeat until vocabulary size reached.

**Properties**:
- Deterministic, frequency-driven merges.
- Handles OOV via byte-level fallback (GPT-2 style).
- No UNK token needed with ByteLevel pre-tokenizer.
- Merge rules are stored as an ordered list.

```
"lower" → "l" "o" "w" "e" "r"  (initial)
         → "lo" "w" "er"        (after merging l+o, e+r)
         → "low" "er"           (merge lo+w)
```

**Used by**: GPT-2, RoBERTa, BART, LLaMA.

### WordPiece

**Origin**: Schuster & Nakamura (2012); adapted for BERT by Devlin et al. (2018).

**Algorithm**:
1. Initialize vocab = all characters.
2. Score each merge candidate as `freq(AB) / (freq(A) × freq(B))` (maximize likelihood).
3. Add highest-scoring pair to vocab.
4. Prefix continuation subwords with `##`.

**Properties**:
- Likelihood-maximizing (vs frequency-maximizing in BPE).
- `##` prefix distinguishes word-start from word-continuation tokens.
- Unknown tokens → `[UNK]` if char not in vocab.

```
"playing" → "play" + "##ing"
"unaffable" → "un" + "##aff" + "##able"
```

**Used by**: BERT, DistilBERT, ELECTRA, MobileBERT.

### Unigram / SentencePiece

**Origin**: Kudo (2018); implemented in the `sentencepiece` library.

**Algorithm**:
1. Start with a large initial vocabulary (all substrings up to a length).
2. Compute unigram language model probabilities.
3. Prune tokens that increase loss the least until target vocab size.

**Properties**:
- Probabilistic model; can return multiple segmentations with probabilities.
- Language-agnostic (no whitespace assumption; uses `▁` for space).
- Trained end-to-end from raw text.
- More robust to spelling variations.

**Used by**: T5, ALBERT, XLNet, mBART, LLaMA (SentencePiece BPE variant).

### WordLevel

The simplest model: a direct word → ID mapping.
- Splits on whitespace/punctuation.
- Unknown words → `[UNK]`.
- Vocabulary can be very large.
- No subword decomposition.

**Used by**: Simple baselines, character-level models.

---

## Special Tokens

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tok = Tokenizer(BPE())
# Add special tokens before training
special = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
# Pass to trainer:
trainer = BpeTrainer(special_tokens=special)
```

After training, access via:
```python
tok.token_to_id("[CLS]")   # → 0 (index in special_tokens list)
tok.id_to_token(0)         # → "[CLS]"
```

---

## Encoding Objects

`tokenizer.encode("text")` returns an `Encoding` object:

```python
enc = tokenizer.encode("Hello world")
enc.ids            # [101, 7592, 2088, 102]
enc.tokens         # ['[CLS]', 'hello', 'world', '[SEP]']
enc.offsets        # [(0,0), (0,5), (6,11), (0,0)]
enc.type_ids       # [0, 0, 0, 0]     segment IDs
enc.attention_mask # [1, 1, 1, 1]
enc.word_ids       # [None, 0, 1, None]  which word each token came from
```

### Character ↔ Token Alignment

```python
enc.char_to_token(0, 3)    # char index 3 in sequence 0 → token index
enc.token_to_chars(1)      # token index 1 → (start_char, end_char)
enc.word_to_tokens(0)      # word 0 → (token_start, token_end)
enc.token_to_word(2)       # token 2 → word index
```

---

## Padding and Truncation

```python
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
tokenizer.enable_truncation(max_length=128)
# Now batch_encode handles variable-length sequences automatically
encodings = tokenizer.encode_batch(["short", "a much longer sentence here"])
```

---

## Relation to `transformers` AutoTokenizer

`transformers` provides `AutoTokenizer` which wraps `tokenizers`:

| | `tokenizers` | `transformers.AutoTokenizer` |
|---|---|---|
| Speed | Rust core (fast) | Same Rust core when `use_fast=True` |
| API | Low-level, explicit | High-level, model-aware |
| Serialization | Single JSON | `tokenizer.json` + config files |
| Special tokens | Manual | Loaded from model config |
| Training | `train_from_iterator` | Less common |
| `is_fast` | Always True | True for fast, False for slow (Python) |

```python
# Under the hood, fast tokenizers use the `tokenizers` Rust backend
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")  # uses tokenizers
print(tok.is_fast)      # True
print(tok.backend_tokenizer)   # the raw tokenizers.Tokenizer object
```

---

## Serialization

```python
# Save
tokenizer.save("tokenizer.json")

# Load
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Or from pretrained (downloads tokenizer.json from Hub)
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
```

---

## Full Feature Tour

```python
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, Sequence as NormSeq
from tokenizers.decoders import BPEDecoder

# 1. Build the tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = NormSeq([Lowercase()])
tokenizer.pre_tokenizer = Whitespace()

# 2. Train
trainer = BpeTrainer(
    vocab_size=500,
    min_frequency=1,
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
)
corpus = ["hello world", "deep learning is fun", ...]
tokenizer.train_from_iterator(corpus, trainer=trainer)

# 3. Encode
enc = tokenizer.encode("deep learning")
print(enc.tokens)        # ['deep', 'learn', '##ing']
print(enc.ids)           # [42, 18, 25]
print(enc.offsets)       # [(0,4), (5,10), (10,13)]

# 4. Decode
text = tokenizer.decode([42, 18, 25])   # 'deep learning'

# 5. Batch encode with padding
tokenizer.enable_padding(pad_id=1, pad_token="[PAD]")
batch = tokenizer.encode_batch(["hello", "deep learning rocks today"])
for enc in batch:
    print(enc.ids)

# 6. Save/load
tokenizer.save("my_tokenizer.json")
tokenizer2 = Tokenizer.from_file("my_tokenizer.json")
```

---

## References

- https://huggingface.co/docs/tokenizers/index
- https://huggingface.co/docs/tokenizers/pipeline
- https://huggingface.co/docs/tokenizers/api/tokenizer
- https://huggingface.co/docs/tokenizers/api/models
- https://huggingface.co/docs/tokenizers/api/trainers
- https://huggingface.co/docs/tokenizers/training_from_memory
- https://huggingface.co/course/chapter6
- Sennrich et al. (2015) Neural Machine Translation of Rare Words with Subword Units
- Kudo (2018) Subword Regularization: Improving Neural Network Translation Models
- Devlin et al. (2018) BERT: Pre-training of Deep Bidirectional Transformers

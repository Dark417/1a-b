"""
03.map_filter.py — map, filter, cast, rename, select, shuffle, train_test_split.

Covers:
- map (single + batched=True)
- filter
- cast_column / ClassLabel
- rename_column / remove_columns
- select
- shuffle(seed=)
- train_test_split
- num_proc note
- tokenizer-style batched map (trivial whitespace split, no model download)

Docs: https://huggingface.co/docs/datasets/process
      https://huggingface.co/docs/datasets/package_reference/main_classes
"""

import random
import os
import sys

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Shared tiny corpus
# ---------------------------------------------------------------------------
CORPUS = {
    "text": [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing enables computers to understand text",
        "Deep learning models require large amounts of training data",
        "Transformers revolutionized the field of NLP in 2017",
        "BERT uses bidirectional attention for language understanding",
        "GPT models generate text autoregressively",
        "Fine-tuning adapts pretrained models to downstream tasks",
        "Tokenization splits text into subword units",
        "Embeddings map tokens to dense vector representations",
    ],
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "score": [0.9, 0.8, 0.7, 0.85, 0.6, 0.95, 0.75, 0.88, 0.72, 0.91],
}

ds = Dataset.from_dict(CORPUS)
print("Original dataset:", ds)
print("Features:", ds.features)


# ---------------------------------------------------------------------------
# 1. Single-example map
# ---------------------------------------------------------------------------
banner("1. .map() — single example (add word count)")


def add_word_count(example):
    """Add 'n_words' and 'text_upper' columns."""
    example["n_words"] = len(example["text"].split())
    example["text_upper"] = example["text"].upper()
    return example


ds_mapped = ds.map(add_word_count)
print("New columns:", ds_mapped.column_names)
print("Row 0:", ds_mapped[0])
print("n_words range:", min(ds_mapped["n_words"]), "–", max(ds_mapped["n_words"]))


# ---------------------------------------------------------------------------
# 2. Batched map (faster — vectorised, fewer Python call overheads)
# ---------------------------------------------------------------------------
banner("2. .map(batched=True) — batch-level transform")


def batch_add_char_count(batch):
    """batch is a dict of lists."""
    batch["n_chars"] = [len(t) for t in batch["text"]]
    batch["score_bin"] = ["high" if s >= 0.8 else "low" for s in batch["score"]]
    return batch


ds_batched = ds.map(batch_add_char_count, batched=True, batch_size=4)
print("New columns:", ds_batched.column_names)
print("First 4 rows (n_chars, score_bin):")
for i in range(4):
    row = ds_batched[i]
    print(f"  [{i}] n_chars={row['n_chars']}  score_bin={row['score_bin']}")

# num_proc note
print("""
NOTE on num_proc:
  ds.map(fn, num_proc=4)  — spawns 4 worker processes (multiprocessing).
  Requires fn to be picklable. Lambda functions usually are picklable in
  Python 3.8+ with dill. On Windows wrap calls in if __name__ == '__main__'.
  For CPU-bound transforms (e.g., tokenization) this gives near-linear speedup.
""")


# ---------------------------------------------------------------------------
# 3. filter
# ---------------------------------------------------------------------------
banner("3. .filter() — keep rows satisfying predicate")

ds_pos = ds_mapped.filter(lambda ex: ex["label"] == 1)
print(f"Positive examples ({len(ds_pos)}):")
for row in ds_pos:
    print(f"  label={row['label']}  text={row['text'][:40]}...")

ds_long = ds_mapped.filter(lambda ex: ex["n_words"] > 8)
print(f"\nExamples with >8 words ({len(ds_long)}):")
for row in ds_long:
    print(f"  n_words={row['n_words']}  {row['text'][:40]}...")


# ---------------------------------------------------------------------------
# 4. cast_column / ClassLabel
# ---------------------------------------------------------------------------
banner("4. cast_column() + ClassLabel")

ds_cast = ds.cast_column("label", ClassLabel(names=["tech", "nlp"]))
print("After cast, features:", ds_cast.features)
print("label int→str:", ds_cast.features["label"].int2str(0), ds_cast.features["label"].int2str(1))
print("label str→int:", ds_cast.features["label"].str2int("tech"), ds_cast.features["label"].str2int("nlp"))

# Cast an entire feature schema at once
new_feats = Features({
    "text": Value("string"),
    "label": ClassLabel(names=["tech", "nlp"]),
    "score": Value("float32"),
})
ds_feats = ds.cast(new_feats)
print("Features after full cast:", ds_feats.features)


# ---------------------------------------------------------------------------
# 5. rename_column / remove_columns
# ---------------------------------------------------------------------------
banner("5. rename_column() + remove_columns()")

ds_ren = ds_batched.rename_column("text", "sentence")
print("After rename:", ds_ren.column_names)

ds_clean = ds_ren.remove_columns(["text_upper", "score_bin"])
print("After remove:", ds_clean.column_names)


# ---------------------------------------------------------------------------
# 6. select
# ---------------------------------------------------------------------------
banner("6. .select() — pick rows by index")

ds_small = ds.select([0, 2, 4, 6, 8])
print(f"Selected {len(ds_small)} rows:", ds_small["text"])

# select via range
ds_first5 = ds.select(range(5))
print("First 5 labels:", ds_first5["label"])


# ---------------------------------------------------------------------------
# 7. shuffle
# ---------------------------------------------------------------------------
banner("7. .shuffle(seed=)")

ds_shuf = ds.shuffle(seed=42)
print("Original labels:", ds["label"])
print("Shuffled labels :", ds_shuf["label"])
# Reproducible
ds_shuf2 = ds.shuffle(seed=42)
print("Same seed again :", ds_shuf2["label"])
assert ds_shuf["label"] == ds_shuf2["label"], "Shuffle should be deterministic!"
print("Reproducibility: OK")


# ---------------------------------------------------------------------------
# 8. train_test_split
# ---------------------------------------------------------------------------
banner("8. .train_test_split()")

splits = ds.train_test_split(test_size=0.2, seed=42)
print("Result type:", type(splits).__name__)
print("Split keys :", list(splits.keys()))
print("train size :", len(splits["train"]))
print("test size  :", len(splits["test"]))
print("train labels:", splits["train"]["label"])
print("test  labels:", splits["test"]["label"])


# ---------------------------------------------------------------------------
# 9. Tokenizer-style batched map (no model download)
# ---------------------------------------------------------------------------
banner("9. Tokenizer-style batched map (trivial whitespace tokenizer)")

VOCAB = {}        # word → id (built during training)
NEXT_ID = [3]     # 0=PAD, 1=UNK, 2=CLS, 3+ = real tokens


def get_id(word: str) -> int:
    if word not in VOCAB:
        VOCAB[word] = NEXT_ID[0]
        NEXT_ID[0] += 1
    return VOCAB[word]


# Build vocab from corpus
for text in CORPUS["text"]:
    for word in text.lower().split():
        get_id(word)


def tokenize_batch(batch, max_length=12, padding_id=0, unk_id=1):
    """
    Whitespace-tokenize each text, convert to IDs, pad/truncate to max_length.
    This mimics what a real fast tokenizer does in a batched .map() call.
    """
    all_input_ids = []
    all_attention_mask = []
    all_tokens = []

    for text in batch["text"]:
        words = text.lower().split()[:max_length]
        ids = [VOCAB.get(w, unk_id) for w in words]
        tokens = words
        # Pad
        pad_len = max_length - len(ids)
        ids += [padding_id] * pad_len
        mask = [1] * len(words) + [0] * pad_len
        tokens += ["[PAD]"] * pad_len

        all_input_ids.append(ids)
        all_attention_mask.append(mask)
        all_tokens.append(tokens)

    return {
        "input_ids":      all_input_ids,
        "attention_mask": all_attention_mask,
        "tokens":         all_tokens,
    }


ds_tok = ds.map(
    tokenize_batch,
    batched=True,
    batch_size=4,
    remove_columns=["text", "score"],   # drop raw columns, keep label
)
print("Tokenized features:", ds_tok.features)
print("Row 0:")
row = ds_tok[0]
print(f"  tokens      : {row['tokens']}")
print(f"  input_ids   : {row['input_ids']}")
print(f"  attn_mask   : {row['attention_mask']}")
print(f"  label       : {row['label']}")
print(f"Vocab size: {len(VOCAB)}")

print("\nAll done — 03.map_filter.py EXIT 0")

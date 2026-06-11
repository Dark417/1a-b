"""
02.stream.py — IterableDataset: streaming and lazy evaluation.

Covers:
- Streaming from Hub (graceful skip)
- IterableDataset from a local generator
- .take() / .skip() / .map() on streams
- Lazy vs eager and memory-mapping vs streaming comparison

Docs: https://huggingface.co/docs/datasets/stream
      https://huggingface.co/docs/datasets/about_arrow
"""

import random
import os
import sys

import numpy as np
from datasets import IterableDataset, load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# 1. Hub streaming (graceful skip)
# ---------------------------------------------------------------------------
banner("1. load_dataset(..., streaming=True) — Hub")

ok, result = safe(load_dataset, "rotten_tomatoes", split="train", streaming=True)
if ok:
    hub_stream = result
    print("Type:", type(hub_stream).__name__)
    print("First 3 rows (lazy .take):")
    for row in hub_stream.take(3):
        print(" ", row)
    # Show that chained transforms are lazy
    mapped = hub_stream.map(lambda ex: {**ex, "text_upper": ex["text"].upper()})
    filtered = mapped.filter(lambda ex: ex["label"] == 1)
    print("\nFirst 2 after map+filter:")
    for row in filtered.take(2):
        print(" ", row)
else:
    note_skip(f"needs network — skipping hub stream ({result})")


# ---------------------------------------------------------------------------
# 2. Local IterableDataset from a generator
# ---------------------------------------------------------------------------
banner("2. IterableDataset.from_generator()")

CORPUS = [
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
]


def text_generator(corpus=CORPUS):
    """Yield dicts, simulating a lazy data pipeline."""
    for idx, text in enumerate(corpus):
        yield {
            "id":    idx,
            "text":  text,
            "words": len(text.split()),
        }


ids = IterableDataset.from_generator(text_generator)
print("Type              :", type(ids).__name__)
print("Features (inferred):", ids.features)

print("\nFirst 3 rows (.take):")
for row in ids.take(3):
    print(" ", row)

print("\nSkip 2, take 3 (.skip then .take):")
for row in ids.skip(2).take(3):
    print(" ", row)


# ---------------------------------------------------------------------------
# 3. .map() on IterableDataset — lazy transform
# ---------------------------------------------------------------------------
banner("3. Lazy .map() on IterableDataset")

# All transforms are composed but NOT executed until iteration
mapped_ids = (
    ids
    .map(lambda ex: {**ex, "text_upper": ex["text"].upper()})
    .map(lambda ex: {**ex, "char_count": len(ex["text"])})
    .filter(lambda ex: ex["words"] > 7)
)

print("Rows with >7 words:")
for row in mapped_ids:
    print(f"  id={row['id']}  words={row['words']}  chars={row['char_count']}")


# ---------------------------------------------------------------------------
# 4. Batched streaming map
# ---------------------------------------------------------------------------
banner("4. Batched streaming .map()")


def batch_tokenize(batch):
    """Trivial whitespace tokenizer — no model download."""
    batch["token_ids"] = [
        list(range(len(text.split()))) for text in batch["text"]
    ]
    batch["n_tokens"] = [len(t) for t in batch["token_ids"]]
    return batch


batched_ids = ids.map(batch_tokenize, batched=True, batch_size=3)
print("First 4 rows after batched tokenization:")
for row in batched_ids.take(4):
    print(f"  id={row['id']}  n_tokens={row['n_tokens']}  token_ids={row['token_ids'][:5]}…")


# ---------------------------------------------------------------------------
# 5. Lazy vs Eager — conceptual comparison
# ---------------------------------------------------------------------------
banner("5. Lazy (IterableDataset) vs Eager (Dataset) comparison")

from datasets import Dataset

eager_ds = Dataset.from_list([{"id": i, "text": CORPUS[i % len(CORPUS)]} for i in range(10)])

print("Eager Dataset:")
print(f"  type     : {type(eager_ds).__name__}")
print(f"  len      : {len(eager_ds)}")          # available immediately
print(f"  ds[3]    : {eager_ds[3]}")             # random access OK
print(f"  ds[2:5]  : {eager_ds[2:5]}")           # slicing OK
print(f"  .data    : pyarrow.Table (mmap'd)")

print("\nIterableDataset (from generator):")
print(f"  type     : {type(ids).__name__}")
print(f"  len      : N/A — raises TypeError")
print(f"  random access: N/A")
print(f"  iteration: one element at a time, O(1) RAM")

# Memory-mapping vs streaming note
print("""
─── Memory-mapping (Dataset) ────────────────────────────────────────────
Arrow files are mmap'd into virtual address space. The OS pages data in
on demand. Resident memory  ~  accessed data, NOT full dataset size.
Random access, slicing, and column selection are all O(1) seeks.
Best for: datasets that fit on disk, need random access / shuffling.

─── Streaming (IterableDataset) ─────────────────────────────────────────
No file on disk. Data generated/fetched lazily row-by-row (or batch).
Memory per step = one row/batch. Works for infinite or remote streams.
Best for: datasets too large for disk, or remote APIs, or generators.
──────────────────────────────────────────────────────────────────────────
""")

print("All done — 02.stream.py EXIT 0")

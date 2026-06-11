# HuggingFace `datasets` Library

> Official docs: https://huggingface.co/docs/datasets/index

---

## What is `datasets`?

`datasets` is HuggingFace's library for loading, processing, and sharing NLP (and multimodal) datasets.
It sits on top of **Apache Arrow** and provides:

- A hub of thousands of ready-to-use datasets (text, audio, image, …).
- Efficient local loading from CSV, JSON, Parquet, and custom generators.
- Dataset operations that feel like pandas but *scale beyond RAM* via memory-mapping.

---

## Apache Arrow & Memory-Mapping

### Why it scales beyond RAM

Traditional Python lists and NumPy arrays live entirely in RAM.
Arrow stores data in a **columnar, binary, language-agnostic** format on disk,
then *memory-maps* (mmap) the file into the process's virtual address space.

| Property | Effect |
|---|---|
| mmap | OS lazily pages data in; only accessed rows hit RAM |
| Columnar | Selecting one column is one contiguous read |
| Zero-copy | A Python slice is a view, not a copy |
| Arrow IPC | Read same file from multiple processes with no copy |

This means a 100 GB dataset can be *opened* instantly and iterated in chunks,
with resident memory proportional to the working set, not the file size.

### Zero-copy slicing

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files="big.parquet", split="train")
# ds.data is a pyarrow.Table — a view into the mmap'd file
chunk = ds[0:1000]   # view, not copy
```

---

## Core Data Structures

### `Dataset`

A table-like object backed by Arrow.

```python
from datasets import Dataset
ds = Dataset.from_dict({"text": ["hello", "world"], "label": [0, 1]})
print(ds.features)          # schema
print(ds[0])                # first row as dict
print(ds["text"])           # entire column as list
print(ds[0:2])              # slice → dict of lists
```

### `DatasetDict`

A dict-like container mapping split names to `Dataset` objects.

```python
from datasets import DatasetDict
dd = DatasetDict({
    "train": Dataset.from_dict({"x": [1, 2, 3]}),
    "test":  Dataset.from_dict({"x": [4, 5]}),
})
print(dd["train"][0])
```

### `IterableDataset`

Lazy dataset — data is not loaded until iterated.
Good for huge datasets that don't fit in RAM at all.

```python
from datasets import IterableDataset
def gen():
    for i in range(10):
        yield {"id": i, "text": f"item {i}"}

ids = IterableDataset.from_generator(gen)
for row in ids.take(3):
    print(row)
```

---

## Loading Datasets

### From the Hub

```python
from datasets import load_dataset
# Full dataset (cached locally after first download)
ds = load_dataset("glue", "sst2")
# Specific split
train = load_dataset("glue", "sst2", split="train")
```

### From local files (CSV / JSON / Parquet)

```python
ds_csv   = load_dataset("csv",     data_files="data.csv")
ds_json  = load_dataset("json",    data_files="data.jsonl")
ds_parq  = load_dataset("parquet", data_files="data.parquet")
```

### From in-memory Python objects

```python
import pandas as pd
from datasets import Dataset

# From dict
ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})

# From pandas
df = pd.DataFrame({"x": [1, 2, 3]})
ds = Dataset.from_pandas(df)

# From generator (IterableDataset)
def gen():
    for i in range(5):
        yield {"i": i}
ids = IterableDataset.from_generator(gen)
```

---

## Core Operations

### `.map()`

Apply a function to every example (or batch).

```python
def add_len(example):
    example["length"] = len(example["text"].split())
    return example

ds = ds.map(add_len)

# Batched is faster (vectorised ops, fewer Python calls)
def batch_upper(batch):
    batch["text"] = [t.upper() for t in batch["text"]]
    return batch

ds = ds.map(batch_upper, batched=True, batch_size=32)
```

- Results are **cached** on disk by default (keyed on function source + arguments).
- Use `num_proc=N` to parallelise (requires picklable function).
- `remove_columns=["col"]` can drop input cols in one call.

### `.filter()`

Keep only rows satisfying a predicate.

```python
ds = ds.filter(lambda ex: ex["label"] == 1)
```

### `.cast_column()` / `ClassLabel`

```python
from datasets import ClassLabel, Value
ds = ds.cast_column("label", ClassLabel(names=["neg", "pos"]))
print(ds.features["label"].int2str(0))  # "neg"
```

### `.rename_column()` / `.remove_columns()`

```python
ds = ds.rename_column("text", "sentence")
ds = ds.remove_columns(["unwanted"])
```

### `.select()`

Pick specific rows by index (like iloc).

```python
small = ds.select(range(100))
```

### `.shuffle()`

```python
ds = ds.shuffle(seed=42)
```

### `.train_test_split()`

```python
splits = ds.train_test_split(test_size=0.2, seed=42)
train, test = splits["train"], splits["test"]
```

---

## Streaming (IterableDataset)

```python
ds = load_dataset("some/big-dataset", split="train", streaming=True)
# No data downloaded yet
for row in ds.take(10):
    print(row)

# .map and .filter work lazily
ds = ds.map(lambda ex: {"tokens": ex["text"].split()})
ds = ds.filter(lambda ex: len(ex["tokens"]) > 5)
```

| | Dataset (eager) | IterableDataset (streaming) |
|---|---|---|
| Storage | Arrow mmap on disk | Nothing until iterated |
| RAM | mmap (near-zero) | One batch at a time |
| Random access | Yes (`ds[i]`) | No |
| `.map` | Cached, re-usable | Re-applied each epoch |
| Shuffling | Full (in memory) | Buffer-based (approximate) |

---

## `set_format` — Interop with ML Frameworks

```python
ds.set_format("torch", columns=["input_ids", "label"])
# ds[0]["input_ids"] is now a torch.Tensor

ds.set_format("numpy")
ds.set_format("pandas")
ds.reset_format()  # back to dicts
```

---

## Features & Schemas

```python
from datasets import Features, Value, ClassLabel, Sequence

features = Features({
    "text":   Value("string"),
    "label":  ClassLabel(names=["neg", "pos"]),
    "scores": Sequence(Value("float32")),
})
ds = Dataset.from_dict({"text": ["hi"], "label": [1], "scores": [[0.1, 0.9]]},
                        features=features)
print(ds.features)
```

---

## Gotchas

1. **Caching**: `.map()` results are cached in `~/.cache/huggingface/datasets/`.
   If you change the function *body* without changing the signature, the old cache
   is re-used. Force recompute with `load_from_cache_file=False`.

2. **Fingerprinting**: The cache key is computed from the function's source code and
   closures via `dill` hashing. Lambda functions with the same body but different
   closures may get the same fingerprint — use named functions to be safe.

3. **`num_proc` + Windows**: Multiprocessing `.map()` requires `if __name__ == "__main__":`
   guards on Windows.

4. **`streaming=True` + `.map()`**: Maps are *not* cached. Every iteration re-applies
   the transform. Use `ds.with_format("torch")` + DataLoader for training loops.

5. **Arrow type mismatch**: When building from dict, Arrow infers types.
   Explicit `Features` prevents surprises (e.g. int32 vs int64).

---

## References

- https://huggingface.co/docs/datasets/index
- https://huggingface.co/docs/datasets/about_arrow
- https://huggingface.co/docs/datasets/process
- https://huggingface.co/docs/datasets/stream
- https://huggingface.co/docs/datasets/create_dataset
- https://arrow.apache.org/docs/python/memory.html

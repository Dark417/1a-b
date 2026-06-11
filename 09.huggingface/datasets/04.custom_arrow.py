"""
04.custom_arrow.py — Custom Arrow datasets, disk persistence, and schema.

Covers:
- Dataset from generator (with explicit Features)
- save_to_disk / load_from_disk
- Arrow internals: .data (pyarrow.Table), .cache_files, .dataset_size
- to_pandas / to_dict
- Features schema: Value, ClassLabel, Sequence

Docs: https://huggingface.co/docs/datasets/create_dataset
      https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset
      https://huggingface.co/docs/datasets/about_arrow
      https://arrow.apache.org/docs/python/memory.html
"""

import random
import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    ClassLabel,
    Sequence,
    load_from_disk,
)

sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Shared tiny corpus used throughout
# ---------------------------------------------------------------------------
TEXTS = [
    "Arrow uses columnar memory layout for zero-copy reads",
    "Memory-mapping lets datasets scale beyond available RAM",
    "Each Arrow buffer is page-aligned for efficient IO",
    "PyArrow tables support zero-copy slicing via views",
    "HuggingFace datasets stores data in Arrow IPC format",
    "Columnar storage improves cache locality for column scans",
    "Chunked arrays allow incremental appends without realloc",
    "Arrow flight protocol enables fast data transfer over gRPC",
    "RecordBatch is the unit of Arrow row-group storage",
    "DictionaryEncoding compresses repeated string values efficiently",
    "Parquet files use Arrow as the in-memory read format",
    "Zero-copy means slicing returns a view, not a copy",
]
LABELS = [i % 3 for i in range(len(TEXTS))]
SCORES = [round(0.5 + 0.4 * (i / len(TEXTS)), 3) for i in range(len(TEXTS))]
KEYWORDS_LIST = [t.lower().split()[:3] for t in TEXTS]


# ---------------------------------------------------------------------------
# 1. Dataset from a generator with explicit Features schema
# ---------------------------------------------------------------------------
banner("1. Dataset.from_generator() with explicit Features")

LABEL_NAMES = ["storage", "access", "protocol"]

features = Features({
    "text":     Value("string"),
    "label":    ClassLabel(names=LABEL_NAMES),
    "score":    Value("float32"),
    "keywords": Sequence(Value("string")),
    "word_ids": Sequence(Value("int32")),
})


def corpus_generator():
    """Yield fully typed examples from our local corpus."""
    for i, (text, label, score, kws) in enumerate(
        zip(TEXTS, LABELS, SCORES, KEYWORDS_LIST)
    ):
        yield {
            "text":     text,
            "label":    label,
            "score":    score,
            "keywords": kws,
            "word_ids": list(range(len(text.split()))),
        }


ds = Dataset.from_generator(corpus_generator, features=features)
print("Dataset   :", ds)
print("Features  :", ds.features)
print("Row 0     :", ds[0])
print("Row 1     :", ds[1])
print("ClassLabel int→str:", ds.features["label"].int2str(0),
      ds.features["label"].int2str(1), ds.features["label"].int2str(2))


# ---------------------------------------------------------------------------
# 2. Arrow internals: .data, .dataset_size, .cache_files
# ---------------------------------------------------------------------------
banner("2. Arrow internals: .data, .dataset_size, .cache_files")

arrow_table = ds.data           # pyarrow.Table (or ChunkedArray view)
print("Type of .data     :", type(arrow_table).__name__)
print("PyArrow schema    :", arrow_table.schema)
print("Num columns       :", arrow_table.num_columns)
print("Num rows          :", arrow_table.num_rows)

# dataset_size is approximate bytes in Arrow buffers
print("Dataset size (B)  :", ds.dataset_size)

# cache_files is populated after save_to_disk / load_from_disk
print("Cache files now   :", ds.cache_files)

# Zero-copy slice: the result shares the same Arrow buffers
slice_table = arrow_table.slice(0, 5)
print("Sliced table rows :", len(slice_table))
print("(slice is a zero-copy view of the same Arrow buffers)")

# Access a single column as pyarrow array
text_col = arrow_table.column("text")
print("text column type  :", type(text_col).__name__, "dtype:", text_col.type)
print("First 3 texts     :", text_col[:3].to_pylist())


# ---------------------------------------------------------------------------
# 3. save_to_disk / load_from_disk
# ---------------------------------------------------------------------------
banner("3. save_to_disk() / load_from_disk()")

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "my_arrow_dataset")

    ds.save_to_disk(save_path)
    print("Saved to:", save_path)
    print("Files on disk:", sorted(os.listdir(save_path)))

    # Reload — this is now memory-mapped from disk
    ds_reloaded = load_from_disk(save_path)
    print("Reloaded dataset:", ds_reloaded)
    print("Cache files (mmap'd):", ds_reloaded.cache_files)
    print("Row 0 matches:", ds_reloaded[0] == ds[0])

    # DatasetDict round-trip
    dd = DatasetDict({
        "train": ds.select(range(9)),
        "test":  ds.select(range(9, 12)),
    })
    dd_path = os.path.join(tmpdir, "my_dataset_dict")
    dd.save_to_disk(dd_path)
    dd_loaded = DatasetDict.load_from_disk(dd_path)
    print("\nDatasetDict reload:", dd_loaded)
    print("Train size:", len(dd_loaded["train"]))
    print("Test  size:", len(dd_loaded["test"]))


# ---------------------------------------------------------------------------
# 4. to_pandas / to_dict
# ---------------------------------------------------------------------------
banner("4. to_pandas() / to_dict()")

# Sequences become Python lists in pandas
df = ds.to_pandas()
print("Pandas DataFrame shape:", df.shape)
print("Columns:", list(df.columns))
print("dtypes:\n", df.dtypes.to_string())
print("Head:\n", df.head(3).to_string())

# to_dict returns a plain Python dict of lists
d = ds.to_dict()
print("\nto_dict() keys     :", list(d.keys()))
print("type(d['label'])   :", type(d["label"]))
print("labels             :", d["label"])

# to_list returns a list of row dicts
rows_list = ds.to_list()
print("to_list() len      :", len(rows_list))
print("to_list()[0]       :", rows_list[0])


# ---------------------------------------------------------------------------
# 5. Features schema deep dive: Value / ClassLabel / Sequence
# ---------------------------------------------------------------------------
banner("5. Features schema — Value, ClassLabel, Sequence")

print("Value types available: string, int8/16/32/64, uint8/16/32/64,")
print("  float16/32/64, bool, binary, large_string, large_binary\n")

rich_features = Features({
    "id":         Value("int32"),
    "text":       Value("string"),
    "label":      ClassLabel(names=["neg", "neu", "pos"]),
    "float_feat": Value("float32"),
    "bool_feat":  Value("bool"),
    "token_ids":  Sequence(Value("int32")),
    "char_probs": Sequence(Value("float32")),
    "tag_ids":    Sequence(ClassLabel(names=["O", "B-ORG", "I-ORG"])),
})

rich_ds = Dataset.from_dict(
    {
        "id":         [1, 2],
        "text":       ["hello world", "foo bar baz"],
        "label":      [0, 2],
        "float_feat": [0.1, 0.9],
        "bool_feat":  [True, False],
        "token_ids":  [[10, 20, 30], [40, 50, 60, 70]],
        "char_probs": [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],
        "tag_ids":    [[0, 1, 2], [0, 0, 1, 2]],
    },
    features=rich_features,
)
print("Rich features schema:")
for col, feat in rich_ds.features.items():
    print(f"  {col:<14}: {feat}")
print("\nRow 0:", rich_ds[0])
print("Row 1:", rich_ds[1])

# ClassLabel inside Sequence
tag_feat = rich_ds.features["tag_ids"].feature
print("\ntag_ids element feature:", tag_feat)
print("int2str(0):", tag_feat.int2str(0))
print("int2str(1):", tag_feat.int2str(1))


# ---------------------------------------------------------------------------
# 6. Building Arrow table directly then wrapping as Dataset
# ---------------------------------------------------------------------------
banner("6. Build pyarrow.Table directly → wrap as Dataset")

pa_table = pa.table({
    "x": pa.array([1, 2, 3, 4], type=pa.int32()),
    "y": pa.array([0.1, 0.2, 0.3, 0.4], type=pa.float32()),
    "z": pa.array(["a", "b", "c", "d"], type=pa.string()),
})
ds_from_arrow = Dataset(pa_table)
print("Dataset from pa.table:", ds_from_arrow)
print("Features:", ds_from_arrow.features)
print("Row 0   :", ds_from_arrow[0])

print("\nAll done — 04.custom_arrow.py EXIT 0")

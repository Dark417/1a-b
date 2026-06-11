"""
01.load.py — Loading datasets from multiple sources.

Covers:
- In-memory dict
- Pandas DataFrame
- Local CSV / JSON (written to tempfile)
- Hub dataset (graceful skip if no network)

Docs: https://huggingface.co/docs/datasets/loading
"""

import random
import tempfile
import os
import json
import csv

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, Features, Value, ClassLabel

import sys
sys.path.insert(0, os.path.dirname(__file__))
from _lib import banner, note_skip, safe

random.seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# 1. From in-memory dict
# ---------------------------------------------------------------------------
banner("1. Dataset.from_dict()")

ds_dict = Dataset.from_dict({
    "text":  ["The cat sat", "A dog ran", "Birds fly high", "Fish swim deep"],
    "label": [0, 1, 0, 1],
})
print("Schema  :", ds_dict.features)
print("Num rows:", len(ds_dict))
print("Row 0   :", ds_dict[0])
print("Slice   :", ds_dict[1:3])       # dict of lists
print("Column  :", ds_dict["label"])   # entire column


# ---------------------------------------------------------------------------
# 2. From pandas DataFrame
# ---------------------------------------------------------------------------
banner("2. Dataset.from_pandas()")

df = pd.DataFrame({
    "sentence": ["NLP is fun", "Deep learning rocks", "Transformers are powerful"],
    "score":    [0.9, 0.8, 0.95],
    "category": ["A", "B", "A"],
})
ds_pd = Dataset.from_pandas(df)
print("Features:", ds_pd.features)
print("Row 1   :", ds_pd[1])
# __index_level_0__ may appear — drop it
if "__index_level_0__" in ds_pd.column_names:
    ds_pd = ds_pd.remove_columns(["__index_level_0__"])
print("Columns :", ds_pd.column_names)


# ---------------------------------------------------------------------------
# 3. From a local CSV tempfile
# ---------------------------------------------------------------------------
banner("3. load_dataset('csv', data_files=...)")

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    csv_path = f.name
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    rows = [
        {"text": "sunny day",   "label": "pos"},
        {"text": "rainy night", "label": "neg"},
        {"text": "cloudy sky",  "label": "neu"},
        {"text": "snowy peak",  "label": "pos"},
    ]
    writer.writerows(rows)

ok, result = safe(load_dataset, "csv", data_files=csv_path, split="train")
if ok:
    ds_csv = result
    print("CSV dataset:", ds_csv)
    print("First row  :", ds_csv[0])
else:
    note_skip(f"load_dataset csv failed ({result}) — building manually")
    ds_csv = Dataset.from_dict({
        "text":  [r["text"]  for r in rows],
        "label": [r["label"] for r in rows],
    })
    print("Fallback CSV dataset:", ds_csv[0])
os.unlink(csv_path)


# ---------------------------------------------------------------------------
# 4. From a local JSON(L) tempfile
# ---------------------------------------------------------------------------
banner("4. load_dataset('json', data_files=...)")

with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
    jsonl_path = f.name
    json_rows = [
        {"id": 1, "text": "hello world",      "tag": "greet"},
        {"id": 2, "text": "goodbye world",    "tag": "farewell"},
        {"id": 3, "text": "the quick brown",  "tag": "pangram"},
    ]
    for row in json_rows:
        f.write(json.dumps(row) + "\n")

ok, result = safe(load_dataset, "json", data_files=jsonl_path, split="train")
if ok:
    ds_json = result
    print("JSON dataset:", ds_json)
    print("First row   :", ds_json[0])
else:
    note_skip(f"load_dataset json failed ({result}) — building manually")
    ds_json = Dataset.from_dict({
        "id":   [r["id"]   for r in json_rows],
        "text": [r["text"] for r in json_rows],
        "tag":  [r["tag"]  for r in json_rows],
    })
    print("Fallback JSON dataset:", ds_json[0])
os.unlink(jsonl_path)


# ---------------------------------------------------------------------------
# 5. Hub dataset — graceful skip
# ---------------------------------------------------------------------------
banner("5. load_dataset from Hub (rotten_tomatoes, small)")

ok, result = safe(load_dataset, "rotten_tomatoes")
if ok:
    ds_hub = result
    print("Hub DatasetDict:", ds_hub)
    print("Train size     :", len(ds_hub["train"]))
    print("First train row:", ds_hub["train"][0])
    print("Features       :", ds_hub["train"].features)
else:
    note_skip(f"needs network — using local fallback ({result})")
    # Build an equivalent tiny in-memory DatasetDict
    feats = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})
    ds_hub = DatasetDict({
        "train": Dataset.from_dict(
            {"text": ["great film", "boring movie", "loved it"],
             "label": [1, 0, 1]}, features=feats),
        "validation": Dataset.from_dict(
            {"text": ["ok film", "terrible"],
             "label": [1, 0]}, features=feats),
        "test": Dataset.from_dict(
            {"text": ["masterpiece"],
             "label": [1]}, features=feats),
    })
    print("Local DatasetDict:", ds_hub)


# ---------------------------------------------------------------------------
# 6. DatasetDict: indexing, splits, features
# ---------------------------------------------------------------------------
banner("6. DatasetDict — splits / indexing / features")

print("Splits       :", list(ds_hub.keys()))
train = ds_hub["train"]
print("Train features:", train.features)
print("train[0]      :", train[0])
print("train[1:3]    :", train[1:3])
print("train['text'] :", train["text"])   # full column
print("train.shape   :", train.shape)
print("column_names  :", train.column_names)

print("\nAll done — 01.load.py EXIT 0")

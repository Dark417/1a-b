# huggingface_hub — University-Grade Explainer

**Official docs:** https://huggingface.co/docs/huggingface_hub/index  
**API reference:** https://huggingface.co/docs/huggingface_hub/package_reference/hf_api

---

## What is huggingface_hub?

`huggingface_hub` is the Python client library for the Hugging Face Hub — the
central registry of models, datasets, and Spaces.  It handles authentication,
downloading, uploading, searching, and model-card management, and it is the
backbone used by `transformers`, `diffusers`, `datasets`, and others whenever
they call `from_pretrained`.

---

## Authentication

### HF_TOKEN

Most read operations are public (no token required).  Write operations (create
repo, upload, push) require an access token created at
https://huggingface.co/settings/tokens.

```python
import os
from huggingface_hub import login, whoami

# Option 1 — environment variable (recommended in CI/CD)
# export HF_TOKEN=hf_...
login(token=os.environ["HF_TOKEN"])   # stores to ~/.cache/huggingface/token

# Option 2 — interactive prompt
login()   # prompts for token in terminal

# Check who you are
info = whoami()
print(info["name"], info["email"])
```

### HF_HUB_OFFLINE

Set `HF_HUB_OFFLINE=1` (or `export HF_HUB_OFFLINE=1`) to prevent *any*
network call.  All `hf_hub_download` calls will serve from cache or raise
`LocalEntryNotFoundError`.  Useful for reproducible offline environments.

**Docs:** https://huggingface.co/docs/huggingface_hub/guides/manage-cache#avoid-network-requests

---

## Downloading Files

### hf_hub_download — single file

Downloads (or retrieves from cache) **one file** from a repository:

```python
from huggingface_hub import hf_hub_download

local_path = hf_hub_download(
    repo_id   = "bert-base-uncased",
    filename  = "config.json",
    cache_dir = "/tmp/my_cache",   # optional; default: ~/.cache/huggingface/hub
    revision  = "main",            # branch / tag / commit SHA
)
print(local_path)  # /tmp/my_cache/models--bert-base-uncased/snapshots/<sha>/config.json
```

Subsequent calls return the **cached path instantly** — no network hit.

**Docs:** https://huggingface.co/docs/huggingface_hub/guides/download#download-a-single-file

### snapshot_download — whole repo (or filtered subset)

Downloads an entire repository into a single snapshot directory:

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id        = "bert-base-uncased",
    allow_patterns = ["*.json", "*.txt"],   # only download matching files
    ignore_patterns= ["*.bin", "*.safetensors"],  # alternatively, exclude heavy files
    cache_dir      = "/tmp/my_cache",
    revision       = "main",
)
print(local_dir)  # path to the snapshot directory
```

`allow_patterns` / `ignore_patterns` accept glob strings or lists.

**Docs:** https://huggingface.co/docs/huggingface_hub/guides/download#download-an-entire-repository

---

## Cache Layout

The Hub cache is a content-addressed store at
`~/.cache/huggingface/hub/` by default (override with `HF_HOME` or
`cache_dir=`).

```
~/.cache/huggingface/hub/
└── models--bert-base-uncased/
    ├── blobs/          # raw file content, named by SHA-256
    ├── refs/
    │   └── main        # text file: the resolved commit SHA
    └── snapshots/
        └── <commit_sha>/
            ├── config.json     → symlink → ../../blobs/<sha>
            ├── vocab.txt       → symlink → ../../blobs/<sha>
            └── pytorch_model.bin → symlink → ../../blobs/<sha>
```

Key properties:
- **Deduplication**: identical blobs are stored once regardless of how many
  repos reference them.
- **Atomic writes**: partial downloads are written to a tmp file and renamed
  only on success, so a crash never corrupts the cache.
- **Re-entrant**: two processes can safely download the same file concurrently.

```python
from huggingface_hub import scan_cache_dir
hf_cache_info = scan_cache_dir()
print(hf_cache_info)   # total size, per-repo breakdown
```

**Docs:** https://huggingface.co/docs/huggingface_hub/guides/manage-cache

---

## HfApi — Search and Metadata

`HfApi` is the low-level client for the Hub REST API.

```python
from huggingface_hub import HfApi
api = HfApi()
```

### list_models / list_datasets

```python
# Iterate over text-classification models, sorted by downloads, take top 5
for m in api.list_models(
    filter      = "text-classification",
    sort        = "downloads",
    direction   = -1,            # descending
    limit       = 5,
):
    print(m.id, m.downloads)

# Datasets
for d in api.list_datasets(search="sentiment", limit=3):
    print(d.id)
```

**Docs:** https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_models

### model_info / dataset_info

```python
info = api.model_info("bert-base-uncased")
print(info.id)           # "bert-base-uncased"
print(info.downloads)    # download count
print(info.cardData)     # parsed YAML front-matter from README.md
print(info.tags)         # ["pytorch", "bert", ...]
print(info.pipeline_tag) # "fill-mask"

dinfo = api.dataset_info("squad")
print(dinfo.id, dinfo.description[:80])
```

**Docs:** https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.model_info

---

## Revisions and Pinning

Every repository is a git repo on the Hub.  You can pin to:
- `revision="main"` — the default branch (mutable, may change)
- `revision="v1.0"` — a tag (stable)
- `revision="abc1234"` — a full commit SHA (immutable, reproducible)

```python
local_path = hf_hub_download(
    repo_id  = "bert-base-uncased",
    filename = "config.json",
    revision = "e8a64d0",   # exact commit — fully reproducible
)
```

---

## Creating Repos and Uploading Files

These operations require a write-access token.

### create_repo

```python
from huggingface_hub import HfApi
api = HfApi()

url = api.create_repo(
    repo_id   = "my-username/my-model",
    repo_type = "model",       # "model" | "dataset" | "space"
    private   = True,
    exist_ok  = True,          # don't raise if already exists
)
print(url)   # https://huggingface.co/my-username/my-model
```

### upload_file

```python
api.upload_file(
    path_or_fileobj = "/local/path/to/model.safetensors",
    path_in_repo    = "model.safetensors",
    repo_id         = "my-username/my-model",
    repo_type       = "model",
    commit_message  = "Add trained weights",
)
```

### upload_folder

```python
api.upload_folder(
    folder_path  = "/local/path/to/output_dir/",
    repo_id      = "my-username/my-model",
    repo_type    = "model",
    ignore_patterns = ["*.tmp", "__pycache__/*"],
)
```

### push_to_hub (from transformers / datasets)

Most `transformers` objects also expose `push_to_hub`:

```python
model.push_to_hub("my-username/my-model")
tokenizer.push_to_hub("my-username/my-model")
```

**Docs:** https://huggingface.co/docs/huggingface_hub/guides/upload

---

## ModelCard and ModelCardData

A `ModelCard` is the `README.md` that lives in each model repo.  Its YAML
front-matter (called *card data*) is machine-readable metadata.

```python
from huggingface_hub import ModelCard, ModelCardData

card_data = ModelCardData(
    language         = ["en"],
    license          = "apache-2.0",
    pipeline_tag     = "text-classification",
    datasets         = ["glue"],
    metrics          = ["accuracy", "f1"],
    tags             = ["bert", "fine-tuned"],
    model_name       = "my-bert-sentiment",
)

card = ModelCard.from_template(
    card_data,
    template_str     = "---\n{{ card_data }}\n---\n\n## Model description\n...",
)

print(card.content)   # full markdown string
card.save("README.md")

# Push the card to the Hub
# card.push_to_hub("my-username/my-model")
```

**Docs:** https://huggingface.co/docs/huggingface_hub/guides/model-cards

---

## Summary of Key Functions

| Function / Class | Purpose |
|---|---|
| `hf_hub_download(repo_id, filename)` | Download a single file |
| `snapshot_download(repo_id)` | Download a whole repo |
| `HfApi().list_models(...)` | Search the model hub |
| `HfApi().model_info(repo_id)` | Fetch model metadata |
| `HfApi().create_repo(...)` | Create a new repo |
| `HfApi().upload_file(...)` | Upload one file |
| `HfApi().upload_folder(...)` | Upload a directory |
| `ModelCard` / `ModelCardData` | Build and push model cards |
| `login()` / `whoami()` | Authentication |
| `scan_cache_dir()` | Inspect local cache |

"""
04.model_card.py — Build and render a ModelCard with ModelCardData

No network required.  push_to_hub is guarded by HF_TOKEN.

Official docs:
  https://huggingface.co/docs/huggingface_hub/guides/model-cards
  https://huggingface.co/docs/huggingface_hub/package_reference/cards
"""

import os
import random
import sys
import tempfile

import numpy as np

random.seed(0)
np.random.seed(0)

from _lib import banner, note_skip, safe
from huggingface_hub import ModelCard, ModelCardData

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HAS_TOKEN = bool(HF_TOKEN)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Build ModelCardData (the YAML front-matter)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 1 — ModelCardData (YAML front-matter)")

print(
    "\nModelCardData holds the machine-readable metadata that appears as YAML\n"
    "front-matter at the top of a model's README.md on the Hub.\n"
)

card_data = ModelCardData(
    language         = ["en"],
    license          = "apache-2.0",
    pipeline_tag     = "text-classification",
    datasets         = ["glue", "sst2"],
    metrics          = ["accuracy", "f1"],
    tags             = ["bert", "sentiment", "fine-tuned", "pytorch"],
    model_name       = "demo-bert-sentiment",
    library_name     = "transformers",
)

print("ModelCardData fields:")
print(f"  language    : {card_data.language}")
print(f"  license     : {card_data.license}")
print(f"  pipeline_tag: {card_data.pipeline_tag}")
print(f"  datasets    : {card_data.datasets}")
print(f"  metrics     : {card_data.metrics}")
print(f"  tags        : {card_data.tags}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Build a ModelCard with a custom template
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 2 — ModelCard.from_template")

TEMPLATE = """\
---
{{ card_data }}
---

# Demo BERT Sentiment Model

This model is a fine-tuned BERT for binary sentiment classification
(positive / negative) on SST-2.

## Model description

- **Base model:** bert-base-uncased
- **Task:** Sentiment analysis (text-classification)
- **Language:** English

## Training data

Fine-tuned on the [SST-2](https://huggingface.co/datasets/sst2) subset of GLUE.

## Evaluation results

| Metric   | Value  |
|----------|--------|
| Accuracy | 93.1%  |
| F1       | 93.0%  |

## Usage

```python
from transformers import pipeline
clf = pipeline("text-classification", model="my-user/demo-bert-sentiment")
result = clf("I love this movie!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

## Limitations

This model was trained on English movie reviews and may not generalise
well to other domains or languages.
"""

card = ModelCard.from_template(card_data, template_str=TEMPLATE)

print("\nRendered ModelCard (full markdown):\n")
print("-" * 60)
print(card.content)
print("-" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Save to a local README.md
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 3 — card.save to a local file")

tmpdir = tempfile.mkdtemp(prefix="hf_model_card_")
readme_path = os.path.join(tmpdir, "README.md")
card.save(readme_path)

print(f"Saved to: {readme_path}")
print(f"File size: {os.path.getsize(readme_path)} bytes")

# Read it back to confirm round-trip
with open(readme_path) as f:
    on_disk = f.read()
print(f"\nFirst 200 chars on disk:\n{on_disk[:200]}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Loading an existing card from a string / file
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 4 — Loading an existing ModelCard")

card_loaded = ModelCard(on_disk)
print(f"Re-loaded card data:")
print(f"  pipeline_tag : {card_loaded.data.pipeline_tag}")
print(f"  license      : {card_loaded.data.license}")
print(f"  tags         : {card_loaded.data.tags}")

# Also show ModelCard.load (class method) on the saved file
card_from_file = ModelCard.load(readme_path)
print(f"\nLoaded via ModelCard.load(path):")
print(f"  datasets : {card_from_file.data.datasets}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: push_to_hub (guarded)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 5 — card.push_to_hub (requires HF_TOKEN)")

if HAS_TOKEN:
    print("HF_TOKEN present — attempting push ...")
    ok, result = safe(card.push_to_hub, "my-username/demo-bert-sentiment",
                      token=HF_TOKEN)
    if ok:
        print(f"  LIVE: Pushed card to Hub: {result}")
    else:
        note_skip(f"push_to_hub failed (token may lack write access): {result}")
else:
    note_skip("no HF_TOKEN — showing the API; not pushing")
    print("\nCode to push a card:")
    print('  card.push_to_hub("my-username/demo-bert-sentiment")')
    print("  # or supply a token explicitly:")
    print('  card.push_to_hub("my-username/demo-bert-sentiment", token=hf_token)')
    print()
    print("  # Alternatively, use HfApi:")
    print("  from huggingface_hub import HfApi")
    print("  api = HfApi(token=hf_token)")
    print("  api.upload_file(")
    print('      path_or_fileobj = readme_path,')
    print('      path_in_repo    = "README.md",')
    print('      repo_id         = "my-username/demo-bert-sentiment",')
    print('      commit_message  = "Update model card",')
    print("  )")

print("\n[done] 04.model_card.py — exit 0")

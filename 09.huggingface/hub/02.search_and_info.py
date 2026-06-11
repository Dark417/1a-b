"""
02.search_and_info.py — HfApi: list_models, model_info, list_datasets

Official docs:
  https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_models
  https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.model_info
  https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_datasets
"""

import random
import sys

import numpy as np

random.seed(0)
np.random.seed(0)

from _lib import banner, note_skip, safe
from huggingface_hub import HfApi

api = HfApi()

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: list_models
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 1 — HfApi.list_models(filter='text-classification', limit=5)")

print("Searching for text-classification models sorted by downloads ...")

def fetch_models():
    # direction=-1 (descending) supported in older hub; use try/except for compat
    try:
        models = list(api.list_models(
            filter    = "text-classification",
            sort      = "downloads",
            direction = -1,
            limit     = 5,
        ))
    except TypeError:
        # newer huggingface_hub removed the direction kwarg
        models = list(api.list_models(
            filter = "text-classification",
            sort   = "downloads",
            limit  = 5,
        ))
    return models

ok, result = safe(fetch_models)

if not ok:
    note_skip(
        f"needs network — demonstrating API shape with local fallback\n  ({result})"
    )
    print("\nAPI shape:")
    print("  from huggingface_hub import HfApi")
    print("  api = HfApi()")
    print("  for m in api.list_models(filter='text-classification', sort='downloads',")
    print("                           direction=-1, limit=5):")
    print("      print(m.id, m.downloads, m.pipeline_tag)")
    print("\nCanned example output (what you would see with network):")
    canned = [
        {"id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
         "downloads": 15_000_000, "pipeline_tag": "text-classification"},
        {"id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
         "downloads": 8_000_000,  "pipeline_tag": "text-classification"},
        {"id": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
         "downloads": 3_000_000,  "pipeline_tag": "text-classification"},
        {"id": "SamLowe/roberta-base-go_emotions",
         "downloads": 2_500_000,  "pipeline_tag": "text-classification"},
        {"id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
         "downloads": 2_000_000,  "pipeline_tag": "text-classification"},
    ]
    for m in canned:
        print(f"  {m['id']:<60}  downloads={m['downloads']:>12,}  "
              f"pipeline={m['pipeline_tag']}")
else:
    models = result
    print(f"\nFound {len(models)} models:\n")
    for m in models:
        dl = getattr(m, "downloads", "N/A")
        tag = getattr(m, "pipeline_tag", "N/A")
        print(f"  {m.modelId:<60}  downloads={str(dl):>12}  pipeline={tag}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: model_info
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 2 — HfApi.model_info('hf-internal-testing/tiny-random-gpt2')")

SMALL_REPO = "hf-internal-testing/tiny-random-gpt2"
print(f"Fetching metadata for: {SMALL_REPO}")

ok2, info = safe(api.model_info, SMALL_REPO)

if not ok2:
    note_skip(
        f"needs network — demonstrating API shape with local fallback\n  ({info})"
    )
    print("\nAPI shape:")
    print(f"  info = api.model_info('{SMALL_REPO}')")
    print("  info.id              # 'hf-internal-testing/tiny-random-gpt2'")
    print("  info.downloads       # int — total downloads")
    print("  info.tags            # list of strings")
    print("  info.pipeline_tag    # 'text-generation'")
    print("  info.cardData        # dict of YAML front-matter from README")
    print("  info.siblings        # list of RepoSibling (each .rfilename, .size)")
    print("\nCanned example:")
    print("  id          : hf-internal-testing/tiny-random-gpt2")
    print("  pipeline_tag: text-generation")
    print("  tags        : ['pytorch', 'gpt2', 'text-generation']")
    print("  downloads   : ~50000")
else:
    print(f"\n  id          : {info.id}")
    print(f"  pipeline_tag: {getattr(info, 'pipeline_tag', 'N/A')}")
    tags = getattr(info, "tags", []) or []
    print(f"  tags        : {tags[:8]}")
    dl = getattr(info, "downloads", "N/A")
    print(f"  downloads   : {dl}")
    card = getattr(info, "cardData", None)
    print(f"  cardData    : {card}")
    siblings = getattr(info, "siblings", []) or []
    print(f"  files in repo:")
    for s in siblings[:8]:
        sz = getattr(s, "size", None)
        print(f"    {s.rfilename:<40} {str(sz) + ' bytes' if sz else ''}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: list_datasets
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 3 — HfApi.list_datasets(search='sentiment', limit=5)")

print("Searching datasets matching 'sentiment' ...")

def fetch_datasets():
    return list(api.list_datasets(search="sentiment", limit=5))

ok3, datasets = safe(fetch_datasets)

if not ok3:
    note_skip(
        f"needs network — demonstrating API shape with local fallback\n  ({datasets})"
    )
    print("\nAPI shape:")
    print("  for d in api.list_datasets(search='sentiment', limit=5):")
    print("      print(d.id, d.downloads)")
    print("\nCanned example output:")
    canned_ds = [
        {"id": "stanfordnlp/sst2",                        "downloads": 5_000_000},
        {"id": "tweet_eval",                               "downloads": 1_200_000},
        {"id": "financial_phrasebank",                     "downloads": 800_000},
        {"id": "mteb/amazon_reviews_multi",                "downloads": 600_000},
        {"id": "sem_eval_2014_task_1",                     "downloads": 400_000},
    ]
    for d in canned_ds:
        print(f"  {d['id']:<50}  downloads={d['downloads']:>10,}")
else:
    print(f"\nFound {len(datasets)} datasets:\n")
    for d in datasets:
        dl = getattr(d, "downloads", "N/A")
        print(f"  {d.id:<50}  downloads={str(dl):>10}")

print(
    "\nKey takeaways:"
    "\n  • list_models / list_datasets return iterators — wrap in list() or iterate."
    "\n  • filter= accepts task tags ('text-classification') or library names ('pytorch')."
    "\n  • sort= accepts 'downloads', 'likes', 'lastModified', 'trending'."
    "\n  • model_info() returns rich metadata including siblings, card data, and SHA."
)

print("\n[done] 02.search_and_info.py — exit 0")

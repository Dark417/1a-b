"""
01.download.py — hf_hub_download and snapshot_download demo

Official docs:
  https://huggingface.co/docs/huggingface_hub/guides/download
  https://huggingface.co/docs/huggingface_hub/package_reference/file_download
"""

import os
import random
import sys
import time

import numpy as np

random.seed(0)
np.random.seed(0)

from _lib import banner, note_skip, safe
from huggingface_hub import hf_hub_download, snapshot_download, constants

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: hf_hub_download — single file
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 1 — hf_hub_download: single file")

REPO_ID  = "hf-internal-testing/tiny-random-gpt2"
FILENAME = "config.json"

print(f"Downloading {FILENAME} from {REPO_ID} ...")

ok, result = safe(
    hf_hub_download,
    repo_id   = REPO_ID,
    filename  = FILENAME,
)

if not ok:
    note_skip(
        f"needs network/model — demonstrating API shape with local fallback\n"
        f"  ({result})"
    )
    print("\nAPI shape (what hf_hub_download looks like):")
    print("  from huggingface_hub import hf_hub_download")
    print(f'  local_path = hf_hub_download(repo_id="{REPO_ID}", filename="{FILENAME}")')
    print("  # Returns e.g.:")
    default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    fake_path = os.path.join(
        default_cache,
        "models--hf-internal-testing--tiny-random-gpt2",
        "snapshots",
        "<commit_sha>",
        FILENAME,
    )
    print(f"  # {fake_path}")
else:
    local_path = result
    print(f"  Downloaded to : {local_path}")
    print(f"  File exists   : {os.path.exists(local_path)}")
    size = os.path.getsize(local_path)
    print(f"  File size     : {size} bytes")

    # Show that a second call is instant (served from cache)
    t0 = time.perf_counter()
    ok2, local_path2 = safe(
        hf_hub_download,
        repo_id  = REPO_ID,
        filename = FILENAME,
    )
    elapsed = time.perf_counter() - t0
    if ok2:
        print(f"\n  Second call (cached) returned in {elapsed*1000:.1f} ms")
        print(f"  Same path: {local_path == local_path2}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: snapshot_download — whole repo filtered to *.json
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 2 — snapshot_download: whole repo (allow_patterns=['*.json'])")

print(f"Downloading *.json files from {REPO_ID} ...")

ok3, snap_result = safe(
    snapshot_download,
    repo_id        = REPO_ID,
    allow_patterns = ["*.json"],
)

if not ok3:
    note_skip(
        f"needs network/model — demonstrating API shape with local fallback\n"
        f"  ({snap_result})"
    )
    print("\nAPI shape:")
    print("  from huggingface_hub import snapshot_download")
    print(f'  local_dir = snapshot_download(repo_id="{REPO_ID}",')
    print('                               allow_patterns=["*.json"])')
    print("  # Returns path to the snapshot directory, e.g.:")
    default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    fake_snap = os.path.join(
        default_cache,
        "models--hf-internal-testing--tiny-random-gpt2",
        "snapshots",
        "<commit_sha>",
    )
    print(f"  # {fake_snap}")
    print("  # Contains only *.json files (allow_patterns filter applied)")
else:
    snap_dir = snap_result
    print(f"  Snapshot directory: {snap_dir}")
    json_files = [f for f in os.listdir(snap_dir) if f.endswith(".json")]
    other_files = [f for f in os.listdir(snap_dir) if not f.endswith(".json")]
    print(f"  JSON files  : {json_files}")
    print(f"  Other files : {other_files}")
    print(f"  (allow_patterns=['*.json'] means only .json were fetched)")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: cache_dir parameter and cache path structure
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 3 — Custom cache_dir and cache path structure")

import tempfile
tmpdir = tempfile.mkdtemp(prefix="hf_demo_cache_")
print(f"Using custom cache_dir: {tmpdir}")

ok4, result4 = safe(
    hf_hub_download,
    repo_id   = REPO_ID,
    filename  = FILENAME,
    cache_dir = tmpdir,
)

if not ok4:
    note_skip(f"network unavailable — showing cache layout conceptually\n  ({result4})")
    print("\nCache layout (when a download succeeds):")
    print("  <cache_dir>/")
    print("  └── models--hf-internal-testing--tiny-random-gpt2/")
    print("      ├── blobs/          ← raw content, named by SHA-256")
    print("      ├── refs/")
    print("      │   └── main        ← resolved commit SHA")
    print("      └── snapshots/")
    print("          └── <commit_sha>/")
    print("              └── config.json  → symlink to ../../blobs/<sha>")
else:
    local_path4 = result4
    print(f"  Downloaded to : {local_path4}")
    # Walk the cache tree
    for root, dirs, files in os.walk(tmpdir):
        level = root.replace(tmpdir, "").count(os.sep)
        indent = "  " + "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for f in files:
            sub = "  " + "  " * (level + 1)
            full = os.path.join(root, f)
            is_sym = os.path.islink(full)
            print(f"{sub}{f}{'  → symlink' if is_sym else ''}")

print(
    "\nKey points:"
    "\n  • hf_hub_download returns a local absolute path."
    "\n  • Re-calling with same args returns cached path instantly."
    "\n  • Blobs are content-addressed (SHA-256); identical files stored once."
    "\n  • cache_dir defaults to ~/.cache/huggingface/hub"
    "\n  • Override globally: HF_HOME env variable."
)

print("\n[done] 01.download.py — exit 0")

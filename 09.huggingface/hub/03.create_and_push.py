"""
03.create_and_push.py — create_repo, upload_file, upload_folder, push_to_hub

This script is OFFLINE-SAFE: it only makes real Hub API calls when HF_TOKEN
is present in the environment.  Without a token it explains each step and
exits 0.

Official docs:
  https://huggingface.co/docs/huggingface_hub/guides/upload
  https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.create_repo
  https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_file
  https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_folder
"""

import os
import random
import sys
import tempfile

import numpy as np

random.seed(0)
np.random.seed(0)

from _lib import banner, note_skip, safe
from huggingface_hub import HfApi

# ─────────────────────────────────────────────────────────────────────────────
# Token detection
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HAS_TOKEN = bool(HF_TOKEN)

banner("Hugging Face Hub — create_repo / upload / push")

if HAS_TOKEN:
    print("HF_TOKEN detected — will attempt LIVE Hub operations.")
else:
    print("[skip] no HF_TOKEN — showing the API; not pushing")
    print("       Set HF_TOKEN=hf_... to enable live operations.\n")

api = HfApi(token=HF_TOKEN if HAS_TOKEN else None)

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: create_repo
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 1 — create_repo")

print(
    "\ncreate_repo creates a new repository on the Hub (model, dataset, or space).\n"
    "  • repo_type: 'model' | 'dataset' | 'space'\n"
    "  • private=True makes the repo private (requires paid plan for some types)\n"
    "  • exist_ok=True prevents an error if the repo already exists.\n"
)

print("Code:")
print("  url = api.create_repo(")
print('      repo_id   = "my-username/my-demo-model",')
print('      repo_type = "model",')
print("      private   = True,")
print("      exist_ok  = True,")
print("  )")
print("  print(url)  # https://huggingface.co/my-username/my-demo-model")

if HAS_TOKEN:
    # Determine username
    ok_who, whoami_info = safe(api.whoami)
    if ok_who:
        username = whoami_info["name"]
        REPO_ID = f"{username}/hf-hub-demo-{os.getpid()}"

        ok_cr, cr_result = safe(
            api.create_repo,
            repo_id   = REPO_ID,
            repo_type = "model",
            private   = True,
            exist_ok  = True,
        )
        if ok_cr:
            print(f"\n  LIVE: Created repo at {cr_result}")
        else:
            note_skip(f"create_repo failed: {cr_result}")
            REPO_ID = None
    else:
        note_skip(f"whoami failed: {whoami_info}")
        REPO_ID = None
else:
    REPO_ID = None

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: upload_file
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 2 — upload_file")

print(
    "\nupload_file uploads a single local file to a specific path inside a repo.\n"
)

print("Code:")
print("  api.upload_file(")
print('      path_or_fileobj = "/local/model.safetensors",')
print('      path_in_repo    = "model.safetensors",')
print('      repo_id         = "my-username/my-demo-model",')
print('      repo_type       = "model",')
print('      commit_message  = "Add model weights",')
print("  )")

if HAS_TOKEN and REPO_ID:
    # Create a dummy file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="hf_demo_"
    ) as f:
        f.write('{"demo": true, "source": "03.create_and_push.py"}\n')
        tmp_file = f.name

    ok_up, up_result = safe(
        api.upload_file,
        path_or_fileobj = tmp_file,
        path_in_repo    = "demo.txt",
        repo_id         = REPO_ID,
        repo_type       = "model",
        commit_message  = "Demo upload from 03.create_and_push.py",
    )
    os.unlink(tmp_file)

    if ok_up:
        print(f"\n  LIVE: Uploaded demo.txt to {REPO_ID}")
        print(f"  Commit URL: {up_result}")
    else:
        note_skip(f"upload_file failed: {up_result}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: upload_folder
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 3 — upload_folder")

print(
    "\nupload_folder uploads all files in a local directory (recursively).\n"
    "Use allow_patterns / ignore_patterns to filter.\n"
)

print("Code:")
print("  api.upload_folder(")
print('      folder_path     = "/local/output_dir/",')
print('      repo_id         = "my-username/my-demo-model",')
print('      repo_type       = "model",')
print('      ignore_patterns = ["*.tmp", "__pycache__/*"],')
print('      commit_message  = "Add full model directory",')
print("  )")

if HAS_TOKEN and REPO_ID:
    # Create a tiny temp folder with a couple files
    tmpdir = tempfile.mkdtemp(prefix="hf_folder_demo_")
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        f.write('{"model_type": "demo"}\n')
    with open(os.path.join(tmpdir, "README.md"), "w") as f:
        f.write("# Demo model\nCreated by 03.create_and_push.py\n")

    ok_fold, fold_result = safe(
        api.upload_folder,
        folder_path    = tmpdir,
        repo_id        = REPO_ID,
        repo_type      = "model",
        commit_message = "Demo folder upload",
    )

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    if ok_fold:
        print(f"\n  LIVE: Uploaded folder to {REPO_ID}")
    else:
        note_skip(f"upload_folder failed: {fold_result}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: push_to_hub (transformers / datasets integration)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 4 — push_to_hub (transformers model / tokenizer)")

print(
    "\nMost transformers / datasets objects support push_to_hub directly:\n"
)

print("  from transformers import AutoModel, AutoTokenizer")
print("  model     = AutoModel.from_pretrained('bert-base-uncased')")
print("  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')")
print("  model.push_to_hub('my-username/my-bert-fine-tuned')")
print("  tokenizer.push_to_hub('my-username/my-bert-fine-tuned')")
print()
print("  # For datasets:")
print("  from datasets import load_dataset")
print("  ds = load_dataset('glue', 'mrpc')")
print("  ds.push_to_hub('my-username/my-glue-mrpc')")
print()
print(
    "push_to_hub internally calls upload_folder after serialising the object.\n"
    "It respects the currently logged-in token (set via login() or HF_TOKEN env var)."
)

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: Clean up demo repo (optional)
# ─────────────────────────────────────────────────────────────────────────────
if HAS_TOKEN and REPO_ID:
    banner("Part 5 — Clean-up: delete demo repo")
    ok_del, del_result = safe(
        api.delete_repo,
        repo_id   = REPO_ID,
        repo_type = "model",
    )
    if ok_del:
        print(f"  LIVE: Deleted demo repo {REPO_ID}")
    else:
        note_skip(f"delete_repo failed (may need manual cleanup): {del_result}")

print("\n[done] 03.create_and_push.py — exit 0")

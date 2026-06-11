# 09 · The Hugging Face Ecosystem

A dense, runnable tour of the Hugging Face stack: the **Hub** (the artifact
store) and the Python libraries that read from and write to it. Read the three
top-level docs in order, then dive into any library folder.

| Read in order | What it covers |
|---|---|
| [`00.manifest.md`](00.manifest.md) | What the Hub *is*: models/datasets/spaces/papers, naming, model cards, gated/licensed models, search, download, the cache, `HF_HOME`, auth. |
| [`01.architecture.md`](01.architecture.md) | The ecosystem map: how Hub + the libraries fit together, the cache/download workflow, `Auto*` resolution, pipelines vs. manual vs. Trainer. |
| [`02.workflow.md`](02.workflow.md) | End-to-end: search → load → preprocess → fine-tune (Trainer/PEFT/TRL) → evaluate → push → serve. |

## Library folders (each: full-feature `README.md` + numbered runnable examples)

| Folder | Layer | Highlights |
|---|---|---|
| [`transformers/`](transformers/) | models & training | pipelines, `Auto*`, generation strategies, attention/hidden states, `Trainer`, tasks, quantization (doc) |
| [`datasets/`](datasets/) | data access | `load_dataset`, streaming, map/filter/cast, Arrow + memory-mapping, custom datasets |
| [`tokenizers/`](tokenizers/) | text → tokens | train BPE & WordPiece from scratch, encode/decode, special tokens, offsets |
| [`accelerate/`](accelerate/) | hardware | device placement, mixed precision, gradient accumulation, `accelerate` config (doc + CPU loop) |
| [`peft/`](peft/) | efficient tuning | LoRA fine-tune (runnable), adapters merge/save |
| [`trl/`](trl/) | alignment | `SFTTrainer` (runnable), DPO loss (concept + math) |
| [`diffusers/`](diffusers/) | diffusion | tiny local UNet + scheduler denoising loop, DDPM math |
| [`hub/`](hub/) | Hub client | `hf_hub_download`, `snapshot_download`, search/info, create/push (offline-safe), model cards |
| [`sentence-transformers/`](sentence-transformers/) | embeddings | sentence vectors + semantic search on a tiny corpus |

## Conventions (per [`.claude/skills/tutorial-architect`](../.claude/skills/tutorial-architect/SKILL.md))

- **Runnable & offline-safe.** Every `.py` exits 0 with `python file.py` even
  with no network: model/dataset loading is wrapped in `try/except` and degrades
  to a tiny local fallback that still demonstrates the API shape. Each folder has
  a small `_lib.py` with the `banner` / `note_skip` / `safe` helpers.
- **Tiny models, CPU-fast, seeded.** Examples use the smallest checkpoints
  (`sshleifer/tiny-gpt2`, `prajjwal1/bert-tiny`, `hf-internal-testing/tiny-*`,
  `all-MiniLM-L6-v2`), set `torch.set_num_threads(1)`, and seed everything.
- **Explainers, not quickstarts.** Each `README.md` derives the concept, tours
  *every* major feature, names the pitfalls, and cites official docs.

Install everything for the section: `pip install -r requirements.txt`
(per-folder `requirements.txt` files pin the subset each tutorial needs).

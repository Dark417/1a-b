# AGENTS.md — Format & Rules

This file tells **any AI coding agent** (Claude Code, Codex, Cursor, Aider, …)
how to contribute to this repository. It is intentionally tool-agnostic: nothing
here is specific to a single assistant. When a rule says "the agent", it means
whichever AI is currently editing the repo, plus human contributors.

This repository is a **dense, comprehensive ML/AI tutorial**. The goal is not a
production library — it is a *teaching* codebase where each algorithm is
implemented twice (from-scratch NumPy + idiomatic PyTorch) and explained with
the underlying mathematics, like the essence of a good university course
condensed into runnable code.

---

## 0. Golden rules

1. **The [`MAP.md`](MAP.md) file is the source of truth.** Always add an
   algorithm to the map *before* writing its files, and update its status box.
2. **Two artifacts per algorithm**, sharing the same base name:
   - `name.py` — runnable module with **both** a NumPy and a PyTorch
     implementation.
   - `name.ipynb` — the same code, *plus* concept explanation and math
     derivation.
3. **Tutorial first.** Clarity beats cleverness. Prefer readable loops with
   comments over vectorized one-liners *when the vectorized form hides the
   idea*. (Show the vectorized version too, when it teaches something.)
4. **Comprehensive, not shallow.** Cover the major algorithm, its **variants**,
   and the **training techniques** that matter for it (see §5).
5. **Self-contained.** Each file runs on its own: `python ml/.../name.py`
   produces output / a plot / printed metrics. No hidden global state.

---

## 1. Directory layout

```
category/                     # ml, dl, generative-models, nlp, transformers
  sub-category/               # e.g. linear-models, gan, attention
    extra-layer/              # OPTIONAL — only when a family is large
      algorithm.py
      algorithm.ipynb
```

- Add an **extra layer** only when a sub-category holds many related models
  (e.g. `generative-models/gan/` already is that layer; a future
  `gan/conditional/` would be a third layer). Do not over-nest.
- Folder names: lowercase, `kebab-case`. File/base names: lowercase,
  `snake_case` (so they are importable Python modules).
- Each top-level category has a `README.md` indexing its contents.

---

## 2. The `.py` file format

Every algorithm module follows this skeleton (see
[`common/template.py`](common/template.py) for the canonical copy):

```python
"""
<Algorithm Name>
================
One-paragraph summary: what it is, what problem it solves, key idea.

Variants implemented here:
    - <variant A>
    - <variant B>

Training techniques demonstrated:
    - <technique> (see training-techniques/README.md)

References:
    - <paper / textbook chapter>
"""

import numpy as np

# ----------------------------------------------------------------------------
# 1. NumPy implementation (from scratch — no autograd)
# ----------------------------------------------------------------------------
class AlgorithmNumPy:
    """Plain-NumPy reference implementation. The math made explicit."""
    ...

# ----------------------------------------------------------------------------
# 2. PyTorch implementation (idiomatic — uses autograd / nn.Module)
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn

class AlgorithmTorch(nn.Module):
    ...

# ----------------------------------------------------------------------------
# 3. Demo — runs when called directly
# ----------------------------------------------------------------------------
def demo():
    """Generate/lo­ad toy data, fit both implementations, compare, visualize."""
    ...

if __name__ == "__main__":
    demo()
```

**Rules for `.py` files**
- The NumPy version is the *teaching* version: implement forward, loss, and the
  gradient/update **by hand**. Comment each non-obvious line with the math.
- The PyTorch version is the *idiomatic* version: use `nn.Module`, autograd,
  and an `optim` optimizer. It should match the NumPy result on the same toy
  data (assert closeness where feasible).
- Keep GPU-awareness: pick device via
  `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  and `.to(device)` tensors/models. See [`docs/gpu-setup.md`](docs/gpu-setup.md).
- Set seeds (`np.random.seed`, `torch.manual_seed`) for reproducibility.
- Prefer tiny synthetic datasets or `sklearn.datasets` toys so files run in
  seconds on CPU.

---

## 3. The `.ipynb` notebook format

The notebook mirrors the `.py` file but is the **course slide deck in code**.
Standard cell order:

1. **Title + intuition** (markdown) — what & why, a picture-in-words.
2. **Concept explanation** (markdown) — the essence, like lecture slides:
   problem setup, model, loss, why it works, when it fails.
3. **Math derivation** (markdown, LaTeX) — derive the objective and its
   gradient/update step by step. Don't skip the algebra; this is the point.
4. **NumPy implementation** (code) — copied from the `.py`, runnable.
5. **PyTorch implementation** (code) — copied from the `.py`, runnable.
6. **Training & techniques** (code + markdown) — train on toy data; explicitly
   illustrate the relevant training technique (e.g. plot gradient norms to show
   vanishing gradients).
7. **Visualization & discussion** (code + markdown) — plots, decision
   boundaries, loss curves; variants comparison; takeaways and pitfalls.

**Rules for notebooks**
- Math uses LaTeX: `$...$` inline, `$$...$$` block.
- Every notebook must run top-to-bottom without error on CPU.
- Keep cell outputs out of version control where practical (clear before
  commit), or keep them small.
- The notebook and the `.py` should not drift: the code cells are the same code.

---

## 4. Variants

Each algorithm file must cover its **major variants** in the same file (small
variants as flags/subclasses; clearly distinct variants as separate classes).
List them in the module docstring and in `MAP.md`. Examples:
- `linear_regression` → OLS, Ridge, Lasso, ElasticNet.
- `kmeans` → Lloyd, k-means++, mini-batch.
- `gan` family → each major variant gets its **own file** under `gan/`.

Rule of thumb: a variant that changes the *objective* or *architecture*
materially → own file; a variant that changes a *hyperparameter / penalty /
init* → a flag inside the file.

---

## 5. Training techniques

Training techniques (vanishing/exploding gradients, gradient clipping,
weight init, batch/layer norm, dropout, residual connections, LR warmup,
reparameterization, label smoothing, teacher forcing, negative sampling, …)
are **woven into the algorithms** where they naturally arise — not isolated
files.

- The **same technique may appear in several algorithms**. That repetition is
  intentional and good for a tutorial.
- Each technique has a canonical "home" demo recorded in the table in
  [`training-techniques/README.md`](training-techniques/README.md); when you use
  it elsewhere, link back to that write-up rather than re-deriving it in full.
- When a model is a natural showcase for a technique (e.g. RNN ↔ vanishing
  gradients, ResNet ↔ skip connections), make that demonstration *explicit*:
  measure it, plot it, explain the fix.

---

## 6. Dependencies & running

- Core stack: `numpy`, `torch`, `matplotlib`, `scikit-learn` (toy datasets),
  `jupyter`. See [`requirements.txt`](requirements.txt).
- Run a module: `python <category>/<sub>/<name>.py`.
- Open a notebook: `jupyter lab <category>/<sub>/<name>.ipynb`.
- GPU is optional; everything must run on CPU. See
  [`docs/gpu-setup.md`](docs/gpu-setup.md) for CUDA/MPS setup.

---

## 7. Workflow for adding an algorithm (checklist)

1. [ ] Add/locate the entry in `MAP.md`; set status to `[~]`.
2. [ ] Copy `common/template.py` → `category/sub/name.py`; implement NumPy +
       PyTorch + demo + variants.
3. [ ] Verify it runs: `python category/sub/name.py`.
4. [ ] Create `name.ipynb` with the 7-section structure (§3); ensure it runs
       top-to-bottom.
5. [ ] Demonstrate the relevant training technique(s) explicitly.
6. [ ] Update the category `README.md` index and set `MAP.md` status to `[x]`.
7. [ ] Commit with a clear message: `add <category>/<name> (numpy+torch+nb)`.

---

## 8. Style

- Python: PEP 8, type hints on public functions, docstrings on classes.
- Comments explain the **math/intuition**, not the obvious syntax.
- Determinism: seed everything.
- No network downloads at import time; generate or use bundled toy data.

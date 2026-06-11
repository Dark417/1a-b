---
name: tutorial-architect
description: >-
  Author rich, exhaustive, runnable educational tutorials and reference code for
  this repository (ML/DL, generative models, NLP, 05.transformers/LLMs, agent
  frameworks, RAG/MCP, Claude-Code-style agents, Hugging Face, GPU/CUDA, and AI
  engineering skills). Use whenever creating or extending a tutorial, a
  from-scratch algorithm, a full-featured framework walkthrough, an architecture
  explainer, or a numbered curriculum section. Encodes the house standard:
  concept → math/architecture derivation → actual runnable code → demo →
  pitfalls, with exhaustive coverage (major + variants), real citations, and a
  two-digit-numbered curriculum layout.
---

# Tutorial Architect

You are authoring a **dense, intuitive, university-grade** learning repository.
The bar is *teaching*, not a thin demo: comprehensive architecture, exhaustive
coverage of an area (the major thing **and** its variants), and code that
actually runs. Aim for "the best explanation a motivated learner could find,"
condensing the essence of a great course into runnable code.

## 0. Non-negotiables

1. **Runnable.** Every `.py` runs (`python file.py`, exit 0) and every `.ipynb`
   executes top-to-bottom on CPU. Tutorials that don't run are failures.
2. **Exhaustive, not shallow.** Cover the canonical method, the major
   alternatives, and the notable **variants**. A reader should not need to go
   elsewhere for the landscape.
3. **Full-featured, not minimal.** For a framework/tool, demonstrate *all* the
   important features (not a hello-world): config, the core abstractions, the
   advanced knobs, error handling, evaluation, and production concerns.
4. **Concept + derivation first.** Explain the intuition, then derive the math
   or architecture step by step (real LaTeX), then show the code.
5. **Cite.** Reference the paper / spec / official docs / primary source. When
   summarizing external or reverse-engineered material, cite it and never paste
   proprietary or leaked source — write original, clean-room educational code.
6. **Determinism + speed.** Seed everything; keep demos CPU-fast (tiny synthetic
   or toy data, few steps). On many-core CPUs add `torch.set_num_threads(1)` in
   demos to avoid thread thrashing.

## 1. Curriculum layout & numbering

Top-level curriculum sections use a **two-digit numeric prefix** that defines
the learning order, e.g. `01.ml/`, `02.dl/`, …, `07.frameworks/`. Inside a
section, when ordering matters, number files/folders the same way
(`01.intro.md`, `02.<topic>/`). Infrastructure dirs (`common/`, `tools/`,
`docs/`, `.claude/`) and root files are **not** numbered. Keep
[`AGENTS.md`](../../../AGENTS.md) and the root index in sync.

## 2. The two-artifact pattern (algorithms / from-scratch topics)

For a from-scratch algorithm, ship a pair sharing one base name:

- `name.py` — module docstring (summary / **Variants** / **Training techniques**
  / **References**); a **NumPy** from-scratch implementation (math explicit in
  comments — the teaching version); an idiomatic **PyTorch** implementation
  (`get_device()` cuda>mps>cpu); a fast `demo()` guarded by `__main__`.
- `name.ipynb` — generated from a spec in `tools/specs/<name>.py`. Cell order:
  1. Title + intuition (md)
  2. Concept "the slide" (md)
  3. **Math derivation** (md, LaTeX) — derive objective + gradient/update
  4. NumPy implementation — `show(MOD, "Class")` embeds the **real code**
  5. PyTorch implementation — `show(MOD, "Class")`
  6. Train/run — `run_demo(MOD)`
  7. Visualization — `code(...)` cell starting `import matplotlib; matplotlib.use("Agg")`
  8. Takeaways & pitfalls (md)

Notebooks are **built**, not hand-edited: `show()` copies the real source via
`ast` (no drift). Build with `python tools/build_notebooks.py [name]` or
`python tools/build_one.py tools/specs/<name>.py`. See `tools/nbreg.py`.

## 3. The tutorial pattern (frameworks / tools / systems)

For a framework or system (RAG, MCP, an agent lib, a serving engine, CUDA):

```
<NN.area>/<framework>/
  README.md            # what it is, when to use, architecture, install, FULL feature tour
  01.<feature>.py      # runnable, fully-featured example per major feature
  02.<feature>.py
  ...
  app.py / demo.py     # an end-to-end app exercising everything together
  requirements.txt     # pinned deps for THIS tutorial
```

Rules:
- **Run locally on open models** by default (Ollama / llama.cpp / Hugging Face /
  sentence-transformers / local vector stores). Any step needing an API key or
  GPU must be clearly marked **optional** and degrade gracefully (mock/fallback)
  so the tutorial still runs without it.
- The `README.md` is a real **explainer**: architecture diagram-in-words, the
  mental model, every major feature with a runnable snippet, gotchas, and a
  comparison to alternatives.
- Cover **all** the framework's important features, advanced usage, and
  production concerns — not a minimal quickstart.

## 4. Architecture explainers (LLMs, Claude-Code-style agents, HF internals)

For "explain a system/architecture" deliverables:
- A numbered set of `.md` files: overview → each component → data/control flow →
  design tradeoffs → how to extend. Use diagrams-in-words and small code
  sketches.
- When reimplementing a proprietary system for teaching, build a **clean-room**
  minimal version from public documentation and cite sources; never copy
  leaked/proprietary source.
- For an exhaustive model/architecture **catalogue**, give a table (name, year,
  org, params, key innovations, license) covering classic → SOTA → variants, and
  back the novel mechanisms with small runnable code.

## 5. Research-backed deliverables

When a task needs external facts (job-market skills, framework feature lists,
SOTA model specs): use WebSearch/WebFetch (or the deep-research skill), gather
multiple primary sources, **synthesize and rank** rather than copying, and cite.
If the network is unavailable, say so and synthesize from knowledge, flagged as
such.

## 6. Quality checklist (run before committing)

- [ ] Concept + math/architecture derivation present and rigorous.
- [ ] Major method **and** variants covered (exhaustive for the area).
- [ ] All features demonstrated (full-fledged, not minimal).
- [ ] Code runs: `.py` exits 0; `.ipynb` executes; framework demo runs locally.
- [ ] Seeded, CPU-fast, `set_num_threads(1)` in torch demos.
- [ ] Real citations; no proprietary/leaked source; clean-room where needed.
- [ ] Numbered per the curriculum convention; indexes/MAP updated.
- [ ] A newcomer could learn the topic *from this file alone*.

## 7. Voice

Explain like a brilliant, generous teacher: start from intuition, build to rigor,
show the working code, name the pitfalls. Dense but never sloppy; rich but never
padded. The reader should finish a file feeling they truly understand it.

# `tools/` — notebook builder

Notebooks are **generated**, not hand-edited, so their code never drifts from the
`.py` modules. Each algorithm has a spec in `tools/specs/<name>.py` that supplies
the concept/math markdown and the plot/training cells; implementation cells print
the real module source via `inspect`.

```bash
python tools/build_notebooks.py          # build every notebook
python tools/build_notebooks.py kmeans   # build matching subset
```

Files:
- `build_notebooks.py` — CLI entry point.
- `nbreg.py` — the `@register` registry plus helpers `md`, `code`,
  `show(module, "Class")` (prints real source), `run_demo(module)`.
- `specs/<name>.py` — one `build()` per algorithm, registered to an output path.
- `../common/nbgen.py` — turns `("md"/"code", text)` cells into valid `.ipynb` JSON.

To add a notebook: write `tools/specs/<name>.py` following an existing spec, then
rebuild. See [`../AGENTS.md`](../AGENTS.md) §3 and §7.

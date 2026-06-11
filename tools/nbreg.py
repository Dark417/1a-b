"""Registry + helpers shared by the notebook builder and the per-algo specs.

`show(module, *names)` makes a notebook embed the **actual implementation code**
(copied verbatim from the module's `.py`), not a reflection/`print` trick. The
real source is extracted at build time via the `ast` module:

- the first `show(...)` cell for a module also carries that module's *preamble*
  (imports, constants, module-level helpers, the `demo()` and data helpers) so
  the notebook is self-contained and runs top-to-bottom;
- later `show(...)` cells carry just their requested classes/functions.

`finalize(cells)` resolves the show-markers into concrete code cells; both
builders (`build_notebooks.py`, `build_one.py`) call it before writing.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.nbgen import md, code, write_notebook  # noqa: E402,F401

# name -> (output_path, builder() -> list[cell])
BUILDERS: dict[str, tuple[str, "callable"]] = {}

# Where algorithm modules live (for locating a module's source file).
_CODE_DIRS = ["01.ml", "02.dl", "03.generative-models", "04.nlp",
              "05.transformers", "common"]


def register(name: str, path: str):
    def deco(fn):
        BUILDERS[name] = (path, fn)
        return fn
    return deco


def show(module: str, *qualnames: str):
    """Marker: embed the real source of these top-level names from `module`.

    Resolved by `finalize()` into an actual, runnable code cell.
    """
    return ("__show__", module, list(qualnames))


def run_demo(module: str) -> tuple[str, str]:
    # demo() is embedded by the first show() cell, so just call it.
    return code("demo()")


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------
def _find_module_file(module: str) -> Path:
    for d in _CODE_DIRS:
        hits = list((ROOT / d).rglob(f"{module}.py"))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"could not locate {module}.py under {_CODE_DIRS}")


def _node_name(node) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    return None


def _is_main_guard(node) -> bool:
    return isinstance(node, ast.If) and "__main__" in ast.dump(node.test)


def _chunk_module(module: str, name_lists: list[list[str]]) -> list[str]:
    """Split a module's source (in ORIGINAL order) into one chunk per show-cell.

    Chunk k contains every top-level node up to and including the last class /
    function requested by show-cell k. This keeps source order intact, so
    module-level code that references a class (registries, aliases) still runs
    after that class is defined. Imports/helpers naturally land in chunk 0
    (they precede the first requested target); trailing nodes join the last chunk.
    """
    path = _find_module_file(module)
    text = path.read_text(encoding="utf-8")
    body = list(ast.parse(text).body)

    # drop the module docstring (redundant with the notebook markdown)
    if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant) and isinstance(
            body[0].value.value, str):
        body = body[1:]
    body = [n for n in body if not _is_main_guard(n)]   # drop `if __name__` demo guard

    def seg(node):
        s = ast.get_source_segment(text, node)
        return s if s is not None else ""

    chunks: list[list[str]] = []
    cur: list[str] = []
    ci = 0
    remaining = set(name_lists[0]) if name_lists else set()
    for node in body:
        cur.append(seg(node))
        nm = _node_name(node)
        if remaining is not None and nm in remaining:
            remaining.discard(nm)
            if not remaining:                          # cell ci fully collected
                chunks.append(cur); cur = []; ci += 1
                remaining = set(name_lists[ci]) if ci < len(name_lists) else None
    if cur:                                            # trailing nodes
        if chunks:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    while len(chunks) < len(name_lists):               # pad if a target was missing
        chunks.append([])

    header = f"# ===== actual implementation from {module}.py ====="
    return [header + "\n" + "\n\n".join(p for p in ch if p.strip()) for ch in chunks]


def finalize(cells: list) -> list:
    """Resolve show-markers into real, source-ordered code cells."""
    order: dict[str, list[list[str]]] = {}
    for c in cells:
        if isinstance(c, tuple) and len(c) == 3 and c[0] == "__show__":
            _, mod, names = c
            order.setdefault(mod, []).append(names)
    chunks = {mod: _chunk_module(mod, nls) for mod, nls in order.items()}
    idx = {mod: 0 for mod in chunks}

    out = []
    for c in cells:
        if isinstance(c, tuple) and len(c) == 3 and c[0] == "__show__":
            _, mod, _names = c
            i = idx[mod]; idx[mod] += 1
            out.append(code(chunks[mod][i]))
        else:
            out.append(c)
    return out

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
_CODE_DIRS = ["ml", "dl", "generative-models", "nlp", "transformers", "common"]


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


def _render_source(module: str, names: list[str], all_targets: set[str],
                   include_preamble: bool) -> str:
    path = _find_module_file(module)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    body = list(tree.body)

    # drop the module docstring (it's redundant with the notebook markdown)
    if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant) and isinstance(
            body[0].value.value, str):
        body = body[1:]

    def seg(node):
        s = ast.get_source_segment(text, node)
        return s if s is not None else ""

    # requested classes/functions, in the order the spec asked for them
    by_name = {_node_name(n): n for n in body if _node_name(n)}
    target_src = "\n\n\n".join(seg(by_name[n]) for n in names if n in by_name)

    parts = []
    if include_preamble:
        pre = [seg(n) for n in body
               if _node_name(n) not in all_targets and not _is_main_guard(n)]
        pre = [p for p in pre if p.strip()]
        parts.append("\n\n".join(pre))
    parts.append(target_src)

    header = f"# ===== actual implementation from {module}.py ====="
    return header + "\n" + "\n\n\n".join(p for p in parts if p.strip())


def finalize(cells: list) -> list:
    """Resolve show-markers into real code cells (with one-time preamble)."""
    targets: dict[str, set[str]] = {}
    for c in cells:
        if isinstance(c, tuple) and len(c) == 3 and c[0] == "__show__":
            _, mod, names = c
            targets.setdefault(mod, set()).update(names)

    seen: set[str] = set()
    out = []
    for c in cells:
        if isinstance(c, tuple) and len(c) == 3 and c[0] == "__show__":
            _, mod, names = c
            out.append(code(_render_source(
                mod, names, targets[mod], include_preamble=mod not in seen)))
            seen.add(mod)
        else:
            out.append(c)
    return out

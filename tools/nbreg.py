"""Registry + helpers shared by the notebook builder and the per-algo specs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.nbgen import md, code, write_notebook  # noqa: E402,F401

# name -> (output_path, builder() -> list[cell])
BUILDERS: dict[str, tuple[str, "callable"]] = {}


def register(name: str, path: str):
    def deco(fn):
        BUILDERS[name] = (path, fn)
        return fn
    return deco


def show(module: str, *qualnames: str) -> tuple[str, str]:
    """Code cell printing the *real* source of objects from `module` (no drift)."""
    targets = ", ".join(f"M.{q}" for q in qualnames)
    return code(
        f"import inspect, {module} as M\n"
        f"for _obj in [{targets}]:\n"
        f"    print(inspect.getsource(_obj))"
    )


def run_demo(module: str) -> tuple[str, str]:
    return code(f"import {module} as M\nM.demo()")

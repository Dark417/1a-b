"""Importing this package registers every notebook spec.

Each spec is imported defensively: a single broken/half-written spec (e.g. while
another contributor is editing it) will not break building the others.
Add a module here automatically by dropping a `<name>.py` in this folder.
"""

from importlib import import_module
from pathlib import Path

_here = Path(__file__).parent
for _f in sorted(_here.glob("*.py")):
    if _f.stem.startswith("_"):
        continue
    try:
        import_module(f"tools.specs.{_f.stem}")
    except Exception as _e:  # pragma: no cover - resilience for concurrent edits
        print(f"[specs] skipped {_f.name}: {_e}")

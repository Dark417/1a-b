"""Importing this package registers every notebook spec.

Add a module here when you add an algorithm notebook spec.
"""

from importlib import import_module
from pathlib import Path

_here = Path(__file__).parent
for _f in sorted(_here.glob("*.py")):
    if _f.stem.startswith("_"):
        continue
    import_module(f"tools.specs.{_f.stem}")

"""
Build a SINGLE notebook from one spec file, in isolation.

Unlike `build_notebooks.py` (which imports the whole `tools.specs` package),
this loads only the given spec file, so it is safe to run while other specs are
being written concurrently. Used by contributors/agents adding one algorithm:

    python tools/build_one.py tools/specs/<name>.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.nbreg import BUILDERS, write_notebook, finalize  # noqa: E402


def main(spec_path: str):
    p = Path(spec_path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)            # @register populates BUILDERS
    if not BUILDERS:
        raise SystemExit(f"{spec_path} registered no builders")
    for name, (path, fn) in BUILDERS.items():
        out = ROOT / path
        out.parent.mkdir(parents=True, exist_ok=True)
        write_notebook(str(out), finalize(fn()))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/build_one.py tools/specs/<name>.py")
    main(sys.argv[1])

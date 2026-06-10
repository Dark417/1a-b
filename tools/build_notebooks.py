"""
Notebook builder.

Each algorithm's `.ipynb` is generated from a spec so the math/concept prose
stays readable and the *code* never drifts from the `.py` module: code cells
import the sibling module and print real source via `inspect`, then run the
demo. Run from the repo root:

    python tools/build_notebooks.py            # build all
    python tools/build_notebooks.py kmeans     # build matching subset
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.nbreg import BUILDERS, write_notebook, finalize  # noqa: E402
import tools.specs  # noqa: E402,F401  (populates BUILDERS)


def main(argv):
    wanted = argv[1:]
    built = 0
    for name, (path, fn) in BUILDERS.items():
        if wanted and not any(w in name for w in wanted):
            continue
        out = ROOT / path
        out.parent.mkdir(parents=True, exist_ok=True)
        write_notebook(str(out), finalize(fn()))
        built += 1
    print(f"\n{built} notebook(s) built.")


if __name__ == "__main__":
    main(sys.argv)

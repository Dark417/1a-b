"""
Tiny Jupyter-notebook generator.

Authoring notebooks as raw JSON is painful, so each tutorial ships a small
build script that calls `write_notebook(path, cells)` where `cells` is a list
of `("md", text)` / `("code", text)` tuples. This keeps the *content* readable
while still producing a valid `.ipynb`.

Usage:
    from common.nbgen import md, code, write_notebook
    write_notebook("foo.ipynb", [md("# Title"), code("print('hi')")])
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> tuple[str, str]:
    return ("md", text)


def code(text: str) -> tuple[str, str]:
    return ("code", text)


def _lines(text: str) -> list[str]:
    """nbformat stores source as a list of lines, each keeping its newline."""
    text = text.strip("\n")
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def write_notebook(path: str, cells: list[tuple[str, str]]) -> None:
    nb_cells = []
    for i, (kind, text) in enumerate(cells):
        cell_id = f"c{i:03d}"  # stable cell id (nbformat 4.5+ requires one)
        if kind == "md":
            nb_cells.append({
                "id": cell_id,
                "cell_type": "markdown",
                "metadata": {},
                "source": _lines(text),
            })
        else:
            nb_cells.append({
                "id": cell_id,
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": _lines(text),
            })
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {path} ({len(nb_cells)} cells)")

"""app.py · Evaluate a tiny RAG dataset on all four RAGAS metrics (offline).

End-to-end: a 3-row RAG dataset (question, answer, contexts, ground_truth) scored
on faithfulness, answer_relevancy, context_precision, context_recall — using the
from-scratch metrics from files 01–03 with the deterministic mock judge. Prints a
per-row table and the aggregate scorecard `ragas.evaluate` would return.

Run:  python app.py     (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _lib import MockEmbedder, banner  # noqa: E402


def _load(modfile: str):
    """Import a sibling file whose name starts with digits (not importable normally)."""
    path = os.path.join(HERE, modfile)
    spec = importlib.util.spec_from_file_location(modfile.replace(".", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m_faith = _load("01.faithfulness.py")
m_rel = _load("02.answer_relevancy.py")
m_ctx = _load("03.context_precision_recall.py")


# ---------------------------------------------------------------------------
# A tiny RAG evaluation dataset.
# ---------------------------------------------------------------------------
DATASET = [
    {
        "question": "Where is the Eiffel Tower and when was it built?",
        "answer": "The Eiffel Tower is in Paris and was completed in 1889.",
        "contexts": [
            "The Eiffel Tower is located in Paris, France.",
            "It was completed in 1889.",
        ],
        "ground_truth": "The Eiffel Tower is in Paris. It was completed in 1889.",
    },
    {
        "question": "Who developed the theory of relativity?",
        "answer": "Albert Einstein developed the theory of relativity, and he also invented the telephone.",
        "contexts": [
            "Albert Einstein developed the theory of relativity.",
            "The telephone was invented by Alexander Graham Bell.",
        ],
        "ground_truth": "Albert Einstein developed the theory of relativity.",
    },
    {
        "question": "What is the boiling point of water at sea level?",
        "answer": "Water boils at 100 degrees Celsius at sea level.",
        "contexts": [
            "At standard atmospheric pressure, water boils at 100 degrees Celsius.",
            "Mount Everest is the tallest mountain on Earth.",
        ],
        "ground_truth": "Water boils at 100 degrees Celsius at sea level.",
    },
]


def evaluate_row(row: dict, emb: MockEmbedder) -> dict[str, float]:
    faith, _ = m_faith.faithfulness(row["answer"], row["contexts"])
    rel, _ = m_rel.answer_relevancy(row["question"], row["answer"], emb)
    prec, _ = m_ctx.context_precision(row["contexts"], row["ground_truth"])
    rec, _ = m_ctx.context_recall(row["contexts"], row["ground_truth"])
    return {
        "faithfulness": faith,
        "answer_relevancy": rel,
        "context_precision": prec,
        "context_recall": rec,
    }


def main() -> int:
    emb = MockEmbedder(dim=128)
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    banner("RAGAS scorecard over a 3-row RAG dataset (offline mock judge)")
    print(f"  {'row':4s} {'faith':>7s} {'a_rel':>7s} {'c_prec':>7s} {'c_rec':>7s}")
    print("  " + "-" * 36)

    totals = {m: 0.0 for m in metrics}
    for i, row in enumerate(DATASET):
        scores = evaluate_row(row, emb)
        for m in metrics:
            totals[m] += scores[m]
        print(f"  {i:<4d} {scores['faithfulness']:7.2f} {scores['answer_relevancy']:7.2f} "
              f"{scores['context_precision']:7.2f} {scores['context_recall']:7.2f}")

    banner("Aggregate (what ragas.evaluate returns)")
    n = len(DATASET)
    for m in metrics:
        print(f"  {m:20s} = {totals[m] / n:.3f}")

    print("\nRead the table: row 1 'invented the telephone' tanks faithfulness")
    print("(unsupported claim) while staying relevant — exactly the failure mode")
    print("RAGAS is designed to surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

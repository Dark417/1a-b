"""03 · Context precision & recall, from scratch (RAGAS).

These two metrics grade the *retriever*, not the generator. Both need a
``ground_truth`` (the reference answer) to decide what "relevant" means.

CONTEXT PRECISION — *are the relevant chunks ranked near the top?*
  Retrieval returns an ordered list of chunks. For each position k, mark the
  chunk relevant (1) or not (0). Then compute a **rank-weighted** precision,
  the mean of precision@k taken only at the relevant positions — this is exactly
  Average Precision (AP) from information retrieval:

      precision@k = (#relevant in top k) / k
      context_precision = Σ_k [ precision@k · rel_k ] / (#relevant total)

  Putting a relevant chunk at rank 1 scores higher than at rank 5.

CONTEXT RECALL — *did we retrieve everything the answer needs?*
  Split the ``ground_truth`` into sentences. A sentence is "covered" if it can be
  attributed to (entailed by) the retrieved context. Score = covered / total.
  Low recall ⇒ the retriever missed information the answer depends on.

We judge relevance / attribution with the deterministic ``mock_nli`` from
``_lib.py`` so the metrics run offline. RAGAS uses an LLM for both judgments.

Docs:
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/
  https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/

Run:  python 03.context_precision_recall.py    (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import banner, mock_nli  # noqa: E402


def _relevant(chunk: str, ground_truth: str) -> bool:
    """A chunk is relevant if it entails (supports) part of the ground truth."""
    return mock_nli(premise=chunk, hypothesis=ground_truth) == "entailment" or \
        mock_nli(premise=ground_truth, hypothesis=chunk) == "entailment"


def context_precision(contexts: list[str], ground_truth: str) -> tuple[float, list[int]]:
    """Rank-weighted precision (Average Precision over relevance labels)."""
    rels = [int(_relevant(c, ground_truth)) for c in contexts]
    total_rel = sum(rels)
    if total_rel == 0:
        return 0.0, rels
    ap, hits = 0.0, 0
    for k, rel in enumerate(rels, start=1):
        if rel:
            hits += 1
            ap += (hits / k)  # precision@k at this relevant position
    return ap / total_rel, rels


def context_recall(contexts: list[str], ground_truth: str) -> tuple[float, list[tuple[str, bool]]]:
    """Fraction of ground-truth sentences attributable to the retrieved context."""
    context = " ".join(contexts)
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", ground_truth) if s.strip()]
    if not sentences:
        return 1.0, []
    covered = []
    for s in sentences:
        ok = mock_nli(premise=context, hypothesis=s) == "entailment"
        covered.append((s, ok))
    score = sum(1 for _, ok in covered if ok) / len(sentences)
    return score, covered


# ---------------------------------------------------------------------------
# Demo: a retriever that returns 4 chunks, 2 relevant, ordered imperfectly.
# ---------------------------------------------------------------------------
GROUND_TRUTH = ("Marie Curie won the Nobel Prize in Physics in 1903. "
                "She later won the Nobel Prize in Chemistry in 1911.")

RETRIEVED = [
    "The Eiffel Tower is a landmark in Paris.",                       # irrelevant (rank 1)
    "Marie Curie won the Nobel Prize in Physics in 1903.",           # relevant (rank 2)
    "Curie also won the Nobel Prize in Chemistry in 1911.",          # relevant (rank 3)
    "Paris is the capital of France.",                                # irrelevant (rank 4)
]


def main() -> int:
    banner("Context precision (rank-aware) & recall (coverage of ground truth)")
    print("Ground truth:")
    print(f"  {GROUND_TRUTH}\n")

    prec, rels = context_precision(RETRIEVED, GROUND_TRUTH)
    print(f"context_precision = {prec:.3f}")
    for k, (chunk, rel) in enumerate(zip(RETRIEVED, rels), start=1):
        print(f"  rank {k}: {'REL' if rel else '---'}  {chunk}")
    print("  (a relevant chunk at a worse rank lowers precision)\n")

    rec, covered = context_recall(RETRIEVED, GROUND_TRUTH)
    print(f"context_recall = {rec:.3f}")
    for sent, ok in covered:
        print(f"  {'COVERED ' if ok else 'MISSING '} {sent}")

    print("\nLesson: precision grades RANKING quality; recall grades COVERAGE.")
    print("Both require a ground_truth to define relevance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""01 · Faithfulness, from scratch (RAGAS's flagship metric).

Faithfulness answers: *is every statement in the answer supported by the
retrieved context?* It is RAGAS's defence against hallucination, and it is
**reference-free** — you don't need a gold answer, only the answer and the
contexts it was supposed to be grounded in.

The algorithm (mirrors RAGAS):

  1. **Decompose** the answer into atomic *claims* (one fact each).
  2. For each claim, run an **NLI entailment** check against the context:
     does the context *entail* the claim?
  3. ``faithfulness = (#claims entailed) / (#claims total)``  ∈ [0, 1].

A score of 1.0 means everything the answer asserts is backed by the context; a
low score means the generator invented things ("hallucinated").

We use a deterministic **mock NLI** (``_lib.mock_nli``) so the metric runs
offline. Real RAGAS uses an LLM both to extract claims and to judge entailment;
the structure is identical.

Docs: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
Paper: Es et al. 2023, https://arxiv.org/abs/2309.15217

Run:  python 01.faithfulness.py     (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import banner, mock_nli  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1 — claim decomposition. A real judge LLM does this; we approximate by
# splitting on sentence/clause boundaries. Each fragment is one "claim".
# ---------------------------------------------------------------------------
def extract_claims(answer: str) -> list[str]:
    # Split on sentence terminators and " and "/";" conjunctions.
    parts = re.split(r"[.!?]\s+|\s+and\s+|;\s*", answer.strip())
    return [p.strip().rstrip(".") for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Step 2+3 — entail each claim against the (joined) context, then aggregate.
# ---------------------------------------------------------------------------
def faithfulness(answer: str, contexts: list[str]) -> tuple[float, list[tuple[str, str]]]:
    context = " ".join(contexts)
    claims = extract_claims(answer)
    if not claims:
        return 1.0, []
    verdicts = []
    supported = 0
    for claim in claims:
        label = mock_nli(premise=context, hypothesis=claim)
        verdicts.append((claim, label))
        if label == "entailment":
            supported += 1
    return supported / len(claims), verdicts


# ---------------------------------------------------------------------------
# Demo data: one faithful answer, one with a hallucinated claim.
# ---------------------------------------------------------------------------
CONTEXT = [
    "The Eiffel Tower is located in Paris, France.",
    "It was completed in 1889 and is made of wrought iron.",
    "It stands about 330 metres tall.",
]

CASES = [
    {
        "label": "faithful answer",
        "answer": "The Eiffel Tower is in Paris and it was completed in 1889.",
    },
    {
        "label": "partly hallucinated answer",
        "answer": ("The Eiffel Tower is in Paris and it is made of solid gold "
                   "and was designed by Leonardo da Vinci."),
    },
]


def main() -> int:
    banner("Faithfulness = supported claims / total claims (NLI vs context)")
    print("Context:")
    for c in CONTEXT:
        print(f"  - {c}")

    for case in CASES:
        score, verdicts = faithfulness(case["answer"], CONTEXT)
        print()
        print(f"[{case['label']}]  faithfulness = {score:.2f}")
        print(f"  answer: {case['answer']}")
        for claim, label in verdicts:
            flag = "OK " if label == "entailment" else "!! "
            print(f"    {flag}[{label:13s}] {claim}")

    print("\nLesson: faithfulness localises hallucination to the offending claim.")
    print("Swap mock_nli for a real LLM/NLI judge and the loop is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

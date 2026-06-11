"""04 · The real RAGAS API (guarded) — what the library call looks like.

Files 01–03 reimplemented the metrics from scratch so you understand them. In
practice you call ``ragas.evaluate(dataset, metrics=[...], llm=..., embeddings=...)``
and get a scorecard. This file shows that exact call and runs it **iff** ragas is
installed AND you wire a judge LLM; otherwise it prints the snippet and skips
gracefully (exit 0, no network).

The canonical real usage:

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness, answer_relevancy,
        context_precision, context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    ds = Dataset.from_dict({
        "question":    [...],
        "answer":      [...],
        "contexts":    [[...], ...],   # list-of-lists
        "ground_truth":[...],
    })
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy,
                 context_precision, context_recall],
        llm=LangchainLLMWrapper(my_chat_model),         # any LangChain LLM
        embeddings=LangchainEmbeddingsWrapper(my_embed),
    )
    print(result)   # {'faithfulness': 0.91, 'answer_relevancy': 0.88, ...}

By default RAGAS uses OpenAI — which needs a key + network. We never call a
hosted model here; everything that runs is local + deterministic.

Docs: https://docs.ragas.io/en/stable/getstarted/

Run:  python 04.ragas_api.py     (exit 0 offline; runs ragas only if installed+wired)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import banner, note_skip, safe  # noqa: E402


SAMPLE = {
    "question": ["What is the capital of France?"],
    "answer": ["The capital of France is Paris."],
    "contexts": [["Paris is the capital and most populous city of France."]],
    "ground_truth": ["Paris is the capital of France."],
}


def try_real_ragas() -> None:
    ok, ragas = safe(__import__, "ragas")
    if not ok:
        note_skip("`ragas` not installed (pip install ragas datasets langchain)")
        print("       See the module docstring for the exact evaluate() call.")
        return

    ok2, datasets = safe(__import__, "datasets")
    if not ok2:
        note_skip("`datasets` not installed — needed to build the RAGAS Dataset")
        return

    print(f"[ok] ragas is installed (version {getattr(ragas, '__version__', '?')}).")
    print("     The real evaluate() call needs a judge LLM + embeddings wired in.")
    print("     RAGAS defaults to OpenAI (API key + network), which this offline")
    print("     tutorial deliberately does NOT invoke. To run it for real:")
    print()
    print("       from ragas import evaluate")
    print("       from ragas.metrics import faithfulness, answer_relevancy")
    print("       result = evaluate(ds, metrics=[faithfulness, answer_relevancy],")
    print("                         llm=<your LLM>, embeddings=<your embedder>)")
    print()
    print("     We skip the network call; the offline scorecard is in app.py.")


def main() -> int:
    banner("RAGAS real API (guarded)")
    print("Sample row that would be evaluated:")
    for k, v in SAMPLE.items():
        print(f"  {k:13s}: {v}")
    print()
    try_real_ragas()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

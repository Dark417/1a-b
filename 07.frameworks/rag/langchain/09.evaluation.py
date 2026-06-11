"""LangChain RAG 09 — evaluation hooks.

You cannot improve what you do not measure. RAG has two failure surfaces:
*retrieval* (did we fetch the right context?) and *generation* (did the answer
use it faithfully?). Core metrics, all computable offline here:

  * **context recall** — fraction of gold-relevant docs that were retrieved.
  * **context precision** — fraction of retrieved docs that are relevant.
  * **faithfulness** — is every answer token grounded in the context? (lexical
    proxy: answer-vs-context token overlap).
  * **answer correctness** — does the answer contain the gold fact?

These are the same axes RAGAS / DeepEval / TruLens score with an LLM judge; the
LangChain hook point is ``langchain.evaluation`` / the LangSmith evaluators.
Docs: https://python.langchain.com/docs/concepts/evaluation/
      https://docs.ragas.io/  (metric definitions)
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _rag_common import (  # noqa: E402
    LocalEmbeddings,
    MiniVectorStore,
    MockLLM,
    banner,
    format_context,
    note,
    sample_documents,
    seed_everything,
    tokenize,
)

# (question, gold_doc_ids, gold_answer_substring)
EVAL_SET = [
    ("What does the Sun fuse in its core?", {"sun"}, "hydrogen"),
    ("How many moons does Mars have?", {"mars"}, "two"),
    ("What is the largest planet in the Solar System?", {"jupiter"}, "Jupiter"),
]


def context_recall(retrieved_ids, gold_ids):
    if not gold_ids:
        return 1.0
    return len(set(retrieved_ids) & gold_ids) / len(gold_ids)


def context_precision(retrieved_ids, gold_ids):
    if not retrieved_ids:
        return 0.0
    return len(set(retrieved_ids) & gold_ids) / len(set(retrieved_ids))


def faithfulness(answer, context):
    """Proxy: fraction of (content) answer tokens present in the context."""
    a = [t for t in tokenize(answer) if len(t) > 3 and t not in {"based", "context"}]
    if not a:
        return 1.0
    c = set(tokenize(context))
    return sum(t in c for t in a) / len(a)


def answer_correct(answer, gold_substr):
    return gold_substr.lower() in answer.lower()


def main():
    seed_everything()
    banner("LangChain 09 — RAG evaluation hooks")

    try:
        import langchain.evaluation  # noqa: F401

        note("langchain.evaluation importable (live evaluator API available)")
    except Exception as e:
        note(f"langchain.evaluation unavailable ({e}); using built-in metrics")

    store = MiniVectorStore(LocalEmbeddings())
    store.add(sample_documents())
    llm = MockLLM()

    rows = []
    for q, gold_ids, gold_sub in EVAL_SET:
        hits = store.similarity(q, k=2)
        rids = [d["id"] for d, _ in hits]
        ctx = format_context([d for d, _ in hits], with_sources=False)
        ans = llm.answer(q, ctx)
        rows.append({
            "q": q,
            "recall": context_recall(rids, gold_ids),
            "precision": context_precision(rids, gold_ids),
            "faithfulness": round(faithfulness(ans, ctx), 2),
            "correct": answer_correct(ans, gold_sub),
        })

    print(f"\n{'recall':>7} {'prec':>6} {'faith':>6} {'correct':>8}  question")
    for r in rows:
        print(f"{r['recall']:>7.2f} {r['precision']:>6.2f} {r['faithfulness']:>6.2f} "
              f"{str(r['correct']):>8}  {r['q']}")

    n = len(rows)
    print("\nAggregates:")
    print(f"   mean context recall    = {sum(r['recall'] for r in rows)/n:.2f}")
    print(f"   mean context precision = {sum(r['precision'] for r in rows)/n:.2f}")
    print(f"   mean faithfulness      = {sum(r['faithfulness'] for r in rows)/n:.2f}")
    print(f"   answer accuracy        = {sum(r['correct'] for r in rows)/n:.2f}")

    print("\nOK")


if __name__ == "__main__":
    main()

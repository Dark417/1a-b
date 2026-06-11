"""LangChain RAG 06 — multi-query expansion + contextual-compression rerank.

Two recall/precision boosters:

  * **Multi-query** — a single user question often phrases things differently
    from the corpus. We ask the LLM to rewrite it into several paraphrases,
    retrieve for each, and union the results. Improves *recall*.
    (LangChain: ``MultiQueryRetriever``.)

  * **Contextual compression / reranking** — after retrieving a wide net,
    re-score each chunk against the query and keep only the most relevant
    sentences. Improves *precision* and shrinks the prompt.
    (LangChain: ``ContextualCompressionRetriever`` + a compressor/reranker.)

Docs: https://python.langchain.com/docs/how_to/MultiQueryRetriever/
      https://python.langchain.com/docs/how_to/contextual_compression/

Offline we emulate both: the MockLLM produces deterministic paraphrases and a
lexical reranker filters sentences.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _rag_common import (  # noqa: E402
    LocalEmbeddings,
    MiniVectorStore,
    banner,
    note,
    sample_documents,
    seed_everything,
    split_sentences,
    tokenize,
)

QUESTION = "Which world has the most moons-like satellites?"


def expand_queries(question):
    """Deterministic 'LLM' query expansion (paraphrase generator)."""
    base = question.rstrip("?")
    return [
        question,
        f"{base} natural satellites",
        f"moons of planets {base}",
    ]


def multi_query_retrieve(store, queries, k=2):
    seen, union = set(), []
    for q in queries:
        for d, score in store.similarity(q, k=k):
            if d["id"] not in seen:
                seen.add(d["id"])
                union.append((d, score))
    return union


def rerank_compress(question, docs, keep_sentences=1):
    """Contextual compression: keep only the top sentence(s) per doc by overlap."""
    q = set(tokenize(question))
    compressed = []
    for d in docs:
        scored = sorted(
            split_sentences(d["text"]),
            key=lambda s: len(q & set(tokenize(s))),
            reverse=True,
        )
        compressed.append({**d, "text": " ".join(scored[:keep_sentences])})
    return compressed


def maybe_live_multiquery():
    """If the real MultiQueryRetriever imports, note it; else stay on fallback."""
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever  # noqa: F401

        note("MultiQueryRetriever import OK (live class available)")
        return True
    except Exception as e:
        note(f"MultiQueryRetriever unavailable ({e}); deterministic emulation")
        return False


def main():
    seed_everything()
    banner("LangChain 06 — multi-query + compression/rerank")
    print(f"Q: {QUESTION}\n")
    maybe_live_multiquery()

    store = MiniVectorStore(LocalEmbeddings())
    store.add(sample_documents())

    queries = expand_queries(QUESTION)
    print("Expanded queries:")
    for q in queries:
        print(f"   - {q}")

    union = multi_query_retrieve(store, queries, k=2)
    print(f"\nMulti-query union -> {len(union)} unique docs:")
    for d, _ in union:
        print(f"   - {d['id']}: {d['text'][:55]!r}")

    compressed = rerank_compress(QUESTION, [d for d, _ in union], keep_sentences=1)
    print("\nAfter contextual compression (top sentence per doc):")
    for d in compressed:
        print(f"   - {d['id']}: {d['text']!r}")

    print("\nOK")


if __name__ == "__main__":
    main()

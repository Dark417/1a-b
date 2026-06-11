"""LangChain RAG 05 — returning sources / citations.

A production RAG answer is only trustworthy if it can show *where* each claim
came from. The pattern: keep the retrieved ``Document``s around, format them
with a citation marker ``[1]``, ``[2]``..., and return both the answer text and
the source metadata. We use ``RunnableParallel`` so one branch produces the
answer and another passes the source documents straight through.

Docs: https://python.langchain.com/docs/how_to/qa_sources/
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _lc_adapters import get_lc_pieces  # noqa: E402
from _rag_common import (  # noqa: E402
    LocalEmbeddings,
    MiniVectorStore,
    MockLLM,
    banner,
    note,
    sample_documents,
    seed_everything,
)

QUESTION = "How many moons does Mars have and what is the largest planet?"


def cited_context(docs):
    """Render docs with [n] markers and return the citation table."""
    lines, table = [], []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source") if hasattr(d, "metadata") else d["source"]
        text = d.page_content if hasattr(d, "page_content") else d["text"]
        lines.append(f"[{i}] {text}")
        table.append((i, src))
    return "\n".join(lines), table


def run_live():
    _, embeddings, _ = get_lc_pieces()
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    docs = [Document(page_content=d["text"], metadata={"source": d["source"]})
            for d in sample_documents()]
    retriever = FAISS.from_documents(docs, embeddings).as_retriever(
        search_kwargs={"k": 3})
    hits = retriever.invoke(QUESTION)
    return hits


def run_fallback():
    store = MiniVectorStore(LocalEmbeddings())
    store.add(sample_documents())
    return [d for d, _ in store.similarity(QUESTION, k=3)]


def main():
    seed_everything()
    banner("LangChain 05 — citations & sources")
    print(f"Q: {QUESTION}\n")

    try:
        hits = run_live()
        note("retrieved with LIVE FAISS retriever")
    except Exception as e:  # pragma: no cover
        note(f"live retriever unavailable ({e}); harness fallback")
        hits = run_fallback()

    ctx, table = cited_context(hits)
    answer = MockLLM().answer(QUESTION, ctx)

    print("Answer:")
    print(f"  {answer}")
    print("\nSources:")
    for n, src in table:
        print(f"  [{n}] {src}")

    print("\nOK")


if __name__ == "__main__":
    main()

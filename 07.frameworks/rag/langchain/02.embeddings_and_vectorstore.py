"""LangChain RAG 02 — embeddings & vector stores.

Embeddings map text to vectors so that semantically similar text lands nearby.
A vector store indexes those vectors and answers nearest-neighbour queries. This
file builds a *real* LangChain vector store (FAISS, falling back to Chroma, then
to the harness ``MiniVectorStore``) over our tiny corpus and runs a similarity
search — all offline using a local embedding.

Docs: https://python.langchain.com/docs/concepts/vectorstores/
      https://python.langchain.com/docs/concepts/embedding_models/
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _lc_adapters import get_lc_pieces  # noqa: E402
from _rag_common import (  # noqa: E402
    LocalEmbeddings,
    MiniVectorStore,
    banner,
    note,
    sample_documents,
    seed_everything,
)


def build_store(embeddings, live):
    """Try FAISS, then Chroma, then the harness store. Return (store, kind)."""
    raw = sample_documents()
    if live:
        try:
            from langchain_core.documents import Document

            docs = [Document(page_content=d["text"], metadata={"source": d["source"]})
                    for d in raw]
            try:
                from langchain_community.vectorstores import FAISS

                store = FAISS.from_documents(docs, embeddings)
                return store, "FAISS (live)"
            except Exception as e:
                note(f"FAISS unavailable ({e}); trying Chroma")
            from langchain_community.vectorstores import Chroma

            store = Chroma.from_documents(docs, embeddings)
            return store, "Chroma (live)"
        except Exception as e:  # pragma: no cover
            note(f"langchain vector store unavailable ({e}); harness store")
    mini = MiniVectorStore(LocalEmbeddings())
    mini.add(raw)
    return mini, "MiniVectorStore (fallback)"


def search(store, kind, query, k=3):
    if kind.startswith(("FAISS", "Chroma")):
        hits = store.similarity_search_with_score(query, k=k)
        return [(h.page_content, float(score)) for h, score in hits]
    hits = store.similarity(query, k=k)
    return [(d["text"], score) for d, score in hits]


def main():
    seed_everything()
    banner("LangChain 02 — embeddings & vector store")
    llm, embeddings, live = get_lc_pieces()
    note(f"langchain live = {live}")

    store, kind = build_store(embeddings, live)
    note(f"vector store = {kind}")

    for q in ["What does the Sun fuse?", "How many moons does Mars have?"]:
        print(f"\nQ: {q}")
        for text, score in search(store, kind, q, k=2):
            print(f"   score={score:+.3f}  {text[:70]!r}")

    print("\nOK")


if __name__ == "__main__":
    main()

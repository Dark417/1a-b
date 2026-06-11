"""LangChain RAG 01 — document loaders & chunking strategies.

Loading turns raw files/URLs into ``Document`` objects (``page_content`` +
``metadata``). Splitting then chops long documents into retrieval-sized chunks.
Chunking is the single biggest lever on RAG quality: each chunk should be small
enough to fit the model's context yet large enough to stand on its own.

This file demonstrates, all offline:
  * building ``Document`` objects (the unit every LangChain loader emits)
  * ``RecursiveCharacterTextSplitter`` — recurse over separators
    (paragraph -> line -> sentence -> word) to keep chunks coherent
  * ``CharacterTextSplitter`` and token-based splitting
  * a hand-rolled "semantic-ish" sentence splitter for comparison

Docs: https://python.langchain.com/docs/concepts/text_splitters/

Runs with no network: if langchain isn't importable we fall back to a local
splitter so the lesson still executes.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _rag_common import (  # noqa: E402
    banner,
    note,
    sample_documents,
    seed_everything,
    split_sentences,
)


def to_langchain_documents(raw):
    """Mimic a loader: emit LangChain ``Document``s, or dicts as a fallback."""
    try:
        from langchain_core.documents import Document

        docs = [
            Document(page_content=d["text"], metadata={"id": d["id"],
                     "title": d["title"], "source": d["source"]})
            for d in raw
        ]
        note(f"loaded {len(docs)} langchain_core.Document objects (live)")
        return docs, True
    except Exception as e:  # pragma: no cover - offline fallback
        note(f"langchain_core unavailable ({e}); using dict documents (mock)")
        return raw, False


def recursive_split(docs, live):
    """Recursive character splitter — the recommended default."""
    if live:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=120, chunk_overlap=20,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_documents(docs)
            return [c.page_content for c in chunks], "RecursiveCharacterTextSplitter (live)"
        except Exception as e:  # pragma: no cover
            note(f"splitter unavailable ({e}); local fallback")
    # Fallback: greedy character windows with overlap.
    texts = [d.page_content if hasattr(d, "page_content") else d["text"] for d in docs]
    out, size, overlap = [], 120, 20
    for t in texts:
        i = 0
        while i < len(t):
            out.append(t[i : i + size])
            i += size - overlap
    return out, "local-char-window (fallback)"


def token_split(docs, live):
    """Token-aware splitting (falls back to word windows)."""
    if live:
        try:
            from langchain_text_splitters import TokenTextSplitter

            splitter = TokenTextSplitter(chunk_size=40, chunk_overlap=8)
            chunks = splitter.split_documents(docs)
            return [c.page_content for c in chunks], "TokenTextSplitter (live)"
        except Exception as e:  # pragma: no cover
            note(f"token splitter unavailable ({e}); word-window fallback")
    texts = [d.page_content if hasattr(d, "page_content") else d["text"] for d in docs]
    out = []
    for t in texts:
        words = t.split()
        for i in range(0, len(words), 32):
            out.append(" ".join(words[i : i + 40]))
    return out, "word-window (fallback)"


def semantic_split(docs):
    """Sentence-boundary splitting — keeps whole thoughts together."""
    out = []
    for d in docs:
        text = d.page_content if hasattr(d, "page_content") else d["text"]
        out.extend(split_sentences(text))
    return out, "sentence-boundary"


def main():
    seed_everything()
    banner("LangChain 01 — loaders & chunking strategies")

    raw = sample_documents()
    docs, live = to_langchain_documents(raw)

    for fn in (recursive_split, token_split):
        chunks, label = fn(docs, live)
        print(f"\n[{label}] -> {len(chunks)} chunks; first 3:")
        for c in chunks[:3]:
            print(f"   - {c[:70]!r}")

    chunks, label = semantic_split(docs)
    print(f"\n[{label}] -> {len(chunks)} chunks; first 3:")
    for c in chunks[:3]:
        print(f"   - {c[:70]!r}")

    print("\nOK")


if __name__ == "__main__":
    main()

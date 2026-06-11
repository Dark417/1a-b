"""LangChain RAG — end-to-end app.

Ties the whole tutorial together into one runnable pipeline:

    load -> split -> embed -> index (FAISS) -> hybrid retrieve (dense+BM25)
         -> contextual compression -> prompt -> LLM -> answer + citations

Runs a small built-in question set by default (so ``python app.py`` exits 0 with
no arguments and no network); pass a question as an argument to ask your own:

    python app.py "What is the largest planet?"

Everything degrades to the offline harness (MockLLM + local embeddings + a
pure-python store) if any LangChain/optional piece is missing.
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
    split_sentences,
    tokenize,
)

DEFAULT_QUESTIONS = [
    "What does the Sun fuse in its core?",
    "How many moons does Mars have?",
    "What is the largest planet in the Solar System?",
]


class RagApp:
    """A small, explicit RAG application with graceful degradation."""

    def __init__(self):
        self.raw = sample_documents()
        self.llm, self.embeddings, self.live = get_lc_pieces()
        self.kind = "harness"
        self._retriever = None
        self._setup()

    def _setup(self):
        if self.live:
            try:
                from langchain_community.retrievers import BM25Retriever
                from langchain_community.vectorstores import FAISS
                from langchain_core.documents import Document

                docs = [Document(page_content=d["text"],
                                 metadata={"source": d["source"], "id": d["id"]})
                        for d in self.raw]
                self._faiss = FAISS.from_documents(docs, self.embeddings).as_retriever(
                    search_kwargs={"k": 3})
                self._bm25 = BM25Retriever.from_documents(docs)
                self._bm25.k = 3
                self.kind = "FAISS+BM25 (live)"
                return
            except Exception as e:  # pragma: no cover
                note(f"live retrievers unavailable ({e}); harness store")
        self._mini = MiniVectorStore(LocalEmbeddings())
        self._mini.add(self.raw)
        self.kind = "MiniVectorStore (fallback)"

    def retrieve(self, question, k=3):
        if self.kind.startswith("FAISS"):
            dense = self._faiss.invoke(question)
            sparse = self._bm25.invoke(question)
            # reciprocal rank fusion
            scores, keep = {}, {}
            for ranking in (dense, sparse):
                for rank, d in enumerate(ranking):
                    key = d.page_content
                    scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
                    keep[key] = d
            order = sorted(scores, key=scores.get, reverse=True)[:k]
            return [{"text": keep[o].page_content,
                     "source": keep[o].metadata.get("source", "?")} for o in order]
        return [{"text": d["text"], "source": d["source"]}
                for d, _ in self._mini.similarity(question, k=k)]

    def compress(self, question, docs, keep=2):
        """Keep only the most relevant sentences per doc."""
        q = set(tokenize(question))
        out = []
        for d in docs:
            sents = sorted(split_sentences(d["text"]),
                           key=lambda s: len(q & set(tokenize(s))), reverse=True)
            out.append({**d, "text": " ".join(sents[:keep])})
        return out

    def answer(self, question):
        docs = self.retrieve(question, k=3)
        docs = self.compress(question, docs)
        ctx_lines, sources = [], []
        for i, d in enumerate(docs, 1):
            ctx_lines.append(f"[{i}] {d['text']}")
            sources.append((i, d["source"]))
        context = "\n".join(ctx_lines)
        ans = MockLLM().answer(question, context)
        return ans, sources


def main(argv):
    seed_everything()
    banner("LangChain RAG — end-to-end app")
    app = RagApp()
    note(f"langchain live = {app.live}; retrieval = {app.kind}")

    questions = [" ".join(argv[1:])] if len(argv) > 1 else DEFAULT_QUESTIONS
    for q in questions:
        ans, sources = app.answer(q)
        print(f"\nQ: {q}")
        print(f"A: {ans}")
        print("   sources: " + ", ".join(f"[{n}] {s}" for n, s in sources))

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

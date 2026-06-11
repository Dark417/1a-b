"""LangChain RAG 03 — retrievers: similarity, MMR, BM25, hybrid.

A *retriever* is anything with ``.invoke(query) -> List[Document]``. LangChain
ships several strategies:

  * **similarity** — pure dense cosine top-k (can return near-duplicates).
  * **MMR** (Maximal Marginal Relevance) — re-rank to balance relevance with
    diversity, so you cover more of the answer space.
  * **BM25** — classic *sparse* lexical retrieval; great for exact terms,
    names, codes that embeddings blur together.
  * **hybrid / ensemble** — fuse dense + sparse rankings (reciprocal rank
    fusion). Usually the most robust in practice.

Docs: https://python.langchain.com/docs/concepts/retrievers/

This builds live LangChain retrievers where available and otherwise reproduces
each strategy with the harness store so the lesson always runs.
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
    tokenize,
)

QUERY = "What is the biggest planet and does Mars have moons?"


def lc_docs():
    from langchain_core.documents import Document

    return [Document(page_content=d["text"], metadata={"source": d["source"]})
            for d in sample_documents()]


def run_live():
    docs = lc_docs()
    _, embeddings, _ = get_lc_pieces()
    from langchain_community.vectorstores import FAISS

    store = FAISS.from_documents(docs, embeddings)

    sim = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    mmr = store.as_retriever(search_type="mmr",
                             search_kwargs={"k": 3, "fetch_k": 5, "lambda_mult": 0.5})

    from langchain_community.retrievers import BM25Retriever

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 3

    results = {
        "similarity (dense)": sim.invoke(QUERY),
        "MMR (dense+diversity)": mmr.invoke(QUERY),
        "BM25 (sparse lexical)": bm25.invoke(QUERY),
    }

    # Hybrid: try EnsembleRetriever, else do reciprocal-rank fusion by hand.
    try:
        from langchain.retrievers import EnsembleRetriever

        hybrid = EnsembleRetriever(retrievers=[bm25, sim], weights=[0.5, 0.5])
        results["hybrid (ensemble/RRF)"] = hybrid.invoke(QUERY)
    except Exception as e:
        note(f"EnsembleRetriever unavailable ({e}); manual RRF")
        results["hybrid (manual RRF)"] = _rrf([bm25.invoke(QUERY), sim.invoke(QUERY)])
    return {k: [d.page_content for d in v] for k, v in results.items()}


def _rrf(rankings, k=60, top=3):
    """Reciprocal Rank Fusion over several ranked Document lists."""
    scores = {}
    keep = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            keep[key] = doc
    order = sorted(scores, key=scores.get, reverse=True)[:top]
    return [keep[o] for o in order]


def run_fallback():
    """Reproduce every strategy with the harness when langchain is absent."""
    raw = sample_documents()
    emb = LocalEmbeddings()
    store = MiniVectorStore(emb)
    store.add(raw)

    sim = [d["text"] for d, _ in store.similarity(QUERY, k=3)]
    mmr = [d["text"] for d, _ in store.mmr(QUERY, k=3, lambda_mult=0.5, fetch_k=5)]

    # BM25-lite: idf-weighted term overlap.
    bm25 = _bm25_lite(QUERY, raw, k=3)

    # Hybrid via reciprocal rank fusion of dense + bm25.
    dense_ids = [d["id"] for d, _ in store.similarity(QUERY, k=5)]
    bm25_ids = [d["id"] for d in _bm25_lite(QUERY, raw, k=5)]
    hybrid = _rrf_ids([bm25_ids, dense_ids], raw, top=3)

    return {
        "similarity (dense)": sim,
        "MMR (dense+diversity)": mmr,
        "BM25 (sparse lexical)": [d["text"] for d in bm25],
        "hybrid (manual RRF)": [d["text"] for d in hybrid],
    }


def _bm25_lite(query, raw, k=3):
    import math

    N = len(raw)
    df = {}
    doc_tokens = [tokenize(d["text"]) for d in raw]
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    q = tokenize(query)
    scored = []
    for d, toks in zip(raw, doc_tokens):
        score = 0.0
        for t in q:
            if t in toks:
                idf = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
                tf = toks.count(t)
                score += idf * tf / (tf + 1.0)
        scored.append((d, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:k]]


def _rrf_ids(rankings, raw, k=60, top=3):
    by_id = {d["id"]: d for d in raw}
    scores = {}
    for ranking in rankings:
        for rank, did in enumerate(ranking):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
    order = sorted(scores, key=scores.get, reverse=True)[:top]
    return [by_id[o] for o in order]


def main():
    seed_everything()
    banner("LangChain 03 — retrievers (similarity / MMR / BM25 / hybrid)")
    print(f"Query: {QUERY}\n")
    try:
        results = run_live()
        note("ran LIVE langchain retrievers")
    except Exception as e:  # pragma: no cover
        note(f"live retrievers unavailable ({e}); harness fallback")
        results = run_fallback()

    for name, hits in results.items():
        print(f"\n[{name}]")
        for h in hits:
            print(f"   - {h[:72]!r}")
    print("\nOK")


if __name__ == "__main__":
    main()

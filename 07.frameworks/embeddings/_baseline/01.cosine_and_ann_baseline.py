"""01 — the always-runs baseline: from-scratch embeddings, cosine, brute-force ANN.

This is the floor of the whole `embeddings/` tutorial and the anchor of the
offline contract: it depends on NOTHING but NumPy and the stdlib, so it runs on
any machine with no model download and no network.

What it teaches:
  1. Two clean-room, from-scratch text->vector encoders (Hashing & TF-IDF).
  2. The three similarity geometries: dot product, cosine, Euclidean — and why
     after L2-normalization they collapse into one ranking.
  3. Brute-force k-NN retrieval (the exact ground truth every ANN index
     approximates), ranking a query against a tiny corpus.
  4. A head-to-head: HashingEmbedder vs TF-IDF retrieval quality, and a crafted
     example where BOTH lexical methods fail because they have no semantics —
     motivating the neural encoders in the sibling folders.

Run:  python 01.cosine_and_ann_baseline.py     (exits 0, offline)

References:
  Weinberger et al., "Feature Hashing for Large Scale Multitask Learning",
    ICML 2009. https://arxiv.org/abs/0902.2206
  Salton & Buckley, "Term-weighting approaches in automatic text retrieval",
    Information Processing & Management 24(5), 1988.
"""
from __future__ import annotations

import numpy as np

# Reuse the canonical fallback that already passes offline.
from localemb import (
    CORPUS,
    HashingEmbedder,
    TfidfEmbedder,
    brute_force_knn,
    cosine_sim,
)

SEED = 1234


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def show_similarity_geometries() -> None:
    """Dot vs cosine vs Euclidean, and the L2-normalization identity."""
    section("1. Three similarity geometries on two raw vectors")
    a = np.array([3.0, 0.0, 4.0], dtype=np.float32)   # ||a|| = 5
    b = np.array([6.0, 0.0, 8.0], dtype=np.float32)   # ||b|| = 10, same direction
    dot = float(a @ b)
    cos = cosine_sim(a, b)
    euc = float(np.linalg.norm(a - b))
    print(f"a={a.tolist()}  b={b.tolist()}")
    print(f"  dot(a,b)      = {dot:.3f}   (grows with magnitude — length-biased)")
    print(f"  cosine(a,b)   = {cos:.3f}   (angle only; a and b are colinear -> 1.0)")
    print(f"  euclidean(a,b)= {euc:.3f}   (large despite same direction: |b|>|a|)")

    # The identity that makes vector search tractable:
    an = a / np.linalg.norm(a)
    bn = b / np.linalg.norm(b)
    print("\nAfter L2-normalization (a<-a/||a||):")
    print(f"  dot(an,bn)        = {float(an @ bn):.3f}")
    print(f"  ||an - bn||^2     = {float(np.sum((an - bn) ** 2)):.3f}"
          f"   == 2 - 2*dot = {2 - 2 * float(an @ bn):.3f}")
    print("  => on the unit sphere, maximizing dot == maximizing cosine ==")
    print("     minimizing Euclidean. ONE ranking. This is why we normalize.")


def retrieve(emb, corpus, query, k=3):
    mat = emb.embed(corpus)
    qv = emb.embed([query])[0]
    idx, scores = brute_force_knn(qv, mat, k=k)
    return list(zip(idx.tolist(), scores.tolist()))


def compare_lexical_quality() -> None:
    """HashingEmbedder vs TF-IDF on a real-ish retrieval query."""
    section("2. Brute-force k-NN retrieval: Hashing vs TF-IDF")
    query = "fast similarity search over vector embeddings"
    print(f"query = {query!r}\n")
    for name, emb in [
        ("HashingEmbedder(dim=256)", HashingEmbedder(dim=256)),
        ("TfidfEmbedder(1,2)", TfidfEmbedder(ngram_range=(1, 2)).fit(CORPUS)),
    ]:
        print(f"{name}:")
        for rank, (i, s) in enumerate(retrieve(emb, CORPUS, query), 1):
            print(f"  {rank}. {s:.3f}  {CORPUS[i]}")
        print()
    print("Both rank the obviously-matching doc first. TF-IDF down-weights the")
    print("common word 'search' via IDF, giving a cleaner, more peaked ranking.")


def show_semantic_failure() -> None:
    """A crafted example where lexical methods have ZERO recall: no shared words."""
    section("3. Where lexical baselines FAIL (the motivation for neural encoders)")
    corpus = [
        "An automobile is a wheeled motor vehicle used for transportation.",
        "Bananas are a rich source of dietary potassium.",
        "The feline predator stalked silently through the tall grass.",
    ]
    query = "a car"   # synonym of 'automobile', but shares NO tokens with it
    print(f"corpus = {corpus}")
    print(f"query  = {query!r}   (means doc 0, but shares no words with it)\n")
    for name, emb in [
        ("HashingEmbedder", HashingEmbedder(dim=256)),
        ("TfidfEmbedder", TfidfEmbedder(ngram_range=(1, 2)).fit(corpus)),
    ]:
        results = retrieve(emb, corpus, query, k=3)
        top_i, top_s = results[0]
        print(f"{name}: top score = {top_s:.3f} -> {corpus[top_i]!r}")
    print("\nBoth give the right doc a score of ~0.0: 'car' != 'automobile' to a")
    print("bag-of-words. A semantic bi-encoder (see ../sentence-transformers/)")
    print("maps 'car' and 'automobile' close together and retrieves doc 0.")


def main() -> None:
    np.random.seed(SEED)
    print("Baseline embeddings — pure NumPy, offline, always runs.")
    show_similarity_geometries()
    compare_lexical_quality()
    show_semantic_failure()
    print("\nOK: baseline ran offline with zero network access (exit 0).")


if __name__ == "__main__":
    main()

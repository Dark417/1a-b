"""localemb — a dependency-free local text embedder (the always-runs fallback).

This module is the backbone of the offline contract for 07.frameworks. Every
similarity / vector-index demo in `vector-databases/`, `embeddings/`, and
`data-processing/` can fall back to these embedders so that a file ALWAYS runs
with no network and no heavy model download.

We ship two clean-room, from-scratch text->vector encoders, plus the cosine and
brute-force k-NN primitives that a vector database accelerates:

  1. HashingEmbedder  — the "hashing trick" (feature hashing). Stateless: maps
     token n-grams into a fixed-width vector by hashing. No vocabulary, no
     fitting, deterministic. This is what scikit-learn's HashingVectorizer does.
     Ref: Weinberger et al., "Feature Hashing for Large Scale Multitask
     Learning", ICML 2009. https://arxiv.org/abs/0902.2206

  2. TfidfEmbedder    — classic TF-IDF over a fitted vocabulary, L2-normalized.
     Ref: Salton & Buckley, "Term-weighting approaches in automatic text
     retrieval", Information Processing & Management, 1988.

Both produce L2-normalized vectors, so inner product == cosine similarity. They
are NOT semantic (no learned meaning) — "car" and "automobile" look unrelated —
but they are perfectly good for exercising index plumbing, and TF-IDF is a real,
strong lexical baseline for retrieval.

Math (what an embedding lets us do):
  cosine(a, b) = (a . b) / (||a|| ||b||).
  After L2-normalization (a <- a/||a||) cosine(a,b) = a . b. A vector database's
  whole job is to find, for a query q, the items x maximizing q . x (or
  minimizing ||q - x||) faster than scanning all of them.

Run `python localemb.py` for a self-test demo.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np

SEED = 1234

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer. Trivial on purpose — teaching, not production."""
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[str]:
    if n <= 1:
        yield from tokens
        return
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


class HashingEmbedder:
    """Stateless feature-hashing embedder.

    For each token n-gram t we compute h = hash(t) and add a signed weight to
    bucket (h mod dim). The sign (a second hash bit) reduces the bias that
    collisions introduce — two colliding features are as likely to cancel as to
    reinforce, so the expected dot-product distortion is ~0 (Weinberger 2009).
    """

    def __init__(self, dim: int = 256, ngram_range=(1, 2)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        # 4 bytes -> bucket index, 1 bit -> sign.
        bucket = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if (digest[4] & 1) else -1.0
        return bucket, sign

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for gram in _ngrams(tokens, n):
                bucket, sign = self._hash(gram)
                vec[bucket] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.embed_one(t) for t in texts]).astype(np.float32)


class TfidfEmbedder:
    """Fitted TF-IDF embedder, L2-normalized. Must .fit(corpus) before .embed()."""

    def __init__(self, ngram_range=(1, 1), max_features: int | None = None):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def _grams(self, text: str) -> List[str]:
        tokens = tokenize(text)
        out: List[str] = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            out.extend(_ngrams(tokens, n))
        return out

    def fit(self, corpus: Sequence[str]) -> "TfidfEmbedder":
        # Document frequency for each term.
        df: Counter[str] = Counter()
        for doc in corpus:
            for term in set(self._grams(doc)):
                df[term] += 1
        terms = sorted(df.keys())
        if self.max_features is not None:
            terms = [t for t, _ in df.most_common(self.max_features)]
            terms.sort()
        self.vocab_ = {t: i for i, t in enumerate(terms)}
        n_docs = len(corpus)
        # Smoothed idf, like sklearn: ln((1+N)/(1+df)) + 1.
        idf = np.zeros(len(terms), dtype=np.float32)
        for t, i in self.vocab_.items():
            idf[i] = math.log((1 + n_docs) / (1 + df[t])) + 1.0
        self.idf_ = idf
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.idf_ is None:
            raise RuntimeError("TfidfEmbedder.embed() before .fit()")
        out = np.zeros((len(texts), len(self.vocab_)), dtype=np.float32)
        for r, text in enumerate(texts):
            counts = Counter(g for g in self._grams(text) if g in self.vocab_)
            for term, c in counts.items():
                j = self.vocab_[term]
                out[r, j] = c * self.idf_[j]
            norm = np.linalg.norm(out[r])
            if norm > 0:
                out[r] /= norm
        return out


# ---------------------------------------------------------------------------
# Similarity + brute-force k-NN primitives (what an ANN index accelerates).
# ---------------------------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def brute_force_knn(query: np.ndarray, matrix: np.ndarray, k: int = 5):
    """Exact nearest neighbours by cosine. Returns (indices, scores) top-k.

    matrix is (N, d). We assume rows are roughly normalized but renormalize the
    query for safety. This O(N*d) scan is the ground truth every ANN index
    approximates.
    """
    q = query / (np.linalg.norm(query) + 1e-12)
    sims = matrix @ q
    k = min(k, matrix.shape[0])
    # argpartition for top-k, then sort those k by score (descending).
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def try_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Attempt to load a real semantic encoder; return None if offline/missing.

    We set HF_HUB_OFFLINE so we never block on a network download — if the model
    is cached we use it, otherwise callers fall back to TfidfEmbedder.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 — broad on purpose: any failure -> fallback
        print(f"[localemb] semantic model unavailable ({type(exc).__name__}); "
              f"using TF-IDF fallback.")
        return None


CORPUS = [
    "The cat sat on the warm windowsill in the morning sun.",
    "A kitten napped beside the sunny window all afternoon.",
    "Python is a popular programming language for data science.",
    "Vector databases index embeddings for fast similarity search.",
    "FAISS, Qdrant and Chroma store and query high-dimensional vectors.",
    "The dog chased the ball across the green park.",
    "Approximate nearest neighbour search trades recall for speed.",
    "Cosine similarity measures the angle between two vectors.",
]


def demo() -> None:
    np.random.seed(SEED)
    print("=" * 70)
    print("localemb self-test: from-scratch embeddings + brute-force k-NN")
    print("=" * 70)

    query = "fast similarity search over vector embeddings"

    for name, emb in [
        ("HashingEmbedder(dim=256)", HashingEmbedder(dim=256)),
        ("TfidfEmbedder", TfidfEmbedder(ngram_range=(1, 2)).fit(CORPUS)),
    ]:
        mat = emb.embed(CORPUS)
        qv = emb.embed([query])[0]
        idx, scores = brute_force_knn(qv, mat, k=3)
        print(f"\n{name}: top-3 for query={query!r}")
        for rank, (i, s) in enumerate(zip(idx, scores), 1):
            print(f"  {rank}. score={s:.3f}  {CORPUS[i]}")

    # Demonstrate cosine identity after normalization: a.b == cosine(a,b).
    emb = TfidfEmbedder().fit(CORPUS)
    v = emb.embed(CORPUS[:2])
    dot = float(v[0] @ v[1])
    cos = cosine_sim(v[0], v[1])
    print(f"\nNormalized dot ({dot:.4f}) == cosine ({cos:.4f}): "
          f"{math.isclose(dot, cos, abs_tol=1e-5)}")
    print("\nOK: localemb ran offline with zero network access.")


if __name__ == "__main__":
    demo()

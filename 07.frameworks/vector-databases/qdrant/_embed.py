"""_embed.py — self-contained local embedder for offline, no-network demos.

Why this file exists
--------------------
Every vector-store tutorial in this folder must run with `python file.py`,
exit 0, and touch NO network. Real semantic encoders (sentence-transformers)
are *nice to have* but require a model that may not be cached. So each demo:

    1. tries a real sentence-transformer (only if already cached, offline), then
    2. falls back to a dependency-free HashingEmbedder (the "hashing trick").

Both paths return L2-normalized float32 vectors, so inner product == cosine.
This is a compact, clean-room cousin of 07.frameworks/embeddings/_baseline/
localemb.py, inlined per-folder so no cross-directory sys.path hacks are needed.

Hashing trick ref: Weinberger et al., "Feature Hashing for Large Scale
Multitask Learning", ICML 2009. https://arxiv.org/abs/0902.2206
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Sequence

import numpy as np

# Force offline so importing/using sentence-transformers never blocks on a
# network download. If the model is cached it loads; otherwise we fall back.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SEED = 1234
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Stateless signed feature-hashing embedder -> L2-normalized vectors.

    Each token / bigram t is hashed to a bucket (h mod dim) with a signed
    weight. The sign bit makes collisions cancel in expectation, so dot
    products stay ~unbiased (Weinberger 2009). No vocabulary, no fitting,
    fully deterministic — perfect for exercising vector-index plumbing.
    """

    def __init__(self, dim: int = 64, ngram_range=(1, 2)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _hash(self, token: str):
        d = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(d[:4], "little") % self.dim
        sign = 1.0 if (d[4] & 1) else -1.0
        return bucket, sign

    def _grams(self, tokens: Sequence[str]):
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if n == 1:
                yield from tokens
            else:
                for i in range(len(tokens) - n + 1):
                    yield " ".join(tokens[i : i + n])

    def embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for g in self._grams(tokenize(text)):
            b, s = self._hash(g)
            v[b] += s
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.embed_one(t) for t in texts]).astype(np.float32)


def brute_force_knn(query: np.ndarray, matrix: np.ndarray, k: int = 5):
    """Exact top-k by cosine (rows ~normalized). Returns (indices, scores).

    This O(N*d) scan is the ground truth every ANN index approximates.
    """
    q = query / (np.linalg.norm(query) + 1e-12)
    sims = matrix @ q
    k = min(k, matrix.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def get_embedder(dim: int = 64, prefer_semantic: bool = True):
    """Return (embed_fn, dim, backend_name).

    embed_fn(list[str]) -> np.ndarray (N, dim), L2-normalized float32.
    Tries a cached sentence-transformer; otherwise HashingEmbedder(dim).
    """
    if prefer_semantic:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer("all-MiniLM-L6-v2")
            real_dim = int(model.get_sentence_embedding_dimension())

            def embed(texts):
                v = model.encode(
                    list(texts), normalize_embeddings=True, show_progress_bar=False
                )
                return np.asarray(v, dtype=np.float32)

            return embed, real_dim, "sentence-transformers/all-MiniLM-L6-v2"
        except Exception as exc:  # noqa: BLE001 — any failure -> fallback
            print(
                f"[skip] semantic model unavailable ({type(exc).__name__}); "
                f"using local HashingEmbedder fallback."
            )

    he = HashingEmbedder(dim=dim)
    return (lambda texts: he.embed(texts)), dim, f"HashingEmbedder(dim={dim})"


# A tiny shared corpus used across the demos.
CORPUS = [
    "The cat sat on the warm windowsill in the morning sun.",
    "A kitten napped beside the sunny window all afternoon.",
    "Python is a popular programming language for data science.",
    "Vector databases index embeddings for fast similarity search.",
    "FAISS, Qdrant and Chroma store and query high-dimensional vectors.",
    "The dog chased the ball across the green park.",
    "Approximate nearest neighbour search trades recall for speed.",
    "Cosine similarity measures the angle between two vectors.",
    "HNSW builds a navigable small-world graph for logarithmic search.",
    "Product quantization compresses vectors into compact codes.",
    "Inverted file indexes partition space into Voronoi cells.",
    "Retrieval augmented generation grounds language models in documents.",
]

if __name__ == "__main__":
    np.random.seed(SEED)
    embed, dim, name = get_embedder()
    print(f"backend={name} dim={dim}")
    mat = embed(CORPUS)
    q = embed(["fast nearest neighbour search over embeddings"])[0]
    idx, sc = brute_force_knn(q, mat, k=3)
    for r, (i, s) in enumerate(zip(idx, sc), 1):
        print(f"  {r}. {s:.3f}  {CORPUS[i]}")
    print("OK: _embed ran offline.")

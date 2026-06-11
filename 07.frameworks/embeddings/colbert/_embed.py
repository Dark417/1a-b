"""_embed — compact, dependency-free fallback embedder for this folder.

Every script in this tutorial first tries to load a *real* semantic model
(cached, offline). If that fails, it falls back to the tiny lexical embedder
below so the file ALWAYS runs offline and exits 0.

This is a deliberately small (~40 line) clone of the canonical
`_baseline/localemb.py` HashingEmbedder so each folder is self-contained.

  - HashingEmbedder: the "hashing trick" (Weinberger et al., ICML 2009,
    https://arxiv.org/abs/0902.2206). Stateless, deterministic, L2-normalized,
    so inner product == cosine similarity. NOT semantic — purely lexical.
  - brute_force_knn: exact cosine nearest-neighbour scan (the ground truth an
    ANN index approximates).
  - try_sentence_transformer: load a cached SentenceTransformer offline or None.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Stateless feature-hashing embedder (unigrams + bigrams), L2-normalized."""

    def __init__(self, dim: int = 256, ngram_range=(1, 2)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _hash(self, tok: str):
        d = hashlib.md5(tok.encode()).digest()
        return int.from_bytes(d[:4], "little") % self.dim, (1.0 if d[4] & 1 else -1.0)

    def embed_one(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            grams = toks if n == 1 else [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
            for g in grams:
                b, s = self._hash(g)
                vec[b] += s
        nrm = np.linalg.norm(vec)
        return vec / nrm if nrm > 0 else vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.embed_one(t) for t in texts]).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def brute_force_knn(query: np.ndarray, matrix: np.ndarray, k: int = 5):
    """Exact top-k by cosine. Returns (indices, scores)."""
    q = query / (np.linalg.norm(query) + 1e-12)
    sims = matrix @ q
    k = min(k, matrix.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def try_sentence_transformer(model_name: str = "all-MiniLM-L6-v2", **kwargs):
    """Load a cached SentenceTransformer offline; return None on any failure."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer(model_name, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] semantic model '{model_name}' unavailable "
              f"({type(exc).__name__}); using local hashing fallback.")
        return None

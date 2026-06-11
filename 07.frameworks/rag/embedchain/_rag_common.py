"""Shared offline harness for the RAG framework tutorials.

Every ``NN.<feature>.py`` and ``app.py`` in this tree imports from this module so
that the examples run **with no API key and even if every network call / model
download fails**. The contract is simple:

    python <file>.py   # ->  exit 0, always

To honour that we provide three building blocks that never touch the network:

1. ``MockLLM`` -- a deterministic, templated "language model". Given a question
   and some retrieved context it returns an extractive answer (the sentence in
   the context most lexically similar to the question) wrapped in a fixed
   template. No randomness, no API, byte-for-byte reproducible. This is what a
   real LLM call degrades to whenever an API key or local model is missing.

2. ``LocalEmbeddings`` -- a lightweight text->vector encoder. It will use
   sentence-transformers ``all-MiniLM-L6-v2`` *if and only if* the package
   imports and the weights are already cached locally; otherwise it falls back
   to a deterministic hashed bag-of-character-ngrams embedding. Either way you
   get L2-normalised float32 vectors and cosine similarity that "works".

3. ``TINY_CORPUS`` / ``sample_documents()`` -- a seeded, tiny knowledge base
   (a handful of facts about the solar system) so retrieval is fast and the
   "right" answer is checkable by eye.

Design notes
------------
* **Determinism.** ``seed_everything`` seeds ``random``/``numpy``. The hashed
  embedding is a pure function of the input string (md5 over char n-grams), so
  two runs produce identical vectors.
* **Graceful import.** Nothing here imports a heavy/optional dependency at
  module import time except inside a guarded function. Importing this module
  cannot fail for lack of torch / sentence-transformers.
* **Self-contained.** A *copy* of this file lives in every framework folder so
  each tutorial directory is independently runnable; there are no cross-folder
  imports. Keep the copies in sync if you edit one.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------

SEED = 1234


def seed_everything(seed: int = SEED) -> None:
    """Seed every RNG we might touch. Safe to call repeatedly."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:  # numpy is a hard dep of sklearn/most frameworks, but guard anyway
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(1)  # avoid CPU thread thrash in demos
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Tiny seeded corpus  (facts a reader can verify by eye)
# ----------------------------------------------------------------------------

TINY_CORPUS: List[Dict[str, str]] = [
    {
        "id": "sun",
        "title": "The Sun",
        "source": "astro/sun.txt",
        "text": (
            "The Sun is the star at the center of the Solar System. "
            "It is a nearly perfect ball of hot plasma. "
            "The Sun's core fuses hydrogen into helium, releasing energy."
        ),
    },
    {
        "id": "earth",
        "title": "Earth",
        "source": "astro/earth.txt",
        "text": (
            "Earth is the third planet from the Sun and the only known world "
            "with life. About 71 percent of Earth's surface is covered by water. "
            "Earth has one natural satellite, the Moon."
        ),
    },
    {
        "id": "mars",
        "title": "Mars",
        "source": "astro/mars.txt",
        "text": (
            "Mars is the fourth planet from the Sun, often called the Red Planet "
            "because iron oxide on its surface gives it a reddish appearance. "
            "Mars has two small moons named Phobos and Deimos."
        ),
    },
    {
        "id": "jupiter",
        "title": "Jupiter",
        "source": "astro/jupiter.txt",
        "text": (
            "Jupiter is the fifth planet from the Sun and the largest in the "
            "Solar System. It is a gas giant with a mass more than twice that of "
            "all the other planets combined. Jupiter has a Great Red Spot, a "
            "giant storm larger than Earth."
        ),
    },
    {
        "id": "moon",
        "title": "The Moon",
        "source": "astro/moon.txt",
        "text": (
            "The Moon is Earth's only natural satellite. It is the fifth largest "
            "moon in the Solar System. The Moon causes ocean tides on Earth and "
            "is slowly drifting away from Earth."
        ),
    },
]


def sample_documents() -> List[Dict[str, str]]:
    """Return a fresh copy of the tiny corpus (so callers can mutate safely)."""
    return [dict(d) for d in TINY_CORPUS]


SAMPLE_QUESTIONS: List[str] = [
    "What does the Sun fuse in its core?",
    "How many moons does Mars have?",
    "What is the largest planet in the Solar System?",
    "What covers most of Earth's surface?",
]


# ----------------------------------------------------------------------------
# Tokenisation + lexical similarity (used by the mock + hashed embedding)
# ----------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ----------------------------------------------------------------------------
# MockLLM -- deterministic, templated, network-free
# ----------------------------------------------------------------------------


@dataclass
class MockLLM:
    """A deterministic stand-in for a chat LLM.

    ``generate(prompt)`` returns a templated string. ``answer(question, context)``
    performs naive extractive QA: it picks the context sentence with the highest
    lexical (token-Jaccard) overlap with the question and frames it as an answer.
    This is intentionally dumb but *useful*: given good retrieval it returns the
    correct fact, which lets the RAG plumbing be exercised end to end offline.
    """

    name: str = "MockLLM"
    max_context_chars: int = 2000

    def answer(self, question: str, context: str) -> str:
        q_tokens = tokenize(question)
        # Drop bare prompt labels like "Context:" / "Answer:" before scoring.
        context = re.sub(r"(?im)^\s*(context|answer|question|q)\s*:\s*$", "", context)
        sentences = split_sentences(context)
        best, best_score = "", -1.0
        for s in sentences:
            s = re.sub(r"(?i)^\s*(context|answer)\s*:\s*", "", s).strip()
            if not s:
                continue
            score = _jaccard(q_tokens, tokenize(s))
            if score > best_score:
                best, best_score = s, score
        if not best:
            return f"[{self.name}] I don't have enough context to answer."
        return f"[{self.name}] Based on the context: {best}"

    def generate(self, prompt: str) -> str:
        """Generic completion: split the prompt into (context, question).

        Many frameworks hand the model one flat prompt. We recover the trailing
        question heuristically (last line containing a '?' or the last line) and
        treat *everything before that line* as the context, so the model never
        "answers" with the question itself.
        """
        prompt = prompt[-self.max_context_chars * 4 :]
        lines = [l for l in prompt.splitlines() if l.strip()]
        q_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if "?" in lines[i]:
                q_idx = i
                break
        if q_idx is None:
            q_idx = len(lines) - 1 if lines else 0
        question = lines[q_idx] if lines else ""
        # Strip a leading "Question:" / "Q:" label from the recovered question.
        question = re.sub(r"^\s*(question|q)\s*:\s*", "", question, flags=re.I)
        # Context = all lines except the question line and any "Answer:" tail.
        ctx_lines = [l for i, l in enumerate(lines)
                     if i != q_idx and not re.match(r"^\s*answer\s*:\s*$", l, re.I)]
        context = "\n".join(ctx_lines) if ctx_lines else prompt
        return self.answer(question, context)

    # Convenience so the mock can masquerade as a callable (LangChain-style).
    def __call__(self, prompt: str) -> str:  # pragma: no cover - thin wrapper
        return self.generate(prompt)

    def invoke(self, prompt) -> str:  # pragma: no cover - thin wrapper
        return self.generate(str(prompt))

    def stream(self, prompt) -> Iterable[str]:
        """Yield the answer token-by-token for streaming demos."""
        text = self.generate(str(prompt))
        for tok in text.split(" "):
            yield tok + " "


# ----------------------------------------------------------------------------
# LocalEmbeddings -- sentence-transformers if cached, else hashed fallback
# ----------------------------------------------------------------------------


@dataclass
class LocalEmbeddings:
    """Deterministic local text embedder.

    Tries sentence-transformers ``all-MiniLM-L6-v2`` (only if it imports and the
    weights load without a download). On any failure it uses a hashed
    character-n-gram embedding -- a poor man's encoder that is nonetheless
    deterministic and gives sensible cosine neighbours for our tiny corpus.
    """

    dim: int = 256
    model_name: str = "all-MiniLM-L6-v2"
    _st_model: Optional[object] = field(default=None, repr=False)
    backend: str = "hash"

    def __post_init__(self) -> None:
        if os.environ.get("RAG_FORCE_HASH_EMB"):
            return
        try:  # only succeeds if the package + cached weights are present
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name)
            self.dim = self._st_model.get_sentence_embedding_dimension()
            self.backend = "sentence-transformers"
        except Exception:
            self._st_model = None
            self.backend = "hash"

    # -- public API -----------------------------------------------------------
    def embed(self, text: str) -> List[float]:
        if self._st_model is not None:
            try:
                vec = self._st_model.encode([text], normalize_embeddings=True)[0]
                return [float(x) for x in vec]
            except Exception:
                self._st_model = None
                self.backend = "hash"
        return self._hash_embed(text)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if self._st_model is not None:
            try:
                vecs = self._st_model.encode(list(texts), normalize_embeddings=True)
                return [[float(x) for x in v] for v in vecs]
            except Exception:
                self._st_model = None
                self.backend = "hash"
        return [self._hash_embed(t) for t in texts]

    # -- hashed fallback ------------------------------------------------------
    def _hash_embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        # unigrams + char trigrams give some sub-word signal
        grams: List[str] = list(toks)
        for tok in toks:
            padded = f"#{tok}#"
            for i in range(len(padded) - 2):
                grams.append(padded[i : i + 3])
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# ----------------------------------------------------------------------------
# A self-contained vector index (used when a real store is unavailable)
# ----------------------------------------------------------------------------


@dataclass
class MiniVectorStore:
    """A tiny in-memory cosine vector store with optional MMR re-ranking."""

    embedder: LocalEmbeddings
    _vecs: List[List[float]] = field(default_factory=list)
    _docs: List[Dict[str, str]] = field(default_factory=list)

    def add(self, docs: Sequence[Dict[str, str]]) -> None:
        texts = [d["text"] for d in docs]
        self._vecs.extend(self.embedder.embed_batch(texts))
        self._docs.extend(dict(d) for d in docs)

    def similarity(self, query: str, k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        q = self.embedder.embed(query)
        scored = [(d, cosine(q, v)) for d, v in zip(self._docs, self._vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def mmr(self, query: str, k: int = 3, lambda_mult: float = 0.5,
            fetch_k: int = 10) -> List[Tuple[Dict[str, str], float]]:
        """Maximal Marginal Relevance: balance relevance vs diversity."""
        q = self.embedder.embed(query)
        cand = [(i, cosine(q, v)) for i, v in enumerate(self._vecs)]
        cand.sort(key=lambda x: x[1], reverse=True)
        cand = cand[:fetch_k]
        selected: List[int] = []
        out: List[Tuple[Dict[str, str], float]] = []
        while cand and len(selected) < k:
            best_idx, best_val = None, -1e9
            for i, rel in cand:
                div = max((cosine(self._vecs[i], self._vecs[j]) for j in selected),
                          default=0.0)
                val = lambda_mult * rel - (1 - lambda_mult) * div
                if val > best_val:
                    best_idx, best_val, best_rel = i, val, rel
            selected.append(best_idx)
            out.append((self._docs[best_idx], best_rel))
            cand = [(i, r) for i, r in cand if i != best_idx]
        return out


# ----------------------------------------------------------------------------
# Prompt assembly + a banner helper for consistent output
# ----------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question using ONLY the context.\n"
    "If the context is insufficient, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


def format_context(docs: Iterable[Dict[str, str]], with_sources: bool = True) -> str:
    chunks = []
    for i, d in enumerate(docs, 1):
        tag = f"[{i}] ({d.get('source', d.get('id', '?'))}) " if with_sources else ""
        chunks.append(tag + d["text"])
    return "\n".join(chunks)


def build_prompt(question: str, docs: Iterable[Dict[str, str]]) -> str:
    return RAG_PROMPT_TEMPLATE.format(
        context=format_context(docs), question=question
    )


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def note(msg: str) -> None:
    """Print a degradation/info note in a consistent, greppable format."""
    print(f"  [note] {msg}")


if __name__ == "__main__":
    # Smoke test the harness itself.
    seed_everything()
    banner("offline harness smoke test")
    emb = LocalEmbeddings()
    note(f"embedding backend = {emb.backend} (dim={emb.dim})")
    store = MiniVectorStore(emb)
    store.add(sample_documents())
    llm = MockLLM()
    for q in SAMPLE_QUESTIONS[:2]:
        hits = store.similarity(q, k=2)
        ctx = format_context(d for d, _ in hits)
        print(f"\nQ: {q}")
        print(llm.answer(q, ctx))
    print("\nOK")

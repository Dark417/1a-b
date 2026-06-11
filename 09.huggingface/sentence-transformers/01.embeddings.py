"""
01.embeddings.py — SentenceTransformer encode + cosine similarity

Official docs:
  https://sbert.net/docs/sentence_transformer/usage/usage.html
  https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
"""

import hashlib
import random
import sys

import numpy as np
import torch

# ── reproducibility ──────────────────────────────────────────────────────────
torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from _lib import banner, note_skip, safe

# ─────────────────────────────────────────────────────────────────────────────
# Local fallback: deterministic bag-of-words + hash embedder
# ─────────────────────────────────────────────────────────────────────────────

def _bow_embed(texts, dim=384, seed=42):
    """
    Deterministic local fallback embedder.

    For each text:
    1. Tokenise by whitespace (lowercase).
    2. For each token, hash to a bucket in [0, dim) and increment that bucket.
    3. L2-normalise the resulting vector.

    This is NOT a real semantic model — it is purely for demonstrating
    the API shape when the real model is unavailable.
    """
    rng = np.random.default_rng(seed)
    vecs = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        tokens = text.lower().split()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            vec[h] += 1.0
        # Add a tiny reproducible noise so identical texts that differ only
        # in unique words still have distinct vectors.
        noise_seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        noise_rng  = np.random.default_rng(noise_seed)
        vec += noise_rng.normal(0, 0.01, dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vecs.append(vec)
    return np.stack(vecs)   # (N, dim)


class _FallbackEncoder:
    """Mimics the SentenceTransformer.encode() interface."""

    def __init__(self, dim=384):
        self.dim = dim

    def encode(self, sentences, normalize_embeddings=True, **kwargs):
        embs = _bow_embed(sentences, dim=self.dim)
        if not normalize_embeddings:
            # un-normalise a bit so callers that skip norm still get valid vecs
            pass
        return embs

    def get_sentence_embedding_dimension(self):
        return self.dim


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Load model (with graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 1 — Load sentence-transformers/all-MiniLM-L6-v2")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

def _load_model():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL_ID)
    return m

ok, model_or_exc = safe(_load_model)

if ok:
    model = model_or_exc
    using_fallback = False
    print(f"Loaded LIVE model: {MODEL_ID}")
    print(f"  Embedding dim  : {model.get_sentence_embedding_dimension()}")
else:
    note_skip(
        f"needs network/model — demonstrating API shape with local fallback\n"
        f"  ({model_or_exc})"
    )
    model = _FallbackEncoder(dim=384)
    using_fallback = True
    print("[fallback] Using deterministic BoW embedder (API shape preserved).")
    print(f"  Embedding dim  : {model.get_sentence_embedding_dimension()}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Encode sentences
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 2 — Encode sentences")

sentences = [
    "The cat sat on the mat.",
    "A dog rested on the rug.",
    "Machine learning models learn from data.",
    "Neural networks are a type of machine learning.",
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
]

print(f"Encoding {len(sentences)} sentences ...")
embeddings = model.encode(sentences, normalize_embeddings=True)

print(f"\nEmbedding matrix:")
print(f"  type  : {type(embeddings)}")
print(f"  shape : {embeddings.shape}   (num_sentences x embedding_dim)")
print(f"  dtype : {embeddings.dtype}")
print(f"  norm of first vector: {np.linalg.norm(embeddings[0]):.6f}  (should be ~1.0)")

# Show a slice of each embedding
for i, s in enumerate(sentences):
    snippet = embeddings[i, :6].tolist()
    print(f"  [{i}] \"{s[:40]:<40}\" first 6 dims: {[round(v,4) for v in snippet]}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Cosine similarity matrix
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 3 — Cosine similarity matrix")

print(
    "\nMath: cos(A, B) = (A · B) / (‖A‖ · ‖B‖)"
    "\nSince embeddings are already L2-normalised, this equals A · Bᵀ.\n"
)

# Manual cosine (since embeddings are unit-normed)
cos_matrix = embeddings @ embeddings.T   # (N, N)

print("Cosine similarity matrix (rounded to 2dp):")
header = "        " + "  ".join(f"[{i}]" for i in range(len(sentences)))
print(header)
for i in range(len(sentences)):
    row = "  ".join(f"{cos_matrix[i, j]:5.2f}" for j in range(len(sentences)))
    print(f"  [{i}]  {row}")

# If sentence-transformers is available, also use util.cos_sim for verification
if not using_fallback:
    from sentence_transformers import util as st_util
    official_matrix = st_util.cos_sim(embeddings, embeddings).numpy()
    max_diff = np.abs(cos_matrix - official_matrix).max()
    print(f"\nutil.cos_sim cross-check: max|diff| = {max_diff:.2e}  (should be ~0)")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Find most similar pair
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 4 — Most similar sentence pair")

# Mask diagonal (self-similarity = 1.0)
masked = cos_matrix.copy()
np.fill_diagonal(masked, -1.0)

best_i, best_j = np.unravel_index(masked.argmax(), masked.shape)
best_score = masked[best_i, best_j]

print(f"\nMost similar pair (cosine = {best_score:.4f}):")
print(f"  [{best_i}] \"{sentences[best_i]}\"")
print(f"  [{best_j}] \"{sentences[best_j]}\"")

# Show top-3 pairs
print("\nTop-3 most similar pairs:")
pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        pairs.append((cos_matrix[i, j], i, j))
pairs.sort(reverse=True)

for rank, (score, i, j) in enumerate(pairs[:3], 1):
    print(f"  #{rank}  score={score:.4f}")
    print(f"       A: \"{sentences[i]}\"")
    print(f"       B: \"{sentences[j]}\"")

if using_fallback:
    print(
        "\nNote: fallback BoW embedder — pair rankings are based on word overlap,"
        "\nnot true semantic similarity. Run with the real model for meaningful results."
    )

print("\n[done] 01.embeddings.py — exit 0")

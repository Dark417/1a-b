"""01 — Encode sentences and compute a cosine-similarity matrix.

The "hello world" of a bi-encoder: turn a list of sentences into dense vectors,
then compare them pairwise. We load the real all-MiniLM-L6-v2 model if it is
cached (it is, in this environment) and run it LIVE & offline; otherwise we fall
back to the local hashing embedder so the file ALWAYS exits 0.

Key ideas demonstrated:
  - SentenceTransformer.encode(..., normalize_embeddings=True): L2-normalize so
    inner product == cosine similarity (see ../_baseline/README.md section 2).
  - util.cos_sim: the pairwise cosine matrix.
  - A semantic win the lexical baseline can't get: "a car" ~ "an automobile".

Run:  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python 01.encode_and_similarity.py
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from _embed import HashingEmbedder, try_sentence_transformer

SEED = 1234

SENTENCES = [
    "A man is eating a piece of bread.",
    "Someone is having a meal with some food.",
    "An automobile is a wheeled motor vehicle for transportation.",
    "I just bought a brand new car.",
    "The weather is sunny and warm today.",
]


def cos_matrix(mat: np.ndarray) -> np.ndarray:
    """Pairwise cosine for L2-normalized rows == mat @ mat.T."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.clip(norms, 1e-12, None)
    return unit @ unit.T


def main() -> None:
    np.random.seed(SEED)
    model = try_sentence_transformer("all-MiniLM-L6-v2")

    if model is not None:
        mode = "LIVE all-MiniLM-L6-v2"
        # normalize_embeddings=True => rows are unit vectors => dot == cosine.
        emb = model.encode(SENTENCES, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
    else:
        mode = "FALLBACK hashing embedder (non-semantic)"
        emb = HashingEmbedder(dim=256).embed(SENTENCES)

    print("=" * 72)
    print(f"Encode + cosine similarity   [{mode}]")
    print("=" * 72)
    print(f"embedding matrix shape: {emb.shape}  (n_sentences x dim)")
    print(f"row norms (should be ~1.0): {np.round(np.linalg.norm(emb, axis=1), 4)}")

    sim = cos_matrix(emb)
    print("\nPairwise cosine similarity matrix:")
    header = "      " + " ".join(f"s{j}" + "    " for j in range(len(SENTENCES)))
    print(header)
    for i, row in enumerate(sim):
        print(f"  s{i}  " + " ".join(f"{v:5.2f}" for v in row))

    print("\nSentences:")
    for i, s in enumerate(SENTENCES):
        print(f"  s{i}: {s}")

    # Highlight the semantically related pairs.
    pairs = [(0, 1, "eating bread ~ having a meal"),
             (2, 3, "automobile ~ car (synonyms, no shared content word)")]
    print("\nSemantic pairs (a bi-encoder should score these high):")
    for i, j, why in pairs:
        print(f"  cos(s{i}, s{j}) = {sim[i, j]:.3f}   <- {why}")
    if model is not None:
        print("\nNote: s2~s3 score high despite sharing no content words — that is")
        print("semantics. The lexical fallback would score them ~0 (see _baseline).")

    print("\nOK: ran offline, exit 0.")


if __name__ == "__main__":
    main()

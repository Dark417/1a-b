"""
02.semantic_search.py — Semantic search with util.semantic_search + CrossEncoder concept

Official docs:
  https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
  https://sbert.net/docs/cross_encoder/usage/usage.html
  https://sbert.net/docs/sentence_transformer/usage/semantic_search.html
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
# Shared local fallback embedder (same as 01.embeddings.py)
# ─────────────────────────────────────────────────────────────────────────────

def _bow_embed(texts, dim=384):
    vecs = []
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        tokens = text.lower().split()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            vec[h] += 1.0
        noise_seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        noise_rng  = np.random.default_rng(noise_seed)
        vec += noise_rng.normal(0, 0.01, dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vecs.append(vec)
    return np.stack(vecs)


class _FallbackEncoder:
    def encode(self, sentences, normalize_embeddings=True, **kwargs):
        return _bow_embed(sentences)


def _manual_topk(query_embs, corpus_embs, top_k=3):
    """Manual cosine topk: returns list-of-lists matching util.semantic_search format."""
    sims = query_embs @ corpus_embs.T   # (Q, C)
    results = []
    for q_idx in range(sims.shape[0]):
        row = sims[q_idx]
        top_idx = np.argsort(-row)[:top_k]
        hits = [{"corpus_id": int(idx), "score": float(row[idx])} for idx in top_idx]
        results.append(hits)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Load model (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 1 — Load sentence-transformers/all-MiniLM-L6-v2")

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID)

ok_load, model_or_exc = safe(_load_model)

if ok_load:
    model = model_or_exc
    using_fallback = False
    print(f"Loaded LIVE model: {MODEL_ID}")
else:
    note_skip(
        f"needs network/model — demonstrating API shape with local fallback\n"
        f"  ({model_or_exc})"
    )
    model = _FallbackEncoder()
    using_fallback = True
    print("[fallback] Using deterministic BoW embedder (API shape preserved).")

# Try to import util
ok_util, st_util = safe(__import__, "sentence_transformers", fromlist=["util"])
if ok_util:
    from sentence_transformers import util as st_util
    has_util = True
else:
    has_util = False

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Build corpus and encode
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 2 — Corpus and query encoding")

corpus = [
    "The quick brown fox jumps over the lazy dog.",               # 0
    "Machine learning enables computers to learn from data.",     # 1
    "Paris is the capital city of France.",                       # 2
    "Deep neural networks achieve state-of-the-art results.",     # 3
    "The Eiffel Tower is a famous landmark in Paris.",            # 4
    "Supervised learning uses labelled training examples.",       # 5
    "The Amazon rainforest is the world's largest tropical forest.", # 6
    "Gradient descent optimizes neural network weights.",         # 7
]

queries = [
    "What is the capital of France?",          # should match 2, 4
    "How do neural networks train?",           # should match 3, 7, 1
    "Tell me about machine learning methods.", # should match 1, 5, 3
]

print(f"Corpus size : {len(corpus)} sentences")
print(f"Queries     : {len(queries)}")
print("\nEncoding corpus ...")
corpus_emb = model.encode(corpus, normalize_embeddings=True)
print(f"  corpus_embeddings shape : {corpus_emb.shape}")

print("Encoding queries ...")
query_emb = model.encode(queries, normalize_embeddings=True)
print(f"  query_embeddings shape  : {query_emb.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: util.semantic_search (or manual fallback)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 3 — Semantic search (top-3 results per query)")

TOP_K = 3

if has_util and not using_fallback:
    import torch as _torch
    results = st_util.semantic_search(
        query_embeddings  = _torch.tensor(query_emb),
        corpus_embeddings = _torch.tensor(corpus_emb),
        top_k             = TOP_K,
    )
    method = "util.semantic_search"
else:
    results = _manual_topk(query_emb, corpus_emb, top_k=TOP_K)
    method = "manual cosine topk (fallback)"

print(f"\nSearch method: {method}\n")

for q_idx, (query, hits) in enumerate(zip(queries, results)):
    print(f"Query [{q_idx}]: \"{query}\"")
    for rank, hit in enumerate(hits, 1):
        cid   = hit["corpus_id"]
        score = hit["score"]
        print(f"  #{rank}  score={score:.4f}  [{cid}] \"{corpus[cid]}\"")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: util.cos_sim directly (pairwise)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 4 — util.cos_sim: query vs corpus matrix")

print(
    "\nutil.cos_sim(A, B) returns a (len_A, len_B) tensor of cosine scores.\n"
    "Since embeddings are already unit-norm, this is just a matrix product.\n"
)

if has_util and not using_fallback:
    import torch as _torch
    sim_matrix = st_util.cos_sim(
        _torch.tensor(query_emb),
        _torch.tensor(corpus_emb),
    ).numpy()
else:
    # Manual: unit-normed embeddings → cosine = dot product
    sim_matrix = query_emb @ corpus_emb.T

print(f"  Shape: {sim_matrix.shape}   (queries × corpus)")
print("\n  Query vs corpus cosine similarity (rounded to 2dp):")
header = "                " + " ".join(f"[{i}]" for i in range(len(corpus)))
print(header)
for qi, q in enumerate(queries):
    row = " ".join(f"{sim_matrix[qi, ci]:4.2f}" for ci in range(len(corpus)))
    print(f"  Q{qi}: {q[:16]:<16} {row}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: CrossEncoder — reranking concept (doc + guarded)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 5 — CrossEncoder reranking (doc + guarded live demo)")

print(
    "\nA CrossEncoder takes a (query, passage) pair jointly and outputs a scalar\n"
    "relevance score.  It is more accurate than a bi-encoder but cannot\n"
    "pre-compute embeddings — use it to RERANK the top-N bi-encoder results.\n"
)

print("Typical retrieval pipeline:")
print("  1. Bi-encoder encodes all documents (once, offline).")
print("  2. At query time: bi-encoder encodes query → ANN search → top-100.")
print("  3. CrossEncoder scores each of the 100 (query, doc) pairs.")
print("  4. Return top-10 by CrossEncoder score.\n")

print("Code:")
print("  from sentence_transformers import CrossEncoder")
print("  cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')")
print("  pairs  = [(query, doc) for doc in top_docs]")
print("  scores = cross.predict(pairs)   # numpy array of floats")
print("  ranked = sorted(zip(scores, top_docs), reverse=True)")

# Try a live CrossEncoder if available (small model, graceful skip)
def _try_cross_encoder(query, passages):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, p) for p in passages]
    scores = ce.predict(pairs)
    return scores

q0     = queries[0]
top3   = [corpus[hit["corpus_id"]] for hit in results[0]]

ok_ce, ce_result = safe(_try_cross_encoder, q0, top3)

if ok_ce:
    print(f"\nLIVE CrossEncoder rerank for: \"{q0}\"")
    ranked = sorted(zip(ce_result, top3), reverse=True)
    for i, (sc, doc) in enumerate(ranked, 1):
        print(f"  #{i}  score={sc:.4f}  \"{doc}\"")
else:
    note_skip(f"CrossEncoder needs network — skipping live demo\n  ({ce_result})")
    print("\n[doc] CrossEncoder.predict returns a float score per (query, passage) pair.")
    print("       Higher score = more relevant.")

print("\n[done] 02.semantic_search.py — exit 0")

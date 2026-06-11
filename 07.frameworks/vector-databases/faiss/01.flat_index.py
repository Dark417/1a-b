"""01.flat_index.py — exact (brute-force) search with IndexFlatL2 / IndexFlatIP.

What this teaches
-----------------
* The two foundational FAISS indexes: ``IndexFlatL2`` (squared Euclidean) and
  ``IndexFlatIP`` (inner product / dot). "Flat" means *no compression and no
  partitioning*: the index stores every vector verbatim and, on each query,
  scans all of them. It is therefore **exact** — its results ARE the ground
  truth that every approximate index (IVF, HNSW, PQ) is measured against.
* The universal FAISS lifecycle: ``index = Index(d)`` → ``index.add(xb)`` →
  ``D, I = index.search(xq, k)``. ``I`` is the (nq, k) matrix of neighbour ids,
  ``D`` the matching distances/scores.
* The two metric families and the crucial identity that lets ``IndexFlatIP``
  compute **cosine** similarity: for L2-normalized vectors, inner product
  equals cosine. We verify flat search reproduces an independent NumPy k-NN
  exactly (recall = 1.0).

Key facts
---------
* ``IndexFlatL2`` returns **squared** L2 distances (no sqrt) — smaller is closer.
* ``IndexFlatIP`` returns dot products — **larger** is closer. With unit-norm
  rows, score = cosine ∈ [-1, 1].
* Flat indexes need **no training** (``is_trained`` is True immediately) and
  support ``reconstruct(i)`` to fetch the stored vector back losslessly.

References
----------
* FAISS wiki — "Faiss indexes": https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
* FAISS wiki — "Getting started": https://github.com/facebookresearch/faiss/wiki/Getting-started
* MetricType enum (METRIC_L2, METRIC_INNER_PRODUCT) in faiss/MetricType.h.
"""
from __future__ import annotations

import numpy as np
import faiss

from _embed import get_embedder, brute_force_knn, CORPUS

SEED = 1234


def main() -> None:
    np.random.seed(SEED)
    faiss.omp_set_num_threads(1)  # deterministic single-thread timing/order

    embed, dim, name = get_embedder()
    print(f"backend = {name}   dim = {dim}")

    # xb: the "database" we index. _embed returns L2-normalized float32 rows,
    # so inner product == cosine. FAISS requires C-contiguous float32.
    xb = np.ascontiguousarray(embed(CORPUS), dtype=np.float32)
    nb = xb.shape[0]
    print(f"corpus vectors = {nb}")

    query_text = "fast nearest neighbour search over embeddings"
    xq = np.ascontiguousarray(embed([query_text]), dtype=np.float32)
    k = 4

    # ---- IndexFlatL2: exact squared-Euclidean search ----------------------
    index_l2 = faiss.IndexFlatL2(dim)
    print(f"\nIndexFlatL2: is_trained={index_l2.is_trained} (flat needs no training)")
    index_l2.add(xb)                       # store all vectors
    print(f"ntotal after add = {index_l2.ntotal}")
    Dl2, Il2 = index_l2.search(xq, k)      # D = squared L2 (smaller = closer)
    print(f"query: {query_text!r}")
    print("IndexFlatL2 top-k (squared-L2 distance):")
    for rank, (i, dist) in enumerate(zip(Il2[0], Dl2[0]), 1):
        print(f"  {rank}. d2={dist:.4f}  {CORPUS[i]}")

    # ---- IndexFlatIP: exact inner-product (== cosine for unit vectors) -----
    index_ip = faiss.IndexFlatIP(dim)
    index_ip.add(xb)
    Dip, Iip = index_ip.search(xq, k)      # D = dot product (larger = closer)
    print("\nIndexFlatIP top-k (inner product == cosine here):")
    for rank, (i, score) in enumerate(zip(Iip[0], Dip[0]), 1):
        print(f"  {rank}. cos={score:.4f}  {CORPUS[i]}")

    # On unit-norm data the two metrics give the SAME ranking, because
    # ||a-b||^2 = 2 - 2*<a,b>  ⇒  larger dot ⇔ smaller squared-L2.
    same_order = list(Il2[0]) == list(Iip[0])
    print(f"\nL2 and IP rank order identical on normalized data? {same_order}")

    # Sanity: squared-L2 reconstructed from the IP scores.
    derived_l2 = 2.0 - 2.0 * Dip[0]
    print("max |D_L2 - (2-2*cos)| =", float(np.max(np.abs(Dl2[0] - derived_l2))))

    # ---- Flat == ground truth: compare to independent NumPy k-NN ----------
    gt_idx, gt_scores = brute_force_knn(xq[0], xb, k=k)
    recall = len(set(Iip[0]) & set(gt_idx)) / k
    print(f"\nrecall@{k} of IndexFlatIP vs NumPy brute force = {recall:.3f} "
          f"(flat is exact, so this is 1.000)")
    assert recall == 1.0, "flat index must reproduce brute force exactly"

    # ---- reconstruct(): flat indexes store vectors losslessly -------------
    back = index_ip.reconstruct(0)
    print("reconstruct(0) lossless? ", bool(np.allclose(back, xb[0], atol=1e-6)))

    # ---- Cosine vs raw dot when vectors are NOT normalized ----------------
    # If you forget to normalize, IndexFlatIP ranks by raw magnitude*angle,
    # which is usually NOT what you want. normalize_L2 fixes it in place.
    raw = np.random.randn(5, dim).astype(np.float32)  # arbitrary magnitudes
    raw_q = np.random.randn(1, dim).astype(np.float32)
    ip_raw = faiss.IndexFlatIP(dim); ip_raw.add(raw)
    top_raw = ip_raw.search(raw_q, 1)[1][0, 0]
    norm = raw.copy(); faiss.normalize_L2(norm)       # in-place row normalize
    norm_q = raw_q.copy(); faiss.normalize_L2(norm_q)
    ip_norm = faiss.IndexFlatIP(dim); ip_norm.add(norm)
    top_norm = ip_norm.search(norm_q, 1)[1][0, 0]
    print(f"\nun-normalized top-1 id={top_raw}  vs  normalized (cosine) top-1 id={top_norm}")
    print("=> always normalize before IndexFlatIP if you mean COSINE.")

    print("\nOK: 01.flat_index ran live FAISS offline.")


if __name__ == "__main__":
    main()

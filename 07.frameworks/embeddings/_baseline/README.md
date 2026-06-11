# `_baseline/` — What an embedding *is*, from first principles (the always-runs floor)

This folder is the **floor of the whole `embeddings/` curriculum** and the anchor
of the **offline contract**: it depends on nothing but NumPy and the Python
standard library, so it runs on any machine, with no model download and no
network. Everything in the sibling folders (sentence-transformers, BGE,
Instructor, Nomic, ColBERT) degrades to the embedders defined here when a real
model is unavailable.

Read this first. By the end you should understand: what a text embedding is, the
three similarity geometries and why normalization collapses them, the exact math
of the two from-scratch encoders we ship, why lexical baselines matter, and
precisely *where they fail* — which is the motivation for every neural model in
the rest of the tutorial.

---

## 1. What is an embedding?

An **embedding** is a function `f: text → ℝ^d` that maps a piece of text to a
fixed-length vector of real numbers, such that **semantic (or at least lexical)
similarity becomes geometric proximity**. Once text lives in a vector space we
can do arithmetic on meaning: rank documents by closeness to a query, cluster
them, classify them, deduplicate them, or feed them to a retriever in a RAG
pipeline.

Two broad families produce these vectors:

- **Sparse / lexical** (this folder): the vector dimensions correspond to terms
  (words, n-grams, or hash buckets). A coordinate is nonzero only if its term
  appears. No learning of *meaning* — "car" and "automobile" are orthogonal.
  Examples: one-hot, bag-of-words, **TF-IDF**, the **hashing trick**, BM25.
- **Dense / semantic** (sibling folders): a neural network maps text to a dense
  vector (e.g. 384 or 768 dims) where *meaning* drives geometry, so "car" and
  "automobile" land close together. Examples: SBERT, BGE, Nomic, Instructor.

This folder builds the sparse family from scratch so the dense family has
something concrete to be compared against.

---

## 2. The three similarity geometries

Given two vectors `a, b ∈ ℝ^d`, there are three standard ways to score how
"close" they are. They are *not* interchangeable in general, but they become so
on the unit sphere.

### Dot product (inner product)
```
dot(a, b) = a · b = Σ_i a_i b_i
```
Grows with both the **angle alignment** and the **magnitudes** `||a||, ||b||`.
This length-bias is usually undesirable for retrieval (a long document should not
win just because its vector is long), but raw dot product is exactly what a
**maximum-inner-product search (MIPS)** index optimizes.

### Cosine similarity
```
cos(a, b) = (a · b) / (||a|| ||b||) ∈ [−1, 1]
```
The cosine of the angle between the vectors — magnitude is divided out, so it
measures **direction only**. This is the default for text similarity because we
care about *what a document is about*, not how long it is.

### Euclidean (L2) distance
```
euclid(a, b) = ||a − b|| = sqrt( Σ_i (a_i − b_i)^2 )
```
Straight-line distance. Sensitive to magnitude. Smaller = closer.

### The normalization identity (why we always L2-normalize)

L2-**normalize** each vector: `â = a / ||a||`, so `||â|| = 1`. Then:

```
cos(a, b) = â · b̂                              (cosine == dot on the unit sphere)
||â − b̂||² = ||â||² − 2 (â · b̂) + ||b̂||²
           = 1 − 2(â · b̂) + 1
           = 2 − 2 cos(a, b)                    (squared L2 is an affine fn of cosine)
```

So **on the unit sphere, maximizing dot product == maximizing cosine ==
minimizing Euclidean distance**: they induce *the same ranking*. This is the
single most important practical fact in vector search. Normalize once, then use
the cheapest operation (a dot product / matrix multiply) and you get cosine
ranking for free. Every embedder in this folder L2-normalizes its output, so
`matrix @ query` directly yields cosine scores.

> **Gotcha — normalize before dot-product.** If you feed *un*-normalized vectors
> to an inner-product index expecting cosine, long vectors dominate and your
> rankings are silently wrong. Normalize at index time *and* query time.

`01.cosine_and_ann_baseline.py` demonstrates all three numerically.

---

## 3. The hashing trick (feature hashing) — the math

The `HashingEmbedder` implements **feature hashing** (Weinberger et al., ICML
2009). The problem it solves: a bag-of-words vector has one dimension per
vocabulary term, and the vocabulary can be millions of terms and must be learned
and stored. Feature hashing eliminates the vocabulary entirely.

Pick an output dimension `d` (e.g. 256). Define two hash functions:
- `h : term → {0, …, d−1}` — picks the output bucket.
- `ξ : term → {−1, +1}` — picks a sign.

For a document with term multiset `T`, the embedding is:
```
φ_j(doc) = Σ_{t ∈ T : h(t) = j}  ξ(t) · count(t)
```
i.e. each term adds `±count` to one bucket. We then L2-normalize.

**Why the sign hash `ξ`?** Two distinct terms can **collide** (`h(t1) = h(t2)`).
Without signs, collisions always *add*, biasing the dot product upward. With an
independent sign bit, a collision is equally likely to add or cancel, so the
*expected* contribution of a collision to any inner product is **zero**. Formally
the hashed inner product is an **unbiased estimator** of the true (vocabulary)
inner product:
```
E[ ⟨φ(x), φ(y)⟩ ] = ⟨x, y⟩
```
with variance that shrinks as `d` grows. That is the whole theoretical guarantee
that makes the hashing trick usable.

Properties: **stateless** (no `.fit`), **deterministic**, O(1) memory in vocab
size, streaming-friendly. The cost is interpretability (you can't read a
dimension back to a word) and a little collision noise. scikit-learn's
`HashingVectorizer` is exactly this.

In our implementation `h` and `ξ` both come from one MD5 digest of the n-gram:
the first 4 bytes mod `d` give the bucket, one bit of the 5th byte gives the
sign.

---

## 4. TF-IDF — the math

The `TfidfEmbedder` implements classic **Term Frequency–Inverse Document
Frequency** weighting (Salton & Buckley, 1988), the canonical strong lexical
baseline.

For term `t` in document `d` over a corpus `D` of `N` documents:

- **Term frequency** `tf(t, d)` — how often `t` occurs in `d` (raw count here).
  Captures *"this document is about t"*.
- **Inverse document frequency** — how *rare* `t` is across the corpus:
  ```
  idf(t) = ln( (1 + N) / (1 + df(t)) ) + 1
  ```
  where `df(t)` is the number of documents containing `t`. The `+1`s are
  smoothing (avoid division by zero and `ln 0`), matching scikit-learn's
  `TfidfVectorizer(smooth_idf=True)`. Rare terms get a **high** weight; ubiquitous
  terms ("the", "is") get a weight near 1 and so contribute little.

The TF-IDF weight is the product, and the document vector (one coordinate per
vocabulary term) is then **L2-normalized**:
```
w(t, d) = tf(t, d) · idf(t)
v(d)    = [ w(t, d) ]_t  /  || [ w(t, d) ]_t ||
```

**Intuition:** IDF is an information-theoretic down-weighting. A term appearing
in every document carries ~0 bits about *which* document you want; a rare term is
highly discriminative. TF-IDF + cosine is still a respectable retrieval baseline
and the backbone of BM25 (a saturating, length-normalized refinement of the same
idea).

> **Gotcha — TF-IDF must be *fitted*.** IDF depends on the corpus, so you call
> `.fit(corpus)` once, then `.embed(...)` queries with the *same* learned vocab
> and IDF. Hashing has no such step.

---

## 5. Why lexical baselines matter (and where they fail)

**Why they matter.**
1. **They always run.** No GPU, no download, no network — they underpin the
   offline contract of this whole tutorial.
2. **They are a real, hard-to-beat baseline.** TF-IDF / BM25 win on exact-match,
   keyword-heavy, and out-of-domain queries; modern systems often **fuse**
   lexical and dense scores (hybrid search).
3. **They are interpretable and cheap.** Sparse vectors are tiny to store and
   exact to search.
4. **They are the control group.** You cannot claim a neural encoder "understands
   meaning" without showing it beats a lexical baseline on a query where meaning
   matters.

**Where they fail.** Lexical methods score on *shared tokens*. They have **no
semantics**:
- **Synonymy** — query "a car", document "an automobile is a wheeled vehicle".
  Zero shared content words ⇒ score ≈ 0, even though they mean the same thing.
- **Paraphrase / polysemy** — "bank" (river vs finance) is one dimension;
  "happy" and "joyful" are orthogonal.
- **Cross-lingual** — "dog" and "perro" never match.

`01.cosine_and_ann_baseline.py` includes a crafted synonymy example where both
Hashing and TF-IDF give the correct document a score of ≈ 0.0. That failure is
the entire motivation for the dense semantic encoders in
`../sentence-transformers/` and the other sibling folders, which map "car" and
"automobile" close together because they were *trained* to.

---

## 6. Brute-force k-NN — the ground truth ANN approximates

Retrieval is: given a query vector `q` and a matrix of `N` document vectors,
return the `k` documents maximizing `q · x`. The exact, O(N·d) way is a full
scan (`matrix @ q`, then top-k). That is `brute_force_knn`. Approximate
nearest-neighbour (ANN) indexes — HNSW, IVF, PQ — trade a little recall for large
speedups, but **brute force is the correctness oracle** you evaluate them
against. See the sibling `../../vector-databases/` section for the ANN side.

---

## Files

| File | What it does |
|------|--------------|
| `localemb.py` | The canonical fallback module: `HashingEmbedder`, `TfidfEmbedder`, `cosine_sim`, `brute_force_knn`, `try_sentence_transformer`. Run it for a self-test. |
| `01.cosine_and_ann_baseline.py` | The always-runs baseline: the three geometries + the normalization identity, Hashing-vs-TF-IDF retrieval quality, and the synonymy-failure demo. |
| `requirements.txt` | `numpy` only. |

## Install & run

```bash
pip install -r requirements.txt        # just numpy
python localemb.py                      # self-test
python 01.cosine_and_ann_baseline.py    # the baseline tour
```

Both exit 0 with no network access.

---

## References

- Weinberger, Dasgupta, Langford, Smola, Attenberg. *Feature Hashing for Large
  Scale Multitask Learning.* ICML 2009. https://arxiv.org/abs/0902.2206
- Salton, Buckley. *Term-weighting approaches in automatic text retrieval.*
  Information Processing & Management 24(5), 1988.
- Robertson, Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.*
  Foundations and Trends in IR, 2009. (the lexical SOTA that refines TF-IDF)
- scikit-learn docs: `HashingVectorizer`, `TfidfVectorizer`.
- Manning, Raghavan, Schütze. *Introduction to Information Retrieval*, 2008,
  ch. 6 (vector space model, TF-IDF).

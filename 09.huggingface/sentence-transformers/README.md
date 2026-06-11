# Sentence Transformers — University-Grade Explainer

**Official docs:** https://sbert.net  
**GitHub:** https://github.com/UKPLab/sentence-transformers  
**Paper (original):** https://arxiv.org/abs/1908.10084 (Reimers & Gurevych, 2019)

---

## The Core Idea: Sentence Embeddings

A **sentence embedding** maps a variable-length text (word, phrase, sentence,
paragraph) to a fixed-dimensional dense vector in a metric space where
*semantically similar texts are geometrically close*.

Raw transformer models (BERT, RoBERTa) produce one hidden state per token.
To get a single sentence vector, early work used the `[CLS]` token.
SentenceTransformers showed that **mean pooling** over all token embeddings
(optionally with attention-mask weighting) gives dramatically better vectors
for semantic similarity tasks.

---

## Architecture: Bi-Encoders vs Cross-Encoders

### Bi-Encoder (SentenceTransformer)

```
Sentence A ──► Transformer ──► Pool ──► Vector A (384-dim)
                                                       ↘
                                                        cosine_sim → score
                                                       ↗
Sentence B ──► Transformer ──► Pool ──► Vector B (384-dim)
```

- Both sentences are encoded **independently**.
- Embeddings can be pre-computed and cached.
- Similarity lookup is a single dot product / cosine.
- Scales to millions of candidates (FAISS, Annoy).
- Trade-off: slightly weaker accuracy than cross-encoders.

**Use case:** semantic search, de-duplication, clustering, retrieval.

### Cross-Encoder (CrossEncoder)

```
[CLS] Sentence A [SEP] Sentence B [SEP] ──► Transformer ──► Linear ──► score
```

- Both sentences are concatenated and fed jointly.
- The model attends to interactions between A and B.
- Much more accurate on pair-scoring tasks.
- **Cannot pre-compute** embeddings — O(N·Q) cost.
- Typical use: **re-ranking** a short list from a bi-encoder retrieval.

**Docs:** https://sbert.net/docs/cross_encoder/usage/usage.html

---

## Mean Pooling (How SentenceTransformer Produces a Vector)

Given:
- `token_embeddings` — shape `(batch, seq_len, hidden_dim)`
- `attention_mask`   — shape `(batch, seq_len)` — 0 for padding tokens

```python
# Expand mask to match embedding dimension
mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

# Zero out padding token embeddings, sum, divide by non-padding count
sum_emb  = torch.sum(token_embeddings * mask, dim=1)
sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
sentence_embedding = sum_emb / sum_mask
```

The result is then L2-normalised for cosine-similarity tasks:

```python
import torch.nn.functional as F
embedding = F.normalize(sentence_embedding, p=2, dim=1)
```

**Why not CLS?**  
BERT's CLS token aggregates sequence info for *next-sentence prediction*, not
semantic similarity.  Mean pooling uses information from all tokens and
transfers better to similarity benchmarks (STS, NLI, BEIR).

---

## Similarity Metrics

### Cosine Similarity

```
cos(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

Range: –1 (opposite) to +1 (identical direction).  
When embeddings are **already L2-normalised**, this reduces to a dot product:

```python
score = (A * B).sum()   # vectors already unit-norm
```

### util.cos_sim

```python
from sentence_transformers import util
matrix = util.cos_sim(embeddings_A, embeddings_B)
# shape: (len_A, len_B)
```

### Dot-Product Similarity

Some models (e.g., `multi-qa-MiniLM-L6-cos-v1`) are trained with dot-product
rather than cosine.  Check the model card before choosing a metric.

---

## Semantic Search

### util.semantic_search

Given a query embedding and a corpus embedding matrix, returns top-k results:

```python
from sentence_transformers import util

# query_embeddings  : (Q, D)
# corpus_embeddings : (C, D)
results = util.semantic_search(
    query_embeddings  = query_embeddings,
    corpus_embeddings = corpus_embeddings,
    top_k             = 5,
)
# results[q] = [{"corpus_id": i, "score": 0.93}, ...]
```

Internally uses cosine similarity; results are sorted descending.

**Docs:** https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html

### Scaling: FAISS / Annoy

For millions of documents, exact cosine search becomes slow.  
Approximate Nearest Neighbours (ANN) indices (FAISS, Annoy, hnswlib) allow
sub-linear query time at the cost of slight accuracy degradation.

---

## Training and Losses

SentenceTransformer models are fine-tuned to push similar pairs close and
dissimilar pairs apart in the embedding space.

### MultipleNegativesRankingLoss (MNR)

The most commonly used loss for retrieval fine-tuning.  
Given a batch of (anchor, positive) pairs, it treats all other positives in
the batch as *hard negatives*:

```
Loss = CrossEntropy(cos_sim(anchor, [pos_1, pos_2, ..., pos_N]) / τ)
```

- Large batches → many implicit negatives → strong training signal.
- Only requires (query, positive) pairs — no manual negatives needed.

**Docs:** https://sbert.net/docs/sentence_transformer/loss_overview.html

### Other common losses

| Loss | Data format | Use case |
|---|---|---|
| `CosineSimilarityLoss` | (a, b, score) | NLI, STS (regression) |
| `TripletLoss` | (anchor, positive, negative) | Retrieval with explicit negatives |
| `SoftmaxLoss` | (a, b, label) | Classification tasks |
| `ContrastiveLoss` | (a, b, binary_label) | Pair similarity / dissimilarity |

---

## Model Sizes and Common Models

| Model | Dimensions | Parameters | Speed | MTEB avg |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | 22M | Very fast | 56.3 |
| `all-MiniLM-L12-v2` | 384 | 33M | Fast | 59.8 |
| `all-mpnet-base-v2` | 768 | 109M | Moderate | 63.3 |
| `e5-large-v2` | 1024 | 335M | Slow | 65.7 |

**Docs / model hub:** https://www.sbert.net/docs/pretrained_models.html

---

## SentenceTransformer vs Raw Transformers CLS

| | SentenceTransformer | Raw transformers CLS |
|---|---|---|
| Pooling | Mean pooling (learned-to-work) | CLS token |
| Fine-tuning | Contrastive / ranking loss | MLM / NSP only |
| Similarity quality | High (STS Spearman ~85%) | Low (~30–50%) |
| Speed | Same inference speed | Same |
| Ease of use | `model.encode(texts)` → numpy | Manual pooling code |

---

## Quick-Start

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "The sky is blue.",
    "The weather is nice today.",
    "My cat is sleeping on the sofa.",
]
embeddings = model.encode(sentences, normalize_embeddings=True)
# shape: (3, 384)  — float32 numpy array

sim_matrix = util.cos_sim(embeddings, embeddings)
print(sim_matrix)
```

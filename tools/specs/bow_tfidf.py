from tools.nbreg import register, md, code, show, run_demo

MOD = "bow_tfidf"


@register("bow_tfidf", "04.nlp/text-representation/bow_tfidf.ipynb")
def build():
    return [
        md(r"""
# Bag-of-Words & TF-IDF — documents as vectors

> Tutorial pair for [`bow_tfidf.py`](bow_tfidf.py).

## 1. Intuition
Before we can do any math on text we need numbers. The simplest idea: throw the
words of a document into a "bag", forget their order, and count them. That is
**Bag-of-Words**. But raw counts over-reward words like *"the"* that appear
everywhere and carry no topical signal. **TF-IDF** fixes this by multiplying each
count by how *rare* the word is across the whole corpus — so distinctive words
dominate the vector. Comparing the resulting vectors with cosine similarity is
the bread-and-butter of classic search engines.
"""),
        md(r"""
## 2. Concept (the slide)
- **Vocabulary:** the sorted set of all terms; column $t$ of the matrix.
- **Term frequency** $\mathrm{tf}(t,d)$: how often term $t$ appears in document $d$.
- **Document frequency** $\mathrm{df}(t)$: in how many documents $t$ appears at all.
- **IDF** down-weights common terms: $\log\frac{N}{\mathrm{df}(t)}$ (smoothed).
- **TF-IDF** $=\mathrm{tf}\cdot\mathrm{idf}$, then **L2-normalize** each row so
  document length doesn't matter.
- **Cosine similarity** measures the angle between two document vectors — robust
  to length, unlike raw dot products.
"""),
        md(r"""
## 3. Math derivation

**Term frequency.** The base signal is the raw count
$$\mathrm{tf}(t,d)=\operatorname{count}(t,d).$$
A document that mentions "market" five times is more about markets than one that
mentions it once — but probably not *five times* more. The **sublinear** variant
dampens this:
$$\mathrm{tf}(t,d)=1+\log\operatorname{count}(t,d)\quad(\text{when count}>0).$$

**Inverse document frequency.** A term in *every* document discriminates nothing.
Let $N$ be the number of documents and $\mathrm{df}(t)$ the number containing $t$.
The unsmoothed IDF is
$$\mathrm{idf}(t)=\log\frac{N}{\mathrm{df}(t)}.$$
To avoid division by zero (a term never seen at fit time) and never-negative
weights, we use the **smoothed** form (the sklearn default), which pretends an
extra document contains every term:
$$\boxed{\;\mathrm{idf}(t)=\log\frac{1+N}{1+\mathrm{df}(t)}+1\;}$$
The trailing $+1$ keeps even ubiquitous terms ($\mathrm{df}=N$, so the log is
$\log\frac{1+N}{1+N}=0$) from being zeroed out entirely.

**TF-IDF weighting.**
$$w(t,d)=\mathrm{tf}(t,d)\,\cdot\,\mathrm{idf}(t).$$

**L2 normalization.** Long documents have larger counts everywhere, inflating all
weights. We project each row onto the unit sphere:
$$\hat w_d=\frac{w_d}{\lVert w_d\rVert_2},\qquad \lVert w_d\rVert_2=\sqrt{\textstyle\sum_t w(t,d)^2}.$$

**Cosine similarity.** The similarity of documents $a,b$ is the cosine of the
angle between their vectors:
$$\cos(a,b)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert}.$$
After L2 normalization $\lVert a\rVert=\lVert b\rVert=1$, so this collapses to the
plain dot product $\hat a\cdot\hat b$ — fast to compute for retrieval. To rank
documents for a query $q$ we vectorize $q$ the same way and sort by
$\cos(q,\,d)$.
"""),
        md("## 4. NumPy implementation"),
        show(MOD, "CountVectorizerNumPy", "TfidfVectorizerNumPy"),
        md("## 5. Reference — cosine retrieval & cross-check against sklearn"),
        show(MOD, "cosine_similarity", "retrieve"),
        md("## 6. Train / run — build vectors, retrieve, and match sklearn"),
        run_demo(MOD),
        md("## 7. Visualization — the TF-IDF document–term heatmap"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import bow_tfidf as M

docs = M.toy_corpus()
vec = M.TfidfVectorizerNumPy()
T = vec.fit_transform(docs)
terms = vec.feature_names_

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(T, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(terms))); ax.set_xticklabels(terms, rotation=90, fontsize=7)
ax.set_yticks(range(len(docs))); ax.set_yticklabels([f"doc {i}" for i in range(len(docs))])
ax.set_title("TF-IDF weights (rows=documents, columns=terms)")
fig.colorbar(im, ax=ax, label="tf-idf weight")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- TF-IDF is a sparse, interpretable, *order-free* representation — a strong,
  fast baseline for retrieval and text classification.
- It cannot capture **word meaning or order**: "dog bites man" and "man bites
  dog" get identical vectors, and *cat*/*kitten* are as unrelated as *cat*/*car*.
  Dense [embeddings](../embeddings/word2vec.ipynb) fix the meaning problem;
  sequence models fix the order problem.
- Watch the **smoothing** and **normalization** conventions — they are exactly
  why our from-scratch matrix reproduces sklearn's to machine precision.
- IDF must be fit on the **training corpus only**; reuse the same `idf_` to
  transform held-out queries/documents.
"""),
    ]

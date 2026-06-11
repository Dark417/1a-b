"""
Bag-of-Words & TF-IDF
=====================
Turn documents into fixed-length numeric vectors so we can compare and retrieve
them. **Bag-of-Words (BoW)** counts how often each vocabulary term appears,
ignoring order. **TF-IDF** reweights those counts so that words that are common
in one document but rare across the corpus (i.e. *discriminative*) get a high
weight, while ubiquitous words ("the", "is") are damped. Cosine similarity over
TF-IDF vectors is the classic information-retrieval baseline.

Variants implemented here:
    - Bag-of-Words (raw counts) and binary BoW
    - Sublinear (log) term-frequency option
    - TF-IDF with smoothed IDF (the sklearn convention) and L2 row normalization
    - Cosine-similarity retrieval (query -> ranked documents)

Training techniques demonstrated:
    - L2 normalization of feature vectors (makes cosine = dot product)
    - Smoothing of the IDF denominator to avoid division by zero

References:
    - Manning, Raghavan & Schütze, "Introduction to Information Retrieval", ch. 6
    - sklearn.feature_extraction.text.TfidfVectorizer documentation
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

SEED = 0

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphabetic tokens (the simplest sane tokenizer)."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
class CountVectorizerNumPy:
    r"""
    Bag-of-Words: build a vocabulary, then map each document to a count vector.

        X[d, t] = number of times term t occurs in document d   (raw counts)

    With ``binary=True`` we record presence/absence (X in {0, 1}) instead.
    Order is discarded — only multiplicities matter ("bag").
    """

    def __init__(self, binary: bool = False):
        self.binary = binary
        self.vocabulary_: dict[str, int] = {}

    def fit(self, docs: list[str]):
        vocab: dict[str, int] = {}
        for d in docs:
            for w in tokenize(d):
                if w not in vocab:
                    vocab[w] = len(vocab)
        # sort terms for a stable, human-readable column order
        terms = sorted(vocab)
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        self.feature_names_ = terms
        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        X = np.zeros((len(docs), len(self.vocabulary_)))
        for i, d in enumerate(docs):
            for w, c in Counter(tokenize(d)).items():
                j = self.vocabulary_.get(w)
                if j is not None:
                    X[i, j] = 1.0 if self.binary else c
        return X

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


class TfidfVectorizerNumPy:
    r"""
    TF-IDF from scratch, matching sklearn's default conventions so we can compare.

    **Term frequency** (per document d, term t):
        tf(t, d) = count(t, d)                       # raw, or
        tf(t, d) = 1 + log(count(t, d))  if sublinear and count>0

    **Inverse document frequency** (smoothed, the sklearn default):
        idf(t) = log((1 + N) / (1 + df(t))) + 1
    where N = #documents, df(t) = #documents containing t. The "+1"s are
    smoothing: they prevent zero division and stop idf from ever being negative.
    The trailing "+1" keeps terms that appear in *every* document (idf would be 0)
    from being zeroed out entirely.

    **Weighting:**  tfidf(t, d) = tf(t, d) * idf(t).

    **Normalization:** each document row is L2-normalized,
        x_d <- x_d / ||x_d||_2 ,
    so that the cosine similarity between two documents is just their dot product.
    """

    def __init__(self, sublinear_tf: bool = False, norm: str | None = "l2",
                 smooth_idf: bool = True):
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.smooth_idf = smooth_idf

    def fit(self, docs: list[str]):
        # vocabulary (sorted for stable column order)
        vocab: dict[str, int] = {}
        for d in docs:
            for w in set(tokenize(d)):
                vocab[w] = 0
        terms = sorted(vocab)
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        self.feature_names_ = terms

        N = len(docs)
        df = np.zeros(len(terms))
        for d in docs:
            for w in set(tokenize(d)):
                df[self.vocabulary_[w]] += 1.0
        # smoothed idf — the "as if one extra doc contains every term" trick
        s = 1.0 if self.smooth_idf else 0.0
        self.idf_ = np.log((N + s) / (df + s)) + 1.0
        return self

    def _tf_matrix(self, docs: list[str]) -> np.ndarray:
        X = np.zeros((len(docs), len(self.vocabulary_)))
        for i, d in enumerate(docs):
            for w, c in Counter(tokenize(d)).items():
                j = self.vocabulary_.get(w)
                if j is None:
                    continue
                X[i, j] = (1.0 + np.log(c)) if self.sublinear_tf else float(c)
        return X

    def transform(self, docs: list[str]) -> np.ndarray:
        X = self._tf_matrix(docs)
        X = X * self.idf_                      # broadcast idf across columns
        if self.norm == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        return X

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""
    Cosine similarity matrix between rows of A and rows of B:
        cos(a, b) = (a . b) / (||a|| ||b||).
    For already-L2-normalized vectors this reduces to the plain dot product.
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    an = np.linalg.norm(A, axis=1, keepdims=True)
    bn = np.linalg.norm(B, axis=1, keepdims=True)
    an[an == 0] = 1.0
    bn[bn == 0] = 1.0
    return (A / an) @ (B / bn).T


def retrieve(query: str, docs: list[str], vec: TfidfVectorizerNumPy,
             k: int = 3) -> list[tuple[int, float]]:
    """Rank documents by cosine similarity of their TF-IDF vectors to the query."""
    D = vec.transform(docs)
    q = vec.transform([query])
    sims = cosine_similarity(q, D)[0]
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# 2. Reference — sklearn's TfidfVectorizer (validate our math matches)
# ---------------------------------------------------------------------------
def sklearn_tfidf(docs: list[str]):
    """Return (matrix, feature_names) from sklearn for cross-checking."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    # token_pattern matches our tokenizer: lowercase alphabetic runs
    v = TfidfVectorizer(token_pattern=r"[a-z]+", lowercase=True)
    X = v.fit_transform(docs).toarray()
    return X, list(v.get_feature_names_out())


# ---------------------------------------------------------------------------
# 3. Demo — a tiny built-in corpus with two clear topics
# ---------------------------------------------------------------------------
def toy_corpus() -> list[str]:
    return [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are popular pets",
        "the stock market fell sharply today",
        "investors sold stocks as the market dropped",
        "a pet dog loves to play and run",
    ]


def demo():
    np.random.seed(SEED)
    docs = toy_corpus()

    bow = CountVectorizerNumPy()
    X = bow.fit_transform(docs)
    print(f"BoW matrix shape = {X.shape} (docs x vocab); vocab size = {len(bow.feature_names_)}")
    print(f"counts for doc 0 nonzero terms: "
          f"{[(bow.feature_names_[j], int(X[0, j])) for j in np.nonzero(X[0])[0]]}")

    tfidf = TfidfVectorizerNumPy()
    T = tfidf.fit_transform(docs)
    # highest-idf (most discriminative) terms
    top_idf = np.argsort(-tfidf.idf_)[:5]
    print(f"\nMost discriminative terms (highest idf): "
          f"{[tfidf.feature_names_[j] for j in top_idf]}")

    # retrieval
    query = "pets like cats and dogs"
    print(f"\nQuery: {query!r}")
    for rank, (i, s) in enumerate(retrieve(query, docs, tfidf, k=3), 1):
        print(f"  {rank}. (sim={s:.3f}) {docs[i]!r}")

    # cross-check our TF-IDF against sklearn
    try:
        Xs, names = sklearn_tfidf(docs)
        # align columns: our feature order is sorted, so is sklearn's
        same_vocab = names == tfidf.feature_names_
        close = np.allclose(T, Xs, atol=1e-6) if same_vocab else False
        print(f"\nMatches sklearn TfidfVectorizer: {close} "
              f"(max abs diff = {np.abs(T - Xs).max():.2e})" if same_vocab
              else "\nVocabularies differ; skipping numeric check.")
    except Exception as e:  # pragma: no cover - sklearn always present here
        print(f"\n(sklearn comparison skipped: {e})")


if __name__ == "__main__":
    demo()

"""
word2vec (Skip-gram & CBOW)
===========================
Learn dense word vectors by predicting context from a word (Skip-gram) or a word
from its context (CBOW). The famous result: vector arithmetic captures meaning
(king - man + woman ≈ queen). We implement both, with **negative sampling** to
make training tractable.

Variants implemented here:
    - Skip-gram and CBOW
    - Negative sampling (instead of the full softmax)
    - (Hierarchical softmax discussed conceptually)

Training techniques demonstrated:
    - NEGATIVE SAMPLING (see training-techniques/README.md)
    - Subsampling frequent words; the unigram^0.75 noise distribution

References:
    - Mikolov et al. (2013), "Efficient Estimation..." & "Distributed Representations..."
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# ---------------------------------------------------------------------------
# 1. NumPy implementation — Skip-gram with negative sampling (SGNS)
# ---------------------------------------------------------------------------
class Word2VecNumPy:
    r"""
    Two embedding tables: center vectors V (in) and context vectors U (out).
    Skip-gram + negative sampling objective (per (center c, true context o)):

        L = -log σ(u_o · v_c)  -  Σ_{k∈neg} log σ(-u_k · v_c)

    Gradients (clean, like logistic regression):
        for o:   du_o += (σ(u_o·v_c) - 1) v_c
        for k:   du_k += (σ(u_k·v_c) - 0) v_c
        dv_c    += Σ (σ(u_*·v_c) - label) u_*
    """

    def __init__(self, dim=50, window=2, neg=5, lr=0.05, mode="sg", seed=SEED):
        self.dim, self.window, self.neg = dim, window, neg
        self.lr, self.mode, self.seed = lr, mode, seed

    def build_vocab(self, sentences):
        from collections import Counter
        counts = Counter(w for s in sentences for w in s)
        self.itos = list(counts)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.array([counts[w] for w in self.itos], float)
        # unigram^0.75 noise distribution for negative sampling
        p = self.counts ** 0.75
        self.noise = p / p.sum()
        V = len(self.itos)
        rng = np.random.default_rng(self.seed)
        self.V = (rng.random((V, self.dim)) - 0.5) / self.dim   # center
        self.U = np.zeros((V, self.dim))                        # context
        return self

    def _pairs(self, sentences):
        """Yield (center_idx, context_idx) within the window."""
        for s in sentences:
            ids = [self.stoi[w] for w in s if w in self.stoi]
            for i, c in enumerate(ids):
                lo, hi = max(0, i - self.window), min(len(ids), i + self.window + 1)
                ctx = [ids[j] for j in range(lo, hi) if j != i]
                if self.mode == "sg":
                    for o in ctx:
                        yield c, o
                else:  # CBOW: average context predicts the center
                    if ctx:
                        yield ctx, c

    def fit(self, sentences, epochs=50):
        self.build_vocab(sentences)
        rng = np.random.default_rng(self.seed)
        pairs = list(self._pairs(sentences))
        self.history = []
        for _ in range(epochs):
            rng.shuffle(pairs); loss = 0.0
            for a, b in pairs:
                if self.mode == "sg":
                    c, o = a, b
                    v = self.V[c]
                else:
                    ctx, o = a, b              # CBOW
                    v = self.V[ctx].mean(0)
                negs = rng.choice(len(self.itos), self.neg, p=self.noise)
                targets = np.concatenate([[o], negs])
                labels = np.concatenate([[1.0], np.zeros(self.neg)])
                scores = _sigmoid(self.U[targets] @ v)
                err = scores - labels                          # (1+neg,)
                dU = np.outer(err, v)
                dv = err @ self.U[targets]
                self.U[targets] -= self.lr * dU
                if self.mode == "sg":
                    self.V[c] -= self.lr * dv
                else:
                    self.V[ctx] -= self.lr * dv / len(ctx)
                loss += -np.log(scores[0] + 1e-9) - np.sum(np.log(1 - scores[1:] + 1e-9))
            self.history.append(loss / len(pairs))
        return self

    def vec(self, w): return self.V[self.stoi[w]]

    def most_similar(self, w, k=5):
        q = self.vec(w); q = q / (np.linalg.norm(q) + 1e-9)
        M = self.V / (np.linalg.norm(self.V, axis=1, keepdims=True) + 1e-9)
        sim = M @ q
        order = np.argsort(-sim)
        return [(str(self.itos[i]), float(sim[i])) for i in order if self.itos[i] != w][:k]


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — SGNS with nn.Embedding
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGNSTorch(nn.Module):
    def __init__(self, vocab, dim=50):
        super().__init__()
        self.center = nn.Embedding(vocab, dim)
        self.context = nn.Embedding(vocab, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, c, pos, neg):
        v = self.center(c)                                  # (B, d)
        pos_s = (self.context(pos) * v).sum(-1)             # (B,)
        neg_s = torch.bmm(self.context(neg), v.unsqueeze(-1)).squeeze(-1)  # (B, K)
        loss = -(F.logsigmoid(pos_s) + F.logsigmoid(-neg_s).sum(1))
        return loss.mean()


def train_sgns_torch(w2v_np, sentences, dim=50, neg=5, epochs=50, lr=0.01):
    """Reuse the NumPy object's vocab/noise to train the torch version."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SGNSTorch(len(w2v_np.itos), dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    pairs = [(c, o) for c, o in w2v_np._pairs(sentences)]   # mode must be "sg"
    c = torch.tensor([p[0] for p in pairs], device=dev)
    o = torch.tensor([p[1] for p in pairs], device=dev)
    for _ in range(epochs):
        negs = torch.tensor(rng.choice(len(w2v_np.itos), (len(pairs), neg),
                                       p=w2v_np.noise), device=dev)
        opt.zero_grad(); loss = model(c, o, negs); loss.backward(); opt.step()
    return model


# ---------------------------------------------------------------------------
# 3. Demo — a tiny toy corpus with clear topical structure
# ---------------------------------------------------------------------------
def toy_corpus():
    animals = "dog cat lion tiger horse cow".split()
    fruits = "apple banana orange grape mango pear".split()
    rng = np.random.default_rng(SEED)
    sents = []
    for _ in range(400):
        grp = animals if rng.random() < 0.5 else fruits
        sents.append(list(rng.choice(grp, size=5)))
    return sents


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    sents = toy_corpus()
    w = Word2VecNumPy(dim=20, window=2, neg=5, mode="sg").fit(sents, epochs=60)
    print("Skip-gram nearest neighbours:")
    for q in ("dog", "apple"):
        print(f"  {q:6s} -> {[t for t, _ in w.most_similar(q, 3)]}")

    cb = Word2VecNumPy(dim=20, mode="cbow").fit(sents, epochs=60)
    print(f"CBOW   'lion' -> {[t for t, _ in cb.most_similar('lion', 3)]}")

    train_sgns_torch(w, sents, dim=20, epochs=40)
    print("Torch SGNS trained (shares vocab & noise dist).")
    # sanity: an animal's neighbours should be animals, not fruits
    animals = set("dog cat lion tiger horse cow".split())
    nn3 = {t for t, _ in w.most_similar("dog", 3)}
    print(f"'dog' neighbours all animals: {nn3 <= animals}")


if __name__ == "__main__":
    demo()

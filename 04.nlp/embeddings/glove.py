"""
GloVe — Global Vectors for Word Representation
==============================================
word2vec learns from *local* windows; GloVe learns directly from the *global*
co-occurrence matrix. The key insight: ratios of co-occurrence probabilities
encode meaning, and the cleanest model that reproduces them is a **weighted
least-squares regression on the log co-occurrence counts**. We build the
co-occurrence matrix from a corpus, then factorize it with SGD from scratch in
NumPy (and an optional PyTorch version), recovering the famous result that the
dot product of two word vectors approximates the log of how often they co-occur.

Variants implemented here:
    - Full GloVe objective with the f(X_ij) weighting function
    - Two vector sets (word + context) averaged at the end (the GloVe default)
    - Symmetric vs harmonic (1/distance) co-occurrence weighting
    - NumPy AdaGrad SGD from scratch + a PyTorch autograd version

Training techniques demonstrated:
    - AdaGrad per-parameter learning rates (what the GloVe paper uses)
    - The f(X) weighting that caps the influence of very frequent pairs

References:
    - Pennington, Socher & Manning (2014), "GloVe: Global Vectors for Word
      Representation"
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

SEED = 0


def _tokenize(text: str) -> list[list[str]]:
    import re
    sents = re.split(r"[.!?]+", text.lower())
    return [s.split() for s in sents if s.split()]


# ---------------------------------------------------------------------------
# 1. NumPy implementation — co-occurrence build + AdaGrad factorization
# ---------------------------------------------------------------------------
class GloVeNumPy:
    r"""
    **Step 1 — co-occurrence matrix.** Slide a window over the corpus and count,
    for every (center i, context j) pair within the window, X_ij. With harmonic
    weighting a context word d positions away contributes 1/d (closer = stronger).

    **Step 2 — the objective.** GloVe fits word vectors w_i, context vectors
    \tilde w_j and biases b_i, \tilde b_j so that
        w_i · \tilde w_j + b_i + \tilde b_j  ≈  log X_ij,
    minimizing the *weighted* least-squares loss
        J = Σ_{i,j} f(X_ij) (w_i·\tilde w_j + b_i + \tilde b_j - log X_ij)^2,
    where the weighting function tames very frequent pairs:
        f(x) = (x / x_max)^alpha  if x < x_max  else  1.

    **Why log co-occurrence?** Ratios P_ik/P_jk distinguish meaning. A model whose
    parameters are linear in vectors and reproduce those ratios forces the dot
    product to equal log P_ik (+ constants) — hence the log target above.
    """

    def __init__(self, dim=50, window=5, x_max=100, alpha=0.75,
                 lr=0.05, harmonic=True, seed=SEED):
        self.dim, self.window = dim, window
        self.x_max, self.alpha = x_max, alpha
        self.lr, self.harmonic, self.seed = lr, harmonic, seed

    def build_vocab(self, sentences):
        counts = Counter(w for s in sentences for w in s)
        self.itos = list(counts)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    def build_cooccurrence(self, sentences):
        """Symmetric co-occurrence counts X_ij (optionally 1/distance weighted)."""
        X = defaultdict(float)
        for s in sentences:
            ids = [self.stoi[w] for w in s if w in self.stoi]
            for i, ci in enumerate(ids):
                lo = max(0, i - self.window)
                for j in range(lo, i):              # left context only -> add both ways
                    d = i - j
                    inc = (1.0 / d) if self.harmonic else 1.0
                    X[(ci, ids[j])] += inc
                    X[(ids[j], ci)] += inc          # keep it symmetric
        # store as parallel arrays for fast SGD
        self.coo_i = np.array([k[0] for k in X], dtype=np.int64)
        self.coo_j = np.array([k[1] for k in X], dtype=np.int64)
        self.coo_x = np.array(list(X.values()), dtype=np.float64)
        return self

    def _f(self, x):
        # weighting function f(X_ij): cap influence of frequent pairs
        return np.where(x < self.x_max, (x / self.x_max) ** self.alpha, 1.0)

    def fit(self, sentences, epochs=50):
        self.build_vocab(sentences).build_cooccurrence(sentences)
        rng = np.random.default_rng(self.seed)
        V = len(self.itos)
        s = 0.5 / self.dim
        self.W = (rng.random((V, self.dim)) - 0.5) * s   # word vectors
        self.Wt = (rng.random((V, self.dim)) - 0.5) * s  # context vectors
        self.b = np.zeros(V)
        self.bt = np.zeros(V)
        # AdaGrad accumulators (sum of squared grads) — start at 1 to bound step
        gW = np.ones_like(self.W); gWt = np.ones_like(self.Wt)
        gb = np.ones_like(self.b); gbt = np.ones_like(self.bt)

        logx = np.log(self.coo_x)
        fx = self._f(self.coo_x)
        n = len(self.coo_x)
        self.history = []
        for _ in range(epochs):
            perm = rng.permutation(n)
            total = 0.0
            for idx in perm:
                i, j = self.coo_i[idx], self.coo_j[idx]
                # prediction error: w_i·w~_j + b_i + b~_j - log X_ij
                diff = self.W[i] @ self.Wt[j] + self.b[i] + self.bt[j] - logx[idx]
                w = fx[idx]
                total += 0.5 * w * diff * diff
                g = w * diff                          # shared scalar gradient
                # gradients
                gradW = g * self.Wt[j]
                gradWt = g * self.W[i]
                # AdaGrad updates: lr / sqrt(accumulated sq grad)
                gW[i] += gradW ** 2; gWt[j] += gradWt ** 2
                gb[i] += g * g; gbt[j] += g * g
                self.W[i] -= self.lr * gradW / np.sqrt(gW[i])
                self.Wt[j] -= self.lr * gradWt / np.sqrt(gWt[j])
                self.b[i] -= self.lr * g / np.sqrt(gb[i])
                self.bt[j] -= self.lr * g / np.sqrt(gbt[j])
            self.history.append(total / n)
        # GloVe uses the SUM/average of the two vector sets as the final embedding
        self.embeddings = self.W + self.Wt
        return self

    def vec(self, w):
        return self.embeddings[self.stoi[w]]

    def most_similar(self, w, k=5):
        q = self.vec(w); q = q / (np.linalg.norm(q) + 1e-9)
        M = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9)
        sim = M @ q
        order = np.argsort(-sim)
        return [(self.itos[i], float(sim[i])) for i in order if self.itos[i] != w][:k]


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — same objective with autograd
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device():
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GloVeTorch(nn.Module):
    def __init__(self, vocab, dim=50, x_max=100, alpha=0.75):
        super().__init__()
        self.W = nn.Embedding(vocab, dim)
        self.Wt = nn.Embedding(vocab, dim)
        self.b = nn.Embedding(vocab, 1)
        self.bt = nn.Embedding(vocab, 1)
        for p in (self.W, self.Wt):
            nn.init.uniform_(p.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.bt.weight)
        self.x_max, self.alpha = x_max, alpha

    def forward(self, i, j, x):
        pred = (self.W(i) * self.Wt(j)).sum(-1) + self.b(i).squeeze(-1) + self.bt(j).squeeze(-1)
        f = torch.where(x < self.x_max, (x / self.x_max) ** self.alpha, torch.ones_like(x))
        return (f * (pred - torch.log(x)) ** 2).mean()


def train_glove_torch(glove_np, dim=50, epochs=80, lr=0.05):
    """Reuse the NumPy object's vocab + co-occurrence arrays to train torch."""
    dev = get_device()
    model = GloVeTorch(len(glove_np.itos), dim, glove_np.x_max, glove_np.alpha).to(dev)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    i = torch.as_tensor(glove_np.coo_i, device=dev)
    j = torch.as_tensor(glove_np.coo_j, device=dev)
    x = torch.as_tensor(glove_np.coo_x, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        opt.zero_grad(); loss = model(i, j, x); loss.backward(); opt.step()
    emb = (model.W.weight + model.Wt.weight).detach().cpu().numpy()
    return model, emb


# ---------------------------------------------------------------------------
# 3. Demo — a tiny topical toy corpus
# ---------------------------------------------------------------------------
def toy_corpus() -> str:
    # Content-only sentences: animal words only co-occur with animal words,
    # fruit words only with fruit words. No shared filler -> clean topic split.
    animals = ("dog cat lion tiger wolf bark hunt run chase prey fur paws. "
               "dog cat hunt prey. lion tiger run chase. wolf dog bark hunt. "
               "cat tiger fur paws. lion wolf chase prey run. ")
    fruits = ("apple banana orange grape mango sweet juicy ripe peel seed grow. "
              "apple banana sweet juicy. orange grape ripe grow. mango apple peel seed. "
              "banana mango sweet grow. grape orange juicy ripe peel. ")
    s = ""
    for _ in range(60):
        s += animals + fruits
    return s


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    sents = _tokenize(toy_corpus())

    g = GloVeNumPy(dim=20, window=3, x_max=30, lr=0.05).fit(sents, epochs=150)
    print(f"Co-occurrence entries: {len(g.coo_x)}  |  vocab: {len(g.itos)}")
    print(f"Final weighted-LS loss: {g.history[-1]:.4f}")
    print("NumPy GloVe nearest neighbours:")
    for q in ("dog", "apple"):
        print(f"  {q:6s} -> {[t for t, _ in g.most_similar(q, 3)]}")

    # sanity: the reconstruction w_i·w~_j + b ≈ log X_ij for a frequent pair
    i, j = g.stoi["dog"], g.stoi["cat"]
    pred = g.W[i] @ g.Wt[j] + g.b[i] + g.bt[j]
    import math
    actual = None
    for a, b, x in zip(g.coo_i, g.coo_j, g.coo_x):
        if a == i and b == j:
            actual = math.log(x); break
    if actual is not None:
        print(f"\nReconstruct log X['dog','cat']: pred={pred:.3f} vs log X={actual:.3f}")

    _, emb = train_glove_torch(g, dim=20, epochs=120)
    print(f"\nTorch GloVe trained (AdaGrad); embedding shape = {emb.shape}")
    fruit_words = set("apple banana orange grape mango sweet juicy ripe peel seed grow".split())
    nn3 = {t for t, _ in g.most_similar("apple", 3)}
    print(f"'apple' neighbours all in fruit topic: {nn3 <= fruit_words}")


if __name__ == "__main__":
    demo()

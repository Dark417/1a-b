"""
fastText — subword (character n-gram) embeddings
================================================
word2vec gives every word its own vector and is helpless on words it never saw
(out-of-vocabulary, OOV). fastText fixes this by representing a word as the
**sum of its character n-gram vectors**. "playing" shares the n-grams "play",
"layi", ... with "played", so morphologically related words land near each other,
and a *brand-new* word like "playful" still gets a sensible vector by summing the
n-grams it does know. It is skip-gram with negative sampling, but the center
vector is built from subwords.

Variants implemented here:
    - Skip-gram with negative sampling over subword (char n-gram) vectors
    - Word vector = sum of its subword vectors (the fastText model)
    - Configurable n-gram range (minn..maxn) with boundary markers < >
    - OOV inference: vectors for words unseen during training

Training techniques demonstrated:
    - NEGATIVE SAMPLING (see nlp/embeddings/word2vec.py)
    - Subword hashing/composition for open-vocabulary embeddings

References:
    - Bojanowski, Grave, Joulin & Mikolov (2017), "Enriching Word Vectors with
      Subword Information"
"""

from __future__ import annotations

from collections import Counter

import numpy as np

SEED = 0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def char_ngrams(word: str, minn=3, maxn=5) -> list[str]:
    r"""
    Character n-grams of a word, with boundary markers '<' and '>' so that
    prefixes/suffixes are distinguishable (e.g. '<wh' as a start, 'ere>' as an
    end). The full word (with markers) is always included as one "n-gram" so a
    seen word keeps its own dedicated vector too.
        where -> <where>  ->  <wh, whe, her, ere, re>, <whe, ..., <where>
    """
    w = "<" + word + ">"
    grams = set()
    L = len(w)
    for n in range(minn, maxn + 1):
        for i in range(L - n + 1):
            grams.add(w[i:i + n])
    grams.add("<" + word + ">")  # the whole word as a special token
    return sorted(grams)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — subword skip-gram with negative sampling
# ---------------------------------------------------------------------------
class FastTextNumPy:
    r"""
    A word's input vector is the SUM of its subword vectors:
        v_w = Σ_{g ∈ G(w)} z_g
    where G(w) is the set of char n-grams (+ the whole word). Output/context
    vectors u_o are per-word (one row each), as in skip-gram.

    SGNS objective for a (center w, true context o), negatives k ~ P_n:
        L = -log σ(u_o · v_w) - Σ_k log σ(-u_k · v_w).
    The gradient w.r.t. v_w is split EQUALLY back to every subword that built it:
        dz_g += dv_w   for each g ∈ G(w).
    """

    def __init__(self, dim=50, window=2, neg=5, minn=3, maxn=5, lr=0.05,
                 buckets=20000, seed=SEED):
        self.dim, self.window, self.neg = dim, window, neg
        self.minn, self.maxn, self.lr = minn, maxn, lr
        self.buckets, self.seed = buckets, seed

    # subword id via hashing (fastText hashes into a fixed-size table) ---------
    @staticmethod
    def _hash(s: str) -> int:
        # FNV-1a — deterministic across processes (unlike Python's hash()).
        h = 2166136261
        for ch in s.encode("utf-8"):
            h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
        return h

    def _subword_ids(self, word):
        ids = []
        for g in char_ngrams(word, self.minn, self.maxn):
            ids.append(self._hash(g) % self.buckets)
        return ids

    def build_vocab(self, sentences):
        counts = Counter(w for s in sentences for w in s)
        self.itos = list(counts)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.array([counts[w] for w in self.itos], float)
        p = self.counts ** 0.75
        self.noise = p / p.sum()
        V = len(self.itos)
        rng = np.random.default_rng(self.seed)
        # Z: subword (input) table; U: per-word context (output) table
        self.Z = (rng.random((self.buckets, self.dim)) - 0.5) / self.dim
        self.U = np.zeros((V, self.dim))
        # cache subword id lists for known words
        self._cache = {w: self._subword_ids(w) for w in self.itos}
        return self

    def word_vector(self, word):
        """v_w = sum of subword vectors — works even for OOV words."""
        ids = self._cache.get(word) or self._subword_ids(word)
        return self.Z[ids].sum(0)

    def _pairs(self, sentences):
        for s in sentences:
            ids = [w for w in s if w in self.stoi]
            for i, c in enumerate(ids):
                lo, hi = max(0, i - self.window), min(len(ids), i + self.window + 1)
                for j in range(lo, hi):
                    if j != i:
                        yield c, ids[j]

    def fit(self, sentences, epochs=50):
        self.build_vocab(sentences)
        rng = np.random.default_rng(self.seed)
        pairs = list(self._pairs(sentences))
        self.history = []
        for _ in range(epochs):
            rng.shuffle(pairs)
            loss = 0.0
            for center, ctx in pairs:
                sub = self._cache[center]                 # subword ids of center
                v = self.Z[sub].sum(0)                    # v_w = Σ z_g
                o = self.stoi[ctx]
                negs = rng.choice(len(self.itos), self.neg, p=self.noise)
                targets = np.concatenate([[o], negs])
                labels = np.concatenate([[1.0], np.zeros(self.neg)])
                scores = _sigmoid(self.U[targets] @ v)
                err = scores - labels                     # (1+neg,)
                dU = np.outer(err, v)
                dv = err @ self.U[targets]                # grad w.r.t. v_w
                self.U[targets] -= self.lr * dU
                # split dv back to EVERY subword that composed v_w
                self.Z[sub] -= self.lr * dv
                loss += -np.log(scores[0] + 1e-9) - np.sum(np.log(1 - scores[1:] + 1e-9))
            self.history.append(loss / max(len(pairs), 1))
        return self

    def most_similar(self, word, k=5, candidates=None):
        q = self.word_vector(word); q = q / (np.linalg.norm(q) + 1e-9)
        cand = candidates if candidates is not None else self.itos
        M = np.stack([self.word_vector(w) for w in cand])
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        sim = M @ q
        order = np.argsort(-sim)
        return [(str(cand[i]), float(sim[i])) for i in order if cand[i] != word][:k]


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — subword SGNS with an EmbeddingBag (sum pooling)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FastTextTorch(nn.Module):
    """EmbeddingBag with mode='sum' realises v_w = Σ subword vectors directly."""

    def __init__(self, buckets, vocab, dim=50):
        super().__init__()
        self.sub = nn.EmbeddingBag(buckets, dim, mode="sum")
        self.context = nn.Embedding(vocab, dim)
        nn.init.uniform_(self.sub.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, sub_ids, offsets, pos, neg):
        v = self.sub(sub_ids, offsets)                       # (B, d)
        pos_s = (self.context(pos) * v).sum(-1)              # (B,)
        neg_s = torch.bmm(self.context(neg), v.unsqueeze(-1)).squeeze(-1)  # (B,K)
        loss = -(F.logsigmoid(pos_s) + F.logsigmoid(-neg_s).sum(1))
        return loss.mean()


def train_fasttext_torch(ft_np, sentences, dim=50, neg=5, epochs=40, lr=0.01):
    """Reuse the NumPy object's vocab/subword hashing to train the torch model."""
    dev = get_device()
    model = FastTextTorch(ft_np.buckets, len(ft_np.itos), dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    pairs = list(ft_np._pairs(sentences))
    # flatten variable-length subword id lists for EmbeddingBag
    flat, offsets, pos = [], [], []
    for center, ctx in pairs:
        offsets.append(len(flat))
        flat.extend(ft_np._cache[center])
        pos.append(ft_np.stoi[ctx])
    sub_ids = torch.tensor(flat, device=dev)
    offsets = torch.tensor(offsets, device=dev)
    pos = torch.tensor(pos, device=dev)
    for _ in range(epochs):
        negs = torch.tensor(rng.choice(len(ft_np.itos), (len(pairs), neg),
                                       p=ft_np.noise), device=dev)
        opt.zero_grad(); loss = model(sub_ids, offsets, pos, negs)
        loss.backward(); opt.step()
    return model


# ---------------------------------------------------------------------------
# 3. Demo — show morphology + OOV generalization
# ---------------------------------------------------------------------------
def toy_corpus():
    """Sentences sharing morphological families so subwords carry signal."""
    rng = np.random.default_rng(SEED)
    play = "play plays playing played player".split()
    walk = "walk walks walking walked walker".split()
    jump = "jump jumps jumping jumped jumper".split()
    glue = "the a he she they will can".split()
    sents = []
    for _ in range(150):
        fam = [play, walk, jump][rng.integers(0, 3)]
        s = list(rng.choice(glue, 2)) + list(rng.choice(fam, 3))
        rng.shuffle(s)
        sents.append(s)
    return sents


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    sents = toy_corpus()
    ft = FastTextNumPy(dim=30, window=2, neg=5, minn=3, maxn=5).fit(sents, epochs=60)

    print("Subword n-grams of 'playing':")
    print("  ", char_ngrams("playing", 3, 5))

    print("\nMorphological neighbours (subwords cluster word families):")
    for q in ("playing", "walked"):
        print(f"  {q:8s} -> {[t for t, _ in ft.most_similar(q, 3)]}")

    # --- OOV: 'playful' was NEVER in training, but shares <pl, pla, play ... ---
    oov = "playful"
    print(f"\nOOV word '{oov}' is in vocab? {oov in ft.stoi}")
    v = ft.word_vector(oov)
    print(f"  Built a {v.shape[0]}-dim vector from {len(ft._subword_ids(oov))} subwords.")
    # Which known word is it closest to?
    nn = ft.most_similar(oov, 3)
    print(f"  Nearest known words: {[t for t, _ in nn]}")
    play_fam = set("play plays playing played player".split())
    print(f"  Top neighbour in the 'play' family: {nn[0][0] in play_fam}")

    train_fasttext_torch(ft, sents, dim=30, epochs=30)
    print("\nTorch fastText trained (EmbeddingBag sum = Σ subword vectors).")


if __name__ == "__main__":
    demo()

"""
Sequence Tagging: BiLSTM + Linear-Chain CRF
===========================================
Named-entity recognition (NER) and POS tagging assign a label to *every* token.
A BiLSTM reads the sentence in both directions and emits per-token label scores —
but predicting each label independently ignores that labels follow grammar
(e.g. an I-PER tag can't start a span; B-PER is usually followed by I-PER, not
I-LOC). A **linear-chain CRF** sits on top and scores the *whole label sequence*
jointly via learned transition potentials, decoding with **Viterbi** and training
with the **forward-algorithm** partition function.

We implement a from-scratch NumPy CRF (forward algorithm + Viterbi) and a
PyTorch **BiLSTM-CRF** trained end-to-end on a tiny synthetic tagging task.

Variants implemented here:
    - Standalone linear-chain CRF in NumPy (forward algorithm + Viterbi) — the
      math made explicit
    - BiLSTM emission scorer + CRF layer in PyTorch (negative log-likelihood loss)
    - Comparison: BiLSTM with independent softmax vs BiLSTM-CRF

Training techniques demonstrated:
    - The log-sum-exp trick for a numerically stable partition function
    - Structured prediction: scoring whole sequences, not independent tokens

References:
    - Lafferty, McCallum & Pereira (2001), "Conditional Random Fields"
    - Lample et al. (2016), "Neural Architectures for Named Entity Recognition"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def log_sum_exp(x, axis=-1):
    """Numerically stable log Σ exp(x) along an axis."""
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — linear-chain CRF (forward algorithm + Viterbi)
# ---------------------------------------------------------------------------
class LinearChainCRF:
    r"""
    A linear-chain CRF over tag sequences y given per-token emission scores E.

    **Potentials.** For a sentence of length L with K tags, we are given emission
    scores E[t, k] (how much token t likes tag k) and learn a transition matrix
    T[i, j] (score of going from tag i to tag j). The score of a full tag path y:

        score(y) = Σ_t E[t, y_t] + Σ_t T[y_{t-1}, y_t]                (+ start/end)

    **Probability.** Soft-max over ALL paths (the structured softmax):

        P(y | E) = exp(score(y)) / Z,   Z = Σ_{y'} exp(score(y')).

    **Partition function Z** is summed over K^L paths — but the chain structure
    lets the FORWARD ALGORITHM compute log Z in O(L K^2):

        α_1[k]   = start[k] + E[0, k]
        α_t[k]   = E[t, k] + logsumexp_i ( α_{t-1}[i] + T[i, k] )
        log Z    = logsumexp_k ( α_L[k] + end[k] )

    **Viterbi** is the same recursion with max instead of logsumexp, plus
    backpointers, to recover the single highest-scoring path.
    """

    def __init__(self, num_tags, seed=SEED):
        self.K = num_tags
        rng = np.random.default_rng(seed)
        self.T = rng.normal(0, 0.1, (num_tags, num_tags))  # transitions T[i,j]
        self.start = np.zeros(num_tags)                    # start[k]
        self.end = np.zeros(num_tags)                      # end[k]

    def score_path(self, E, y):
        """Unnormalized score of a specific tag path y given emissions E."""
        s = self.start[y[0]] + E[0, y[0]]
        for t in range(1, len(y)):
            s += self.T[y[t - 1], y[t]] + E[t, y[t]]
        s += self.end[y[-1]]
        return s

    def log_partition(self, E):
        """Forward algorithm: log Z = log Σ_paths exp(score). O(L K^2)."""
        alpha = self.start + E[0]                          # α_1
        for t in range(1, len(E)):
            # broadcast: scores[i,k] = α[i] + T[i,k]; sum over previous tag i
            scores = alpha[:, None] + self.T               # (K, K)
            alpha = E[t] + log_sum_exp(scores, axis=0)     # α_t[k]
        return log_sum_exp(alpha + self.end, axis=0)

    def neg_log_likelihood(self, E, y):
        """-log P(y|E) = log Z - score(y). The CRF training loss."""
        return self.log_partition(E) - self.score_path(E, y)

    def viterbi(self, E):
        """Decode the single best tag path (max-product). Returns (path, score)."""
        L = len(E)
        delta = self.start + E[0]                          # best score ending in k
        back = np.zeros((L, self.K), dtype=int)
        for t in range(1, L):
            scores = delta[:, None] + self.T               # (K_prev, K_cur)
            back[t] = np.argmax(scores, axis=0)            # best previous tag
            delta = E[t] + np.max(scores, axis=0)
        delta = delta + self.end
        best_last = int(np.argmax(delta))
        path = [best_last]
        for t in range(L - 1, 0, -1):                      # follow backpointers
            path.append(int(back[t, path[-1]]))
        path.reverse()
        return path, float(np.max(delta))

    # -- a tiny gradient-free trainer for the standalone NumPy CRF -----------
    def fit(self, emissions_list, tags_list, epochs=200, lr=0.05):
        """
        Train transitions by gradient on the CRF NLL (emissions are fixed/given).
        d(NLL)/dT[i,j] = E_P[count(i->j)] - count_gold(i->j). We get the expected
        counts from the forward-backward marginals; here we use a simple finite
        approximation via the analytic transition gradient.
        """
        self.history = []
        for _ in range(epochs):
            total = 0.0
            gT = np.zeros_like(self.T)
            gs = np.zeros_like(self.start)
            ge = np.zeros_like(self.end)
            for E, y in zip(emissions_list, tags_list):
                total += self.neg_log_likelihood(E, y)
                # gold (empirical) transition/start/end counts
                gs[y[0]] -= 1; ge[y[-1]] -= 1
                for t in range(1, len(y)):
                    gT[y[t - 1], y[t]] -= 1
                # expected counts under the model via forward-backward
                exp_start, exp_end, exp_trans = self._expected_counts(E)
                gs += exp_start; ge += exp_end; gT += exp_trans
            self.T -= lr * gT / len(tags_list)
            self.start -= lr * gs / len(tags_list)
            self.end -= lr * ge / len(tags_list)
            self.history.append(total / len(tags_list))
        return self

    def _expected_counts(self, E):
        """Forward-backward marginals -> expected start/end/transition counts."""
        L, K = E.shape
        # forward
        alpha = np.zeros((L, K)); alpha[0] = self.start + E[0]
        for t in range(1, L):
            alpha[t] = E[t] + log_sum_exp(alpha[t - 1][:, None] + self.T, axis=0)
        logZ = log_sum_exp(alpha[-1] + self.end, axis=0)
        # backward
        beta = np.zeros((L, K)); beta[-1] = self.end
        for t in range(L - 2, -1, -1):
            beta[t] = log_sum_exp(self.T + (E[t + 1] + beta[t + 1])[None, :], axis=1)
        # node marginals P(y_t=k)
        gamma = np.exp(alpha + beta - logZ)                # (L,K)
        exp_start = gamma[0]
        exp_end = gamma[-1]
        # edge marginals P(y_{t-1}=i, y_t=j)
        exp_trans = np.zeros((K, K))
        for t in range(1, L):
            xi = (alpha[t - 1][:, None] + self.T
                  + (E[t] + beta[t])[None, :] - logZ)
            exp_trans += np.exp(xi)
        return exp_start, exp_end, exp_trans


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — BiLSTM emission scorer + CRF layer
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

torch.set_num_threads(1)  # tiny CPU BiLSTM, keep it fast & deterministic


def get_device():
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _lse(x, dim):
    m, _ = x.max(dim, keepdim=True)
    return (m + (x - m).exp().sum(dim, keepdim=True).log()).squeeze(dim)


class CRFTorch(nn.Module):
    """Differentiable linear-chain CRF (single sequence; batch=1 for clarity)."""

    def __init__(self, num_tags):
        super().__init__()
        self.K = num_tags
        self.T = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))

    def log_partition(self, E):  # E: (L, K)
        alpha = self.start + E[0]
        for t in range(1, E.size(0)):
            alpha = E[t] + _lse(alpha.unsqueeze(1) + self.T, 0)
        return _lse(alpha + self.end, 0)

    def score(self, E, y):
        s = self.start[y[0]] + E[0, y[0]]
        for t in range(1, len(y)):
            s = s + self.T[y[t - 1], y[t]] + E[t, y[t]]
        return s + self.end[y[-1]]

    def nll(self, E, y):
        return self.log_partition(E) - self.score(E, y)

    @torch.no_grad()
    def viterbi(self, E):
        L = E.size(0)
        delta = self.start + E[0]
        back = torch.zeros(L, self.K, dtype=torch.long)
        for t in range(1, L):
            scores = delta.unsqueeze(1) + self.T
            best, idx = scores.max(0)
            back[t] = idx; delta = E[t] + best
        delta = delta + self.end
        last = int(delta.argmax())
        path = [last]
        for t in range(L - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return path[::-1]


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab, num_tags, emb=24, hidden=32, use_crf=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.emit = nn.Linear(2 * hidden, num_tags)   # emission scores E[t,k]
        self.use_crf = use_crf
        self.crf = CRFTorch(num_tags) if use_crf else None
        self.K = num_tags

    def emissions(self, x):  # x: (L,) -> E: (L, K)
        e = self.embed(x).unsqueeze(0)
        out, _ = self.lstm(e)
        return self.emit(out.squeeze(0))

    def loss(self, x, y):
        E = self.emissions(x)
        if self.use_crf:
            return self.crf.nll(E, y)
        return nn.functional.cross_entropy(E, torch.as_tensor(y, device=E.device))

    def predict(self, x):
        E = self.emissions(x)
        if self.use_crf:
            return self.crf.viterbi(E)
        return E.argmax(-1).tolist()

    def fit(self, sents, tags, epochs=60, lr=0.01):
        dev = get_device(); self.to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.history = []
        for _ in range(epochs):
            total = 0.0
            for x, y in zip(sents, tags):
                xb = torch.as_tensor(x, dtype=torch.long, device=dev)
                yb = torch.as_tensor(y, dtype=torch.long, device=dev)
                loss = self.loss(xb, yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                opt.step(); total += loss.item()
            self.history.append(total / len(sents))
        return self


# ---------------------------------------------------------------------------
# 3. Toy tagging task — a tiny "NER" with BIO tags and grammar constraints
# ---------------------------------------------------------------------------
# Vocabulary: words fall into roles. Tags use the BIO scheme.
WORDS = ["<pad>", "the", "a", "mr", "ms", "john", "smith", "paris", "london",
         "visited", "lives", "in", "and", "river"]
W2I = {w: i for i, w in enumerate(WORDS)}
TAGS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]
T2I = {t: i for i, t in enumerate(TAGS)}

PER_TITLE = ["mr", "ms"]
PER_NAME = ["john", "smith"]
LOC = ["paris", "london"]


def make_tagging_data(n=400, seed=SEED):
    """Generate sentences with PER (title + name) and LOC entities, BIO-tagged."""
    rng = np.random.default_rng(seed)
    sents, tags = [], []
    templates = [
        ("PER visited LOC", ),
        ("the PER lives in LOC", ),
        ("PER and PER visited LOC", ),
        ("a PER lives in LOC and LOC", ),
    ]
    for _ in range(n):
        tmpl = templates[rng.integers(len(templates))][0].split()
        s, y = [], []
        for tok in tmpl:
            if tok == "PER":
                # title (optional) + name -> B-PER (I-PER)
                if rng.random() < 0.6:
                    s.append(W2I[PER_TITLE[rng.integers(2)]]); y.append(T2I["B-PER"])
                    s.append(W2I[PER_NAME[rng.integers(2)]]); y.append(T2I["I-PER"])
                else:
                    s.append(W2I[PER_NAME[rng.integers(2)]]); y.append(T2I["B-PER"])
            elif tok == "LOC":
                s.append(W2I[LOC[rng.integers(2)]]); y.append(T2I["B-LOC"])
            else:
                s.append(W2I[tok]); y.append(T2I["O"])
        sents.append(s); tags.append(y)
    return sents, tags


def token_accuracy(model, sents, tags):
    correct = total = 0
    for x, y in zip(sents, tags):
        xb = torch.as_tensor(x, dtype=torch.long, device=next(model.parameters()).device)
        pred = model.predict(xb)
        correct += sum(int(a == b) for a, b in zip(pred, y)); total += len(y)
    return correct / total


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)

    # --- NumPy CRF sanity: forward partition is consistent, Viterbi works ---
    K = len(TAGS)
    crf = LinearChainCRF(K)
    E = np.random.default_rng(1).normal(size=(4, K))
    # log Z must be >= the score of any single path (it's a logsumexp over all)
    logZ = crf.log_partition(E)
    best_path, best_score = crf.viterbi(E)
    print(f"NumPy CRF  log Z = {logZ:.3f}  >=  best path score {best_score:.3f}: "
          f"{logZ >= best_score - 1e-6}")
    # brute-force check log Z on a short sequence (K^L paths)
    from itertools import product
    brute = log_sum_exp(np.array([crf.score_path(E, p) for p in product(range(K), repeat=4)]))
    print(f"           forward log Z = {logZ:.4f}  vs brute force {brute:.4f}")

    # standalone CRF training (forward-backward gradient on transitions only):
    # given fixed emissions, the CRF should learn that gold paths beat the rest.
    rng = np.random.default_rng(2)
    Es = [rng.normal(size=(5, K)) for _ in range(30)]
    ys = [LinearChainCRF(K).viterbi(e)[0] for e in Es]   # synthetic gold paths
    crf2 = LinearChainCRF(K).fit(Es, ys, epochs=120, lr=0.1)
    print(f"           standalone CRF NLL: {crf2.history[0]:.2f} -> {crf2.history[-1]:.2f}")

    # --- Train BiLSTM-CRF vs BiLSTM-softmax on the tagging task ---
    sents, tags = make_tagging_data(n=200)
    tr = slice(0, 150); te = slice(150, 200)

    soft = BiLSTMCRF(len(WORDS), K, use_crf=False).fit(sents[tr], tags[tr], epochs=25)
    crfm = BiLSTMCRF(len(WORDS), K, use_crf=True).fit(sents[tr], tags[tr], epochs=25)
    print(f"\nBiLSTM (softmax) token acc = {token_accuracy(soft, sents[te], tags[te]):.3f}")
    print(f"BiLSTM-CRF       token acc = {token_accuracy(crfm, sents[te], tags[te]):.3f}")

    # --- The CRF learns the grammar: I-PER almost never follows O ---
    Tm = crfm.crf.T.detach().numpy()
    print(f"\nLearned transition O -> I-PER = {Tm[T2I['O'], T2I['I-PER']]:+.2f} "
          f"(should be low; I- can't start a span)")
    print(f"Learned transition B-PER -> I-PER = {Tm[T2I['B-PER'], T2I['I-PER']]:+.2f} "
          f"(should be higher; valid continuation)")

    # --- Show one tagged sentence ---
    x = sents[te][0]
    pred = crfm.predict(torch.as_tensor(x, dtype=torch.long))
    print("\nexample:")
    print("  words:", [WORDS[i] for i in x])
    print("  gold :", [TAGS[i] for i in tags[te][0]])
    print("  pred :", [TAGS[i] for i in pred])


if __name__ == "__main__":
    demo()

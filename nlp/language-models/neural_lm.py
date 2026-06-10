"""
Neural Language Model
=====================
Count-based n-gram models can't share statistical strength across similar words
("dog" vs "puppy") and blow up combinatorially with context length. A **neural
language model** instead embeds each word into a dense vector, summarizes the
history with a neural network, and predicts the next word with a softmax over the
vocabulary. We implement two classic architectures:

  - a **fixed-window feed-forward (MLP) LM** (Bengio et al., 2003) — concatenate
    the embeddings of the previous n-1 words and push through an MLP;
  - an **LSTM LM** — consume the whole history token by token (unbounded context).

We train on a toy text, report **perplexity**, and **sample** new text with a
temperature knob.

Variants implemented here:
    - Feed-forward (n-gram) neural LM — NumPy (manual backprop) + PyTorch
    - LSTM recurrent LM — PyTorch
    - Temperature sampling for generation

Training techniques demonstrated:
    - Cross-entropy / softmax training; perplexity = exp(cross-entropy)
    - Weight tying discussion; gradient clipping for the RNN
    - Adam optimization

References:
    - Bengio et al. (2003), "A Neural Probabilistic Language Model"
    - Mikolov et al. (2010), "Recurrent neural network based language model"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _tokenize(text: str) -> list[str]:
    return text.lower().replace("\n", " ").split()


# ---------------------------------------------------------------------------
# 1. NumPy implementation — fixed-window feed-forward LM with manual backprop
# ---------------------------------------------------------------------------
class FeedForwardLMNumPy:
    r"""
    Bengio-style neural n-gram LM. Predict w_t from the previous (n-1) words.

    Forward:
        x   = [E[w_{t-n+1}] ; ... ; E[w_{t-1}]]   (concatenated embeddings)
        h   = tanh(W_1 x + b_1)
        z   = W_2 h + b_2                          (logits over the vocabulary)
        p   = softmax(z)
    Loss = cross-entropy  -log p[w_t].

    The gradients are standard MLP backprop; the embedding gradient is scattered
    back to the rows of E for the words in the context window.
    """

    def __init__(self, n=3, dim=16, hidden=32, lr=0.3, seed=SEED):
        self.n, self.dim, self.hidden, self.lr, self.seed = n, dim, hidden, lr, seed

    def build_vocab(self, tokens):
        self.itos = sorted(set(tokens))
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.V = len(self.itos)
        rng = np.random.default_rng(self.seed)
        self.E = rng.normal(0, 0.1, (self.V, self.dim))             # embeddings
        ctx = (self.n - 1) * self.dim
        self.W1 = rng.normal(0, 1 / np.sqrt(ctx), (ctx, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(self.hidden), (self.hidden, self.V))
        self.b2 = np.zeros(self.V)
        return self

    def _windows(self, ids):
        for i in range(self.n - 1, len(ids)):
            yield ids[i - self.n + 1:i], ids[i]      # (context words, target)

    def fit(self, tokens, epochs=300):
        self.build_vocab(tokens)
        ids = [self.stoi[w] for w in tokens]
        data = list(self._windows(ids))
        rng = np.random.default_rng(self.seed)
        self.history = []
        for _ in range(epochs):
            rng.shuffle(data)
            loss = 0.0
            for ctx, y in data:
                x = self.E[ctx].reshape(-1)                  # concat embeddings
                a = x @ self.W1 + self.b1
                h = np.tanh(a)
                z = h @ self.W2 + self.b2
                p = softmax(z)
                loss += -np.log(p[y] + 1e-12)
                # ---- backprop ----
                dz = p.copy(); dz[y] -= 1.0                  # dL/dz (softmax-CE)
                dW2 = np.outer(h, dz); db2 = dz
                dh = self.W2 @ dz
                da = dh * (1 - h ** 2)                        # tanh'
                dW1 = np.outer(x, da); db1 = da
                dx = self.W1 @ da
                # ---- update ----
                self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
                dE = dx.reshape(self.n - 1, self.dim)
                for k, wid in enumerate(ctx):
                    self.E[wid] -= self.lr * dE[k]
            self.history.append(loss / len(data))
        return self

    def _logits(self, ctx_ids):
        x = self.E[ctx_ids].reshape(-1)
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def perplexity(self, tokens):
        ids = [self.stoi.get(w, 0) for w in tokens]
        total, n = 0.0, 0
        for ctx, y in self._windows(ids):
            p = softmax(self._logits(ctx))
            total += -np.log(p[y] + 1e-12); n += 1
        return float(np.exp(total / max(n, 1)))

    def generate(self, prefix, n_words=20, temperature=1.0, seed=SEED):
        rng = np.random.default_rng(seed)
        ids = [self.stoi[w] for w in prefix]
        for _ in range(n_words):
            ctx = ids[-(self.n - 1):]
            if len(ctx) < self.n - 1:                        # left-pad with the first id
                ctx = [ids[0]] * (self.n - 1 - len(ctx)) + ctx
            logits = self._logits(ctx) / max(temperature, 1e-6)
            p = softmax(logits)
            ids.append(int(rng.choice(self.V, p=p)))
        return [self.itos[i] for i in ids]


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — feed-forward LM and an LSTM LM
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


class LSTMLanguageModel(nn.Module):
    """Embed -> LSTM -> linear softmax over the vocabulary (next-token prediction)."""

    def __init__(self, vocab, dim=32, hidden=64, layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.lstm = nn.LSTM(dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
        self.vocab = vocab

    def forward(self, x, hidden=None):
        e = self.embed(x)                       # (B, T, dim)
        out, hidden = self.lstm(e, hidden)      # (B, T, hidden)
        return self.head(out), hidden           # logits (B, T, vocab)

    def fit(self, ids, seq_len=16, epochs=120, lr=0.005, batch=8):
        dev = get_device(); self.to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        ids = torch.as_tensor(ids, dtype=torch.long, device=dev)
        N = (len(ids) - 1) // seq_len
        X = ids[:N * seq_len].view(N, seq_len)
        Y = ids[1:N * seq_len + 1].view(N, seq_len)   # targets shifted by one
        self.history = []
        for _ in range(epochs):
            perm = torch.randperm(N, device=dev)
            total = 0.0
            for s in range(0, N, batch):
                idx = perm[s:s + batch]
                logits, _ = self(X[idx])
                loss = loss_fn(logits.reshape(-1, self.vocab), Y[idx].reshape(-1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)  # tame RNN grads
                opt.step()
                total += loss.item()
            self.history.append(total / max(1, N // batch))
        return self

    @torch.no_grad()
    def perplexity(self, ids):
        dev = next(self.parameters()).device
        x = torch.as_tensor(ids[:-1], dtype=torch.long, device=dev).unsqueeze(0)
        y = torch.as_tensor(ids[1:], dtype=torch.long, device=dev).unsqueeze(0)
        logits, _ = self(x)
        ce = nn.functional.cross_entropy(logits.reshape(-1, self.vocab), y.reshape(-1))
        return float(torch.exp(ce))

    @torch.no_grad()
    def generate(self, prefix_ids, n_words=20, temperature=1.0, seed=SEED):
        dev = next(self.parameters()).device
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = list(prefix_ids)
        hidden = None
        x = torch.as_tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
        for _ in range(n_words):
            logits, hidden = self(x, hidden)
            logits = logits[0, -1] / max(temperature, 1e-6)
            p = torch.softmax(logits, -1).cpu()
            nxt = int(torch.multinomial(p, 1, generator=g))
            ids.append(nxt)
            x = torch.as_tensor([[nxt]], dtype=torch.long, device=dev)
        return ids


# ---------------------------------------------------------------------------
# 3. Demo — a tiny built-in toy text
# ---------------------------------------------------------------------------
def toy_text() -> str:
    base = (
        "the cat sat on the mat . "
        "the dog sat on the log . "
        "the cat chased the dog . "
        "the dog chased the cat . "
        "a happy cat naps on the mat . "
        "a happy dog runs in the park . "
        "the cat and the dog are friends . "
    )
    return base * 8   # repeat so the small net has enough signal


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    tokens = _tokenize(toy_text())
    cut = int(len(tokens) * 0.85)
    train_tok, test_tok = tokens[:cut], tokens[cut:]

    # ---- NumPy feed-forward LM ----
    ff = FeedForwardLMNumPy(n=3, dim=16, hidden=32, lr=0.3).fit(train_tok, epochs=150)
    print(f"NumPy FF-LM  train ppl = {ff.perplexity(train_tok):6.3f} | "
          f"test ppl = {ff.perplexity(test_tok):6.3f}")
    gen = ff.generate(["the", "cat"], n_words=12, temperature=0.7)
    print("  sample:", " ".join(gen))

    # ---- PyTorch LSTM LM (shares the NumPy vocab) ----
    ids = [ff.stoi[w] for w in train_tok]
    lm = LSTMLanguageModel(ff.V, dim=32, hidden=64).fit(ids, seq_len=12, epochs=120)
    test_ids = [ff.stoi.get(w, 0) for w in test_tok]
    print(f"Torch LSTM-LM             test ppl = {lm.perplexity(test_ids):6.3f}")
    prefix = [ff.stoi[w] for w in ["the", "cat"]]
    for temp in (0.5, 1.0):
        out = lm.generate(prefix, n_words=12, temperature=temp)
        print(f"  sample (T={temp}):", " ".join(ff.itos[i] for i in out))


if __name__ == "__main__":
    demo()

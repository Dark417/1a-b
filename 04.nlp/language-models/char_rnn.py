"""
Character-level RNN (char-RNN)
==============================
Model text one *character* at a time. With a tiny vocabulary (just the distinct
characters) the model has to learn spelling, word boundaries, and a bit of
grammar entirely from the sequence of characters — yet it produces surprisingly
word-like text. We implement Karpathy's classic **min-char-rnn**: a vanilla RNN
with **backpropagation through time (BPTT) done by hand in NumPy**, plus an
idiomatic PyTorch LSTM version. Generation uses **temperature sampling**.

Variants implemented here:
    - Vanilla RNN char-LM with manual BPTT + Adagrad (the min-char-rnn recipe)
    - PyTorch LSTM char-LM
    - Temperature-controlled sampling

Training techniques demonstrated:
    - Backpropagation through time (BPTT) and gradient CLIPPING (RNNs explode)
    - Adagrad adaptive learning rate (per Karpathy's original)
    - Temperature sampling for text generation

References:
    - Karpathy (2015), "The Unreasonable Effectiveness of Recurrent Neural
      Networks" / min-char-rnn.py
    - Mikolov et al. (2010), RNN language model
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation — vanilla RNN, manual BPTT (min-char-rnn)
# ---------------------------------------------------------------------------
class CharRNNNumPy:
    r"""
    A single-layer vanilla RNN over one-hot characters.

    Recurrence (per step t, x_t one-hot of the current char):
        h_t = tanh(Wxh x_t + Whh h_{t-1} + b_h)
        y_t = Why h_t + b_y                         (logits)
        p_t = softmax(y_t)                           (next-char distribution)
    Loss = Σ_t -log p_t[target_t]  (cross-entropy, char-level factorization).

    BPTT: unroll a chunk of `seq_len`, accumulate gradients backwards through
    time. dh at step t gets contributions from the output AND from h_{t+1}:
        dh_t = Why^T dy_t + Whh^T dh_raw_{t+1}
        dh_raw_t = (1 - h_t^2) ⊙ dh_t               (tanh')
    Gradients are CLIPPED to [-5, 5] because the recurrent product can explode.
    """

    def __init__(self, hidden=100, seq_len=25, lr=0.1, seed=SEED):
        self.H, self.seq_len, self.lr, self.seed = hidden, seq_len, lr, seed

    def build_vocab(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}
        self.V = len(self.chars)
        rng = np.random.default_rng(self.seed)
        H, V = self.H, self.V
        self.Wxh = rng.normal(0, 0.01, (H, V))
        self.Whh = rng.normal(0, 0.01, (H, H))
        self.Why = rng.normal(0, 0.01, (V, H))
        self.bh = np.zeros(H)
        self.by = np.zeros(V)
        return self

    def _loss_and_grads(self, inputs, targets, hprev):
        """Forward + BPTT over one chunk. Returns (loss, grads, last hidden)."""
        xs, hs, ps = {}, {-1: hprev}, {}
        loss = 0.0
        # ---- forward ----
        for t in range(len(inputs)):
            x = np.zeros(self.V); x[inputs[t]] = 1.0      # one-hot
            xs[t] = x
            hs[t] = np.tanh(self.Wxh @ x + self.Whh @ hs[t - 1] + self.bh)
            y = self.Why @ hs[t] + self.by
            e = np.exp(y - y.max()); p = e / e.sum()
            ps[t] = p
            loss += -np.log(p[targets[t]] + 1e-12)
        # ---- backward (BPTT) ----
        dWxh = np.zeros_like(self.Wxh); dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh); dby = np.zeros_like(self.by)
        dh_next = np.zeros(self.H)
        for t in reversed(range(len(inputs))):
            dy = ps[t].copy(); dy[targets[t]] -= 1.0      # softmax-CE grad
            dWhy += np.outer(dy, hs[t]); dby += dy
            dh = self.Why.T @ dy + dh_next                # backprop into h_t
            dh_raw = (1 - hs[t] ** 2) * dh                # through tanh
            dbh += dh_raw
            dWxh += np.outer(dh_raw, xs[t])
            dWhh += np.outer(dh_raw, hs[t - 1])
            dh_next = self.Whh.T @ dh_raw                 # to previous step
        grads = [dWxh, dWhh, dWhy, dbh, dby]
        for g in grads:
            np.clip(g, -5, 5, out=g)                      # gradient clipping
        return loss, grads, hs[len(inputs) - 1]

    def fit(self, text, epochs=40):
        self.build_vocab(text)
        data = [self.stoi[c] for c in text]
        params = [self.Wxh, self.Whh, self.Why, self.bh, self.by]
        mem = [np.zeros_like(p) for p in params]          # Adagrad accumulators
        self.history = []
        n = self.seq_len
        for _ in range(epochs):
            hprev = np.zeros(self.H)
            total, steps = 0.0, 0
            p = 0
            while p + n + 1 <= len(data):
                inputs = data[p:p + n]
                targets = data[p + 1:p + n + 1]
                loss, grads, hprev = self._loss_and_grads(inputs, targets, hprev)
                # Adagrad update: param -= lr * g / sqrt(accumulated g^2)
                for param, g, m in zip(params, grads, mem):
                    m += g * g
                    param -= self.lr * g / np.sqrt(m + 1e-8)
                total += loss / n; steps += 1; p += n
            self.history.append(total / max(steps, 1))
        return self

    def sample(self, seed_char, n_chars=200, temperature=1.0, seed=SEED):
        rng = np.random.default_rng(seed)
        h = np.zeros(self.H)
        idx = self.stoi.get(seed_char, 0)
        out = [idx]
        for _ in range(n_chars):
            x = np.zeros(self.V); x[idx] = 1.0
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = (self.Why @ h + self.by) / max(temperature, 1e-6)
            e = np.exp(y - y.max()); p = e / e.sum()
            idx = int(rng.choice(self.V, p=p))
            out.append(idx)
        return "".join(self.itos[i] for i in out)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — LSTM char-LM
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

# Single-threaded keeps the tiny CPU LSTM fast/deterministic in this tutorial.
torch.set_num_threads(1)


def get_device():
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CharLSTMTorch(nn.Module):
    def __init__(self, vocab, hidden=128, layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
        self.vocab, self.hidden = vocab, hidden

    def forward(self, x, state=None):
        e = self.embed(x)
        out, state = self.lstm(e, state)
        return self.head(out), state

    def fit(self, data, stoi, seq_len=25, epochs=60, lr=0.005):
        dev = get_device(); self.to(dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        ids = torch.as_tensor(data, dtype=torch.long, device=dev)
        N = (len(ids) - 1) // seq_len
        X = ids[:N * seq_len].view(N, seq_len)
        Y = ids[1:N * seq_len + 1].view(N, seq_len)
        self.history = []
        for _ in range(epochs):
            logits, _ = self(X)
            loss = loss_fn(logits.reshape(-1, self.vocab), Y.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5.0)   # RNNs explode
            opt.step()
            self.history.append(loss.item())
        self.stoi, self.itos = stoi, {i: c for c, i in stoi.items()}
        return self

    @torch.no_grad()
    def sample(self, seed_char, n_chars=200, temperature=1.0, seed=SEED):
        dev = next(self.parameters()).device
        g = torch.Generator().manual_seed(seed)
        idx = self.stoi.get(seed_char, 0)
        out = [idx]
        state = None
        x = torch.tensor([[idx]], device=dev)
        for _ in range(n_chars):
            logits, state = self(x, state)
            logits = logits[0, -1] / max(temperature, 1e-6)
            p = torch.softmax(logits, -1).cpu()
            idx = int(torch.multinomial(p, 1, generator=g))
            out.append(idx); x = torch.tensor([[idx]], device=dev)
        return "".join(self.itos[i] for i in out)


# ---------------------------------------------------------------------------
# 3. Demo — train on a short built-in string
# ---------------------------------------------------------------------------
def toy_text() -> str:
    return (
        "the quick brown fox jumps over the lazy dog. "
        "a quick brown dog jumps over the lazy fox. "
        "the lazy fox and the quick dog are good friends. "
    ) * 6


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    text = toy_text()

    rnn = CharRNNNumPy(hidden=100, seq_len=25, lr=0.1).fit(text, epochs=40)
    print(f"NumPy char-RNN: vocab={rnn.V} chars, final loss={rnn.history[-1]:.3f}")
    print("  sample (T=0.5):")
    print("   ", repr(rnn.sample("t", n_chars=80, temperature=0.5)))

    data = [rnn.stoi[c] for c in text]
    lm = CharLSTMTorch(rnn.V, hidden=128).fit(data, rnn.stoi, seq_len=25, epochs=60)
    print(f"\nTorch char-LSTM: final loss={lm.history[-1]:.3f}")
    for temp in (0.4, 1.0):
        print(f"  sample (T={temp}):")
        print("   ", repr(lm.sample("t", n_chars=80, temperature=temp)))


if __name__ == "__main__":
    demo()

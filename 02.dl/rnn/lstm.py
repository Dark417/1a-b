"""
Long Short-Term Memory (LSTM)
=============================
A gated recurrent cell with a protected cell state that flows through time with
mostly additive updates, so gradients don't vanish as fast as in a vanilla RNN.
The classic fix for long-range dependencies.

Variants implemented here:
    - LSTM cell (input/forget/output gates) — full forward + BPTT in NumPy
    - GRU comparison (PyTorch) — fewer gates, similar power
    - The constant-error-carousel intuition

Training techniques demonstrated:
    - Gated additive memory as a vanishing-gradient remedy (vs dl/rnn/rnn)
    - Teacher forcing is discussed in nlp/seq2seq

References:
    - Hochreiter & Schmidhuber (1997); Cho et al. (2014) for GRU
"""

from __future__ import annotations

import numpy as np

SEED = 0


def sigmoid(z):  return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
def softmax(z):  z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — one LSTM cell, unrolled, BPTT by hand
# ---------------------------------------------------------------------------
class LSTMNumPy:
    r"""
    Gates (at each step, inputs = [x_t, h_{t-1}] concatenated):
        f_t = σ(W_f z + b_f)      forget gate   — what to erase from memory
        i_t = σ(W_i z + b_i)      input gate    — what to write
        g_t = tanh(W_g z + b_g)   candidate     — the new content
        o_t = σ(W_o z + b_o)      output gate   — what to expose
    State:
        c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t         (mostly ADDITIVE -> stable grad)
        h_t = o_t ⊙ tanh(c_t)
    The additive path ∂c_t/∂c_{t-1} = f_t (≈1 when the gate stays open) is the
    "constant error carousel" that keeps gradients alive across many steps.
    """

    def __init__(self, in_dim, hidden, out_dim, lr=0.1, clip=5.0, seed=SEED):
        rng = np.random.default_rng(seed)
        Z = in_dim + hidden
        s = 1.0 / np.sqrt(Z)
        self.W = {k: rng.normal(0, s, (Z, hidden)) for k in "figo"}
        self.b = {k: np.zeros(hidden) for k in "figo"}
        self.b["f"] += 1.0                       # forget-gate bias=1: remember by default
        self.Why = rng.normal(0, 1 / np.sqrt(hidden), (hidden, out_dim))
        self.by = np.zeros(out_dim)
        self.H, self.in_dim, self.lr, self.clip = hidden, in_dim, lr, clip

    def forward(self, X):
        T = len(X); H = self.H
        self.cache = []
        h = np.zeros(H); c = np.zeros(H)
        for t in range(T):
            z = np.concatenate([X[t], h])
            f = sigmoid(z @ self.W["f"] + self.b["f"])
            i = sigmoid(z @ self.W["i"] + self.b["i"])
            g = np.tanh(z @ self.W["g"] + self.b["g"])
            o = sigmoid(z @ self.W["o"] + self.b["o"])
            c = f * c + i * g
            h = o * np.tanh(c)
            self.cache.append((z, f, i, g, o, c, h))
        self.logits = h @ self.Why + self.by
        return softmax(self.logits)

    def backward(self, y):
        T = len(self.cache)
        p = softmax(self.logits); p[y] -= 1.0
        dW = {k: np.zeros_like(self.W[k]) for k in "figo"}
        db = {k: np.zeros_like(self.b[k]) for k in "figo"}
        dWhy = np.outer(self.cache[-1][-1], p); dby = p
        dh = p @ self.Why.T
        dc_next = np.zeros(self.H)
        c_prev_list = [np.zeros(self.H)] + [self.cache[t][5] for t in range(T - 1)]
        for t in reversed(range(T)):
            z, f, i, g, o, c, h = self.cache[t]
            c_prev = c_prev_list[t]
            do = dh * np.tanh(c)
            dc = dh * o * (1 - np.tanh(c) ** 2) + dc_next
            df = dc * c_prev; di = dc * g; dg = dc * i
            da = {                                       # pre-activation grads
                "o": do * o * (1 - o), "f": df * f * (1 - f),
                "i": di * i * (1 - i), "g": dg * (1 - g ** 2),
            }
            for k in "figo":
                dW[k] += np.outer(z, da[k]); db[k] += da[k]
            dz = sum(da[k] @ self.W[k].T for k in "figo")
            dh = dz[self.in_dim:]                        # split grad back to h_{t-1}
            dc_next = dc * f                             # the additive carousel
        grads = (dW, db, dWhy, dby)
        # global-norm clipping
        flat = [g for d in (dW, db) for g in d.values()] + [dWhy, dby]
        total = np.sqrt(sum((g ** 2).sum() for g in flat))
        if total > self.clip:
            scale = self.clip / total
            for d in (dW, db):
                for k in d: d[k] *= scale
            dWhy *= scale; dby *= scale
        return grads

    def step(self, grads):
        dW, db, dWhy, dby = grads
        for k in "figo":
            self.W[k] -= self.lr * dW[k]; self.b[k] -= self.lr * db[k]
        self.Why -= self.lr * dWhy; self.by -= self.lr * dby

    def fit(self, seqs, labels, epochs=40):
        self.history = []
        for _ in range(epochs):
            loss = 0.0
            for X, y in zip(seqs, labels):
                p = self.forward(X); loss += -np.log(p[y] + 1e-12)
                self.step(self.backward(y))
            self.history.append(loss / len(seqs))
        return self

    def predict(self, seqs):
        return np.array([self.forward(X).argmax() for X in seqs])


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (LSTM and GRU)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class RecurrentTorch(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, cell="lstm"):
        super().__init__()
        Cell = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = Cell(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])

    def fit(self, seqs, labels, epochs=60, lr=0.01):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        X = torch.as_tensor(np.array(seqs), dtype=torch.float32, device=dev)
        y = torch.as_tensor(labels, dtype=torch.long, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad(); loss = loss_fn(self(X), y); loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5.0); opt.step()
        return self

    @torch.no_grad()
    def predict(self, seqs):
        dev = next(self.parameters()).device
        X = torch.as_tensor(np.array(seqs), dtype=torch.float32, device=dev)
        return self(X).argmax(1).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo — same long-memory task where the vanilla RNN struggles
# ---------------------------------------------------------------------------
def make_memory_task(n=300, T=25, seed=SEED):
    """Label = identity of the FIRST token; rest is noise. Needs long memory."""
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        first = rng.integers(0, 2)
        x = rng.normal(0, 0.3, size=(T, 2)); x[0, first] += 2.0
        seqs.append(x); labels.append(first)
    return np.array(seqs), np.array(labels)


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    seqs, labels = make_memory_task(n=300, T=25)
    tr, te = slice(0, 240), slice(240, 300)

    net = LSTMNumPy(2, 24, 2, lr=0.1).fit(seqs[tr], labels[tr], epochs=40)
    print(f"NumPy LSTM  test acc = {np.mean(net.predict(seqs[te]) == labels[te]):.3f}")

    for cell in ("lstm", "gru"):
        m = RecurrentTorch(2, 24, 2, cell=cell).fit(seqs[tr], labels[tr])
        print(f"Torch {cell.upper():4s} test acc = {np.mean(m.predict(seqs[te]) == labels[te]):.3f}")

    print("\nLSTM keeps the t=0 signal across 25 steps via the additive cell state —")
    print("compare with dl/rnn/rnn.py, where gradients to t=0 vanish.")


if __name__ == "__main__":
    demo()

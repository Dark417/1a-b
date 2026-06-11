"""
Recurrent Neural Network (vanilla RNN)
======================================
Process sequences by carrying a hidden state forward in time and updating it at
each step with the same shared weights. Trained by Backpropagation Through Time
(BPTT). The classic demonstration of **vanishing / exploding gradients** — and
why gradient clipping and gated cells (LSTM/GRU) were invented.

Variants implemented here:
    - Vanilla Elman RNN (tanh), many-to-one for sequence classification
    - BPTT with explicit per-time-step gradients
    - Gradient clipping (the fix for exploding gradients)

Training techniques demonstrated:
    - BPTT (chain rule unrolled over time)
    - VANISHING / EXPLODING GRADIENTS over the sequence length (measured)
    - Gradient clipping (see training-techniques/README.md)

References:
    - Elman (1990); Bengio et al. (1994) on long-term dependencies; Pascanu et al. (2013)
"""

from __future__ import annotations

import numpy as np

SEED = 0


def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — BPTT by hand
# ---------------------------------------------------------------------------
class RNNNumPy:
    r"""
    h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)
    Read out the LAST hidden state:  logits = h_T W_hy + b_y  (many-to-one).

    BPTT: the gradient w.r.t. h_t accumulates a product of Jacobians
        ∂h_k/∂h_{k-1} = diag(1 - h_k^2) W_hh^T
    Repeated multiplication of these → vanishing (||·||<1) or exploding (||·||>1).
    """

    def __init__(self, in_dim, hidden, out_dim, lr=0.05, clip=None, seed=SEED):
        rng = np.random.default_rng(seed)
        self.Wxh = rng.normal(0, 1 / np.sqrt(in_dim), (in_dim, hidden))
        self.Whh = rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden))
        self.Why = rng.normal(0, 1 / np.sqrt(hidden), (hidden, out_dim))
        self.bh = np.zeros(hidden); self.by = np.zeros(out_dim)
        self.hidden, self.lr, self.clip = hidden, lr, clip

    def forward(self, X):                        # X: (T, in_dim)
        T = len(X)
        self.X = X
        self.h = np.zeros((T + 1, self.hidden))  # h[-1] used as h_0 = 0
        for t in range(T):
            self.h[t] = np.tanh(X[t] @ self.Wxh + self.h[t - 1] @ self.Whh + self.bh)
        self.logits = self.h[T - 1] @ self.Why + self.by
        return softmax(self.logits)

    def backward(self, y):
        T = len(self.X)
        p = softmax(self.logits); p[y] -= 1.0    # dL/dlogits
        dWhy = np.outer(self.h[T - 1], p); dby = p
        dWxh = np.zeros_like(self.Wxh); dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dh = p @ self.Why.T                      # grad into last hidden state
        self.bptt_norms = []                     # ||dh|| over time (vanishing demo)
        for t in reversed(range(T)):
            draw = dh * (1 - self.h[t] ** 2)     # through tanh
            dWxh += np.outer(self.X[t], draw)
            dWhh += np.outer(self.h[t - 1], draw)
            dbh += draw
            self.bptt_norms.append(np.linalg.norm(dh))
            dh = draw @ self.Whh.T               # propagate one step back in time
        self.bptt_norms.reverse()
        grads = [dWxh, dWhh, dWhy, dbh, dby]
        if self.clip is not None:                # gradient clipping (global norm)
            total = np.sqrt(sum((g ** 2).sum() for g in grads))
            if total > self.clip:
                grads = [g * (self.clip / total) for g in grads]
        return grads

    def step(self, grads):
        for p, g in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], grads):
            p -= self.lr * g

    def fit(self, seqs, labels, epochs=30):
        self.history = []
        for _ in range(epochs):
            loss = 0.0
            for X, y in zip(seqs, labels):
                p = self.forward(X)
                loss += -np.log(p[y] + 1e-12)
                self.step(self.backward(y))
            self.history.append(loss / len(seqs))
        return self

    def predict(self, seqs):
        return np.array([self.forward(X).argmax() for X in seqs])


# ---------------------------------------------------------------------------
# 2. PyTorch implementation
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class RNNTorch(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.rnn = nn.RNN(in_dim, hidden, batch_first=True, nonlinearity="tanh")
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):                        # x: (B, T, in_dim)
        out, h = self.rnn(x)
        return self.head(out[:, -1])             # last time-step

    def fit(self, seqs, labels, epochs=30, lr=0.05, clip=1.0):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        X = torch.as_tensor(np.array(seqs), dtype=torch.float32, device=dev)
        y = torch.as_tensor(labels, dtype=torch.long, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), clip)   # gradient clipping
            opt.step()
        return self

    @torch.no_grad()
    def predict(self, seqs):
        dev = next(self.parameters()).device
        X = torch.as_tensor(np.array(seqs), dtype=torch.float32, device=dev)
        return self(X).argmax(1).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo  — a "remember the first token" task (long-range dependency)
# ---------------------------------------------------------------------------
def make_memory_task(n=300, T=15, seed=SEED):
    """Label = identity of the FIRST token; the rest is noise. Needs long memory."""
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        first = rng.integers(0, 2)
        x = rng.normal(0, 0.3, size=(T, 2))
        x[0, first] += 2.0                       # signal only at t=0
        seqs.append(x); labels.append(first)
    return np.array(seqs), np.array(labels)


def vanishing_demo():
    """Show ||grad|| decaying back through time for a long sequence."""
    np.random.seed(SEED)
    X, y = make_memory_task(n=1, T=30)
    net = RNNNumPy(2, 16, 2)
    net.forward(X[0]); net.backward(y[0])
    norms = np.array(net.bptt_norms)
    print("||gradient at hidden state h_t|| from t=T (recent) back to t=0 (distant):")
    print("  " + " ".join(f"{g:6.1e}" for g in norms[::3]))
    print(f"  decay factor over {len(norms)} steps: "
          f"{norms[0] / (norms[-1] + 1e-12):.1e}x  -> distant steps barely learn.")


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    seqs, labels = make_memory_task(n=300, T=12)
    tr, te = slice(0, 240), slice(240, 300)

    net = RNNNumPy(2, 16, 2, lr=0.02, clip=5.0).fit(seqs[tr], labels[tr], epochs=40)
    print(f"NumPy RNN (clip=5) test acc = {np.mean(net.predict(seqs[te]) == labels[te]):.3f}")

    tnet = RNNTorch(2, 16, 2).fit(seqs[tr], labels[tr], epochs=60)
    print(f"Torch RNN          test acc = {np.mean(tnet.predict(seqs[te]) == labels[te]):.3f}")

    print()
    vanishing_demo()


if __name__ == "__main__":
    demo()

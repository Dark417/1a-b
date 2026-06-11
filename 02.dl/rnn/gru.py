"""
Gated Recurrent Unit (GRU)
==========================
A streamlined gated RNN cell (Cho et al., 2014) that uses just two gates — a
**reset** gate and an **update** gate — instead of the LSTM's three, and keeps a
single hidden state (no separate cell state). The update gate forms a
*gradient highway*: it linearly interpolates between the previous state and a
fresh candidate, so when it chooses to "carry", gradients flow back unchanged.
Fewer parameters than an LSTM, often comparable accuracy.

Variants implemented here:
    - GRU cell (reset + update gates) — full forward + BPTT by hand in NumPy
    - Idiomatic PyTorch GRU (`nn.GRU`) with gradient clipping
    - Comparison against the vanilla RNN / LSTM gradient paths (in the math)

Training techniques demonstrated:
    - Gated *convex-combination* memory as a vanishing-gradient remedy
      (canonical home: dl/rnn/rnn.py; gated fix also in dl/rnn/lstm.py)
    - Gradient clipping (global-norm) — see training-techniques/README.md
    - Seeding everything for reproducibility

References:
    - Cho et al. (2014), "Learning Phrase Representations using RNN
      Encoder-Decoder for Statistical Machine Translation"
    - Chung et al. (2014), "Empirical Evaluation of Gated Recurrent Neural
      Networks on Sequence Modeling"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — one GRU cell, unrolled, BPTT by hand
# ---------------------------------------------------------------------------
class GRUNumPy:
    r"""
    Gates (z_t = [x_t, h_{t-1}] concatenated for the gate pre-activations):
        r_t = σ(W_r [x_t, h_{t-1}] + b_r)      reset gate  — how much past to forget
        u_t = σ(W_u [x_t, h_{t-1}] + b_u)      update gate — how much past to keep
        n_t = tanh(W_n [x_t, (r_t ⊙ h_{t-1})] + b_n)   candidate state
    State (a CONVEX COMBINATION — this is the gradient highway):
        h_t = (1 - u_t) ⊙ n_t + u_t ⊙ h_{t-1}

    The carry path ∂h_t/∂h_{t-1} contains an explicit additive term diag(u_t):
    when the update gate stays near 1 the state is copied through and the
    gradient passes back essentially unattenuated — like the LSTM's constant
    error carousel, but with one state vector and two gates instead of three.

    We read out the LAST hidden state (many-to-one):  logits = h_T W_hy + b_y.
    """

    def __init__(self, in_dim, hidden, out_dim, lr=0.1, clip=5.0, seed=SEED):
        rng = np.random.default_rng(seed)
        Z = in_dim + hidden
        s = 1.0 / np.sqrt(Z)
        # Each gate sees [x_t, h_{t-1}] (or [x_t, r⊙h] for the candidate).
        self.W = {k: rng.normal(0, s, (Z, hidden)) for k in ("r", "u", "n")}
        self.b = {k: np.zeros(hidden) for k in ("r", "u", "n")}
        self.Why = rng.normal(0, 1 / np.sqrt(hidden), (hidden, out_dim))
        self.by = np.zeros(out_dim)
        self.H, self.in_dim, self.lr, self.clip = hidden, in_dim, lr, clip

    def forward(self, X):
        """X: (T, in_dim). Returns softmax over the last-step logits."""
        T = len(X)
        H = self.H
        self.cache = []
        h = np.zeros(H)
        for t in range(T):
            xz = np.concatenate([X[t], h])              # [x_t, h_{t-1}]
            r = sigmoid(xz @ self.W["r"] + self.b["r"])  # reset gate
            u = sigmoid(xz @ self.W["u"] + self.b["u"])  # update gate
            rn = np.concatenate([X[t], r * h])           # candidate sees reset state
            n = np.tanh(rn @ self.W["n"] + self.b["n"])  # candidate
            h_new = (1 - u) * n + u * h                  # convex combination
            self.cache.append((X[t], h, r, u, n, rn, h_new))
            h = h_new
        self.logits = h @ self.Why + self.by
        return softmax(self.logits)

    def backward(self, y):
        T = len(self.cache)
        H = self.H
        p = softmax(self.logits)
        p[y] -= 1.0                                      # dL/dlogits
        dW = {k: np.zeros_like(self.W[k]) for k in ("r", "u", "n")}
        db = {k: np.zeros_like(self.b[k]) for k in ("r", "u", "n")}
        h_last = self.cache[-1][6]
        dWhy = np.outer(h_last, p)
        dby = p.copy()
        dh = p @ self.Why.T                              # grad into last hidden state
        self.bptt_norms = []                             # ||dh|| over time
        for t in reversed(range(T)):
            x_t, h_prev, r, u, n, rn, h_new = self.cache[t]
            self.bptt_norms.append(np.linalg.norm(dh))
            # h_t = (1 - u) n + u h_prev
            dn = dh * (1 - u)
            du = dh * (h_prev - n)
            dh_prev = dh * u                             # <-- the gradient highway term
            # candidate n = tanh(W_n [x, r⊙h_prev])
            da_n = dn * (1 - n ** 2)                     # pre-activation grad
            dW["n"] += np.outer(rn, da_n)
            db["n"] += da_n
            drn = da_n @ self.W["n"].T                   # grad into [x, r⊙h_prev]
            d_rh = drn[self.in_dim:]                     # grad into (r ⊙ h_prev)
            dr = d_rh * h_prev
            dh_prev += d_rh * r                          # path through candidate
            # update gate u = σ(W_u [x, h_prev])
            da_u = du * u * (1 - u)
            xz = np.concatenate([x_t, h_prev])
            dW["u"] += np.outer(xz, da_u)
            db["u"] += da_u
            dh_prev += (da_u @ self.W["u"].T)[self.in_dim:]
            # reset gate r = σ(W_r [x, h_prev])
            da_r = dr * r * (1 - r)
            dW["r"] += np.outer(xz, da_r)
            db["r"] += da_r
            dh_prev += (da_r @ self.W["r"].T)[self.in_dim:]
            dh = dh_prev                                 # propagate one step back
        self.bptt_norms.reverse()
        grads = (dW, db, dWhy, dby)
        # global-norm gradient clipping
        flat = [g for d in (dW, db) for g in d.values()] + [dWhy, dby]
        total = np.sqrt(sum((g ** 2).sum() for g in flat))
        if self.clip is not None and total > self.clip:
            scale = self.clip / total
            for d in (dW, db):
                for k in d:
                    d[k] *= scale
            dWhy *= scale
            dby *= scale
        return grads

    def step(self, grads):
        dW, db, dWhy, dby = grads
        for k in ("r", "u", "n"):
            self.W[k] -= self.lr * dW[k]
            self.b[k] -= self.lr * db[k]
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby

    def fit(self, seqs, labels, epochs=40):
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
# 2. PyTorch implementation (idiomatic — nn.GRU + autograd)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device():
    """CUDA > MPS > CPU. See docs/gpu-setup.md."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GRUTorch(nn.Module):
    """Idiomatic GRU classifier (many-to-one) using PyTorch's fused `nn.GRU`."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):                                # x: (B, T, in_dim)
        out, _ = self.rnn(x)
        return self.head(out[:, -1])                     # last time-step

    def fit(self, seqs, labels, epochs=60, lr=0.01, clip=5.0):
        dev = get_device()
        self.to(dev)
        X = torch.as_tensor(np.array(seqs), dtype=torch.float32, device=dev)
        y = torch.as_tensor(labels, dtype=torch.long, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.history = []
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), clip)   # gradient clipping
            opt.step()
            self.history.append(loss.item())
        return self

    @torch.no_grad()
    def predict(self, seqs):
        dev = next(self.parameters()).device
        X = torch.as_tensor(np.array(seqs), dtype=torch.float32, device=dev)
        return self(X).argmax(1).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo — the long-memory toy task (same as rnn.py / lstm.py)
# ---------------------------------------------------------------------------
def make_memory_task(n=300, T=25, seed=SEED):
    """Label = identity of the FIRST token; the rest is noise. Needs long memory.

    Copied locally so this module is self-contained (no sibling imports).
    """
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for _ in range(n):
        first = rng.integers(0, 2)
        x = rng.normal(0, 0.3, size=(T, 2))
        x[0, first] += 2.0                               # signal only at t=0
        seqs.append(x)
        labels.append(first)
    return np.array(seqs), np.array(labels)


def seed_everything(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # On this CPU box, small RNN ops suffer heavy thread-oversubscription overhead
    # (multi-thread BLAS on tiny matmuls is far slower); one thread keeps the demo
    # well under the time budget. Harmless for these toy-sized tensors.
    torch.set_num_threads(1)


def demo():
    # Kept deliberately small (short sequences, few epochs) so the whole demo
    # runs in well under 30 s on CPU — PyTorch's CPU GRU has no cuDNN fast path.
    seed_everything(SEED)
    seqs, labels = make_memory_task(n=160, T=12)
    tr, te = slice(0, 120), slice(120, 160)

    net = GRUNumPy(2, 16, 2, lr=0.2).fit(seqs[tr], labels[tr], epochs=40)
    acc_np = np.mean(net.predict(seqs[te]) == labels[te])
    print(f"NumPy GRU   test acc = {acc_np:.3f}")

    m = GRUTorch(2, 16, 2).fit(seqs[tr], labels[tr], epochs=40)
    acc_pt = np.mean(m.predict(seqs[te]) == labels[te])
    print(f"Torch GRU   test acc = {acc_pt:.3f}")

    print("\nThe update gate u_t lets the GRU COPY the hidden state across the steps")
    print("(h_t = (1-u)·n + u·h_{t-1}); the carry term diag(u_t) is the gradient")
    print("highway — like the LSTM cell state, but with two gates and one state.")


if __name__ == "__main__":
    demo()

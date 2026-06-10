"""
Sequence-to-Sequence (Encoder–Decoder) with Teacher Forcing & Attention
========================================================================
The canonical architecture for mapping one sequence to another of possibly
different length (translation, summarization, copy/reverse): an **encoder** RNN
compresses the source into a context, and a **decoder** RNN generates the target
one token at a time, conditioned on that context and its own previous outputs.

This module is the tutorial **home for teacher forcing** — during training the
decoder is fed the *ground-truth* previous token instead of its own (possibly
wrong) prediction, which stabilizes and speeds up learning. It also adds an
**attention** option (Bahdanau additive / Luong dot-product) so the decoder can
look back at *all* encoder states instead of squeezing everything through a
single fixed-size context vector (the information-bottleneck problem).

Variants implemented here:
    - Compact NumPy GRU encoder–decoder with manual BPTT and teacher forcing
      (no attention — the fixed-context baseline)
    - Idiomatic PyTorch encoder–decoder (GRU or LSTM cell) with:
        * teacher forcing (with a tunable teacher-forcing ratio)
        * NO attention  -> single context vector
        * Bahdanau (additive) attention
        * Luong (dot-product) attention
    - Greedy decoding at inference

Training techniques demonstrated:
    - TEACHER FORCING (canonical home; also referenced from dl/rnn/lstm.py)
    - Gradient clipping (global-norm) — see training-techniques/README.md
    - Seeding everything for reproducibility

References:
    - Sutskever, Vinyals & Le (2014), "Sequence to Sequence Learning with NNs"
    - Cho et al. (2014), encoder–decoder + GRU
    - Bahdanau, Cho & Bengio (2015), "Neural MT by Jointly Learning to Align and Translate"
    - Luong, Pham & Manning (2015), "Effective Approaches to Attention-based NMT"
"""

from __future__ import annotations

import numpy as np

SEED = 0

# Special tokens (shared by both implementations).
PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# Toy task: REVERSE a short integer sequence. Self-contained (no sibling imports).
# ---------------------------------------------------------------------------
def make_reverse_task(n=400, vocab=6, length=5, seed=SEED):
    """Source = random symbols; target = the source reversed.

    Symbols are drawn from [N_SPECIAL, N_SPECIAL+vocab).  The decoder target is
    the reversed source wrapped as  BOS r1 r2 ... rL EOS.
    Returns (src, tgt) integer arrays; src: (n, length), tgt: (n, length+2).
    """
    rng = np.random.default_rng(seed)
    src = rng.integers(N_SPECIAL, N_SPECIAL + vocab, size=(n, length))
    rev = src[:, ::-1]
    bos = np.full((n, 1), BOS)
    eos = np.full((n, 1), EOS)
    tgt = np.concatenate([bos, rev, eos], axis=1)
    return src, tgt


def seed_everything(seed=SEED):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — compact GRU encoder–decoder, manual BPTT,
#    teacher forcing, single fixed context vector (no attention).
# ---------------------------------------------------------------------------
class Seq2SeqNumPy:
    r"""
    Encoder GRU reads the source x_{1:S} -> final state  c = h^enc_S  (the context).
    Decoder GRU is initialized with  h^dec_0 = c  and, at each step, consumes the
    *previous target token* (teacher forcing) plus emits a distribution over the
    vocabulary:
        h^dec_t = GRU(emb(y_{t-1}), h^dec_{t-1})
        logits_t = h^dec_t W_o + b_o,    p_t = softmax(logits_t)

    Conditional factorization being modeled:
        p(y_{1:T} | x_{1:S}) = ∏_t p(y_t | y_{<t}, x_{1:S}).
    Teacher forcing replaces the sampled y_{<t} with the ground truth during
    training, turning the product into independent per-step classification.

    Both GRUs share the minimal-GRU update (gates r, z; candidate n):
        r = σ(W_r[e,h]),  z = σ(W_z[e,h]),  n = tanh(W_n[e, r⊙h]),
        h' = (1-z)⊙n + z⊙h.
    BPTT is done by hand for the whole decoder→encoder chain.
    """

    def __init__(self, vocab, emb=16, hidden=24, lr=0.1, clip=5.0, seed=SEED):
        rng = np.random.default_rng(seed)
        V = N_SPECIAL + vocab
        self.V, self.E, self.H = V, emb, hidden
        self.lr, self.clip = lr, clip
        self.Emb = rng.normal(0, 0.1, (V, emb))               # token embeddings
        # one GRU's parameters: input is [emb, hidden] -> hidden
        Zi = emb + hidden

        def gru_params(tag):
            s = 1.0 / np.sqrt(Zi)
            return {k: rng.normal(0, s, (Zi, hidden)) for k in ("r", "z", "n")}, \
                   {k: np.zeros(hidden) for k in ("r", "z", "n")}

        self.encW, self.encb = gru_params("enc")
        self.decW, self.decb = gru_params("dec")
        self.Wo = rng.normal(0, 1 / np.sqrt(hidden), (hidden, V))
        self.bo = np.zeros(V)

    # --- one GRU step (returns new h plus cache for backprop) ---
    def _gru_step(self, W, b, e, h):
        eh = np.concatenate([e, h])
        r = sigmoid(eh @ W["r"] + b["r"])
        z = sigmoid(eh @ W["z"] + b["z"])
        en = np.concatenate([e, r * h])
        n = np.tanh(en @ W["n"] + b["n"])
        h_new = (1 - z) * n + z * h
        return h_new, (e, h, r, z, n, en)

    def forward(self, src, tgt_in):
        """src: (S,) ints; tgt_in: (Tdec,) ints (BOS + reversed, no final EOS)."""
        self.src, self.tgt_in = src, tgt_in
        # ---- encode ----
        h = np.zeros(self.H)
        self.enc_cache = []
        for t in range(len(src)):
            e = self.Emb[src[t]]
            h, c = self._gru_step(self.encW, self.encb, e, h)
            self.enc_cache.append(c)
        self.context = h.copy()                               # fixed context vector
        # ---- decode (teacher forcing: feed ground-truth tgt_in) ----
        self.dec_cache = []
        self.logits = []
        for t in range(len(tgt_in)):
            e = self.Emb[tgt_in[t]]
            h, c = self._gru_step(self.decW, self.decb, e, h)
            self.dec_cache.append((c, h))
            self.logits.append(h @ self.Wo + self.bo)
        self.logits = np.array(self.logits)                   # (Tdec, V)
        return softmax(self.logits)

    def _gru_backward(self, W, b, cache, dh_out, dW, db):
        """Backprop one GRU step. Returns (d_emb, d_h_prev)."""
        e, h_prev, r, z, n, en = cache
        emb_dim = self.E
        dn = dh_out * (1 - z)
        dz = dh_out * (h_prev - n)
        dh_prev = dh_out * z                                  # carry term diag(z)
        da_n = dn * (1 - n ** 2)
        dW["n"] += np.outer(en, da_n)
        db["n"] += da_n
        den = da_n @ W["n"].T
        de = den[:emb_dim].copy()
        d_rh = den[emb_dim:]
        dr = d_rh * h_prev
        dh_prev = dh_prev + d_rh * r
        eh = np.concatenate([e, h_prev])
        da_z = dz * z * (1 - z)
        dW["z"] += np.outer(eh, da_z)
        db["z"] += da_z
        dez = da_z @ W["z"].T
        de += dez[:emb_dim]
        dh_prev = dh_prev + dez[emb_dim:]
        da_r = dr * r * (1 - r)
        dW["r"] += np.outer(eh, da_r)
        db["r"] += da_r
        der = da_r @ W["r"].T
        de += der[:emb_dim]
        dh_prev = dh_prev + der[emb_dim:]
        return de, dh_prev

    def backward(self, tgt_out):
        """tgt_out: (Tdec,) ground-truth next tokens (reversed + EOS)."""
        T = len(self.dec_cache)
        p = softmax(self.logits)
        dlogits = p.copy()
        dlogits[np.arange(T), tgt_out] -= 1.0                 # softmax-CE grad
        dWo = self.dec_cache_h().T @ dlogits
        dbo = dlogits.sum(0)
        dEmb = np.zeros_like(self.Emb)
        decdW = {k: np.zeros_like(self.decW[k]) for k in ("r", "z", "n")}
        decdb = {k: np.zeros_like(self.decb[k]) for k in ("r", "z", "n")}
        encdW = {k: np.zeros_like(self.encW[k]) for k in ("r", "z", "n")}
        encdb = {k: np.zeros_like(self.encb[k]) for k in ("r", "z", "n")}
        # ---- decoder BPTT ----
        dh = np.zeros(self.H)
        for t in reversed(range(T)):
            cache, h = self.dec_cache[t]
            dh_t = dlogits[t] @ self.Wo.T + dh
            de, dh = self._gru_backward(self.decW, self.decb, cache, dh_t, decdW, decdb)
            dEmb[self.tgt_in[t]] += de
        # gradient flows from decoder's initial state into the encoder's last state
        d_context = dh
        # ---- encoder BPTT ----
        dh = d_context
        for t in reversed(range(len(self.src))):
            cache = self.enc_cache[t]
            de, dh = self._gru_backward(self.encW, self.encb, cache, dh, encdW, encdb)
            dEmb[self.src[t]] += de
        grads = (dEmb, encdW, encdb, decdW, decdb, dWo, dbo)
        self._clip(grads)
        return grads

    def dec_cache_h(self):
        return np.array([h for (_c, h) in self.dec_cache])    # (Tdec, H)

    def _clip(self, grads):
        dEmb, encdW, encdb, decdW, decdb, dWo, dbo = grads
        flat = [dEmb, dWo, dbo]
        for d in (encdW, encdb, decdW, decdb):
            flat += list(d.values())
        total = np.sqrt(sum((g ** 2).sum() for g in flat))
        if self.clip is not None and total > self.clip:
            s = self.clip / total
            for g in (dEmb, dWo, dbo):
                g *= s
            for d in (encdW, encdb, decdW, decdb):
                for k in d:
                    d[k] *= s

    def step(self, grads):
        dEmb, encdW, encdb, decdW, decdb, dWo, dbo = grads
        self.Emb -= self.lr * dEmb
        self.Wo -= self.lr * dWo
        self.bo -= self.lr * dbo
        for k in ("r", "z", "n"):
            self.encW[k] -= self.lr * encdW[k]
            self.encb[k] -= self.lr * encdb[k]
            self.decW[k] -= self.lr * decdW[k]
            self.decb[k] -= self.lr * decdb[k]

    def fit(self, src, tgt, epochs=30):
        self.history = []
        for _ in range(epochs):
            loss = 0.0
            for s, full in zip(src, tgt):
                tgt_in = full[:-1]                            # BOS + reversed
                tgt_out = full[1:]                            # reversed + EOS
                p = self.forward(s, tgt_in)
                loss += -np.log(p[np.arange(len(tgt_out)), tgt_out] + 1e-12).mean()
                self.step(self.backward(tgt_out))
            self.history.append(loss / len(src))
        return self

    def greedy_decode(self, src, max_len=12):
        """Autoregressive (no teacher forcing): feed the model's own outputs."""
        h = np.zeros(self.H)
        for t in range(len(src)):
            h, _ = self._gru_step(self.encW, self.encb, self.Emb[src[t]], h)
        out, tok = [], BOS
        for _ in range(max_len):
            h, _ = self._gru_step(self.decW, self.decb, self.Emb[tok], h)
            tok = int((h @ self.Wo + self.bo).argmax())
            if tok == EOS:
                break
            out.append(tok)
        return out

    def sequence_accuracy(self, src, tgt):
        correct = 0
        for s, full in zip(src, tgt):
            gold = [int(x) for x in full[1:-1]]               # strip BOS/EOS
            correct += int(self.greedy_decode(s) == gold)
        return correct / len(src)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — encoder–decoder with teacher forcing + attention
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    """CUDA > MPS > CPU. See docs/gpu-setup.md."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Attention(nn.Module):
    r"""
    Bahdanau (additive):  score(h_t, h^enc_s) = v^T tanh(W_d h_t + W_e h^enc_s).
    Luong  (dot):         score(h_t, h^enc_s) = h_t^T h^enc_s.
    Weights  α_{t,s} = softmax_s(score);  context_t = Σ_s α_{t,s} h^enc_s.
    """

    def __init__(self, hidden, kind="bahdanau"):
        super().__init__()
        self.kind = kind
        if kind == "bahdanau":
            self.Wd = nn.Linear(hidden, hidden, bias=False)
            self.We = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, dec_h, enc_out, mask=None):
        # dec_h: (B, H); enc_out: (B, S, H)
        if self.kind == "bahdanau":
            score = self.v(torch.tanh(self.Wd(dec_h).unsqueeze(1) + self.We(enc_out))).squeeze(-1)
        else:  # luong dot
            score = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(~mask, float("-inf"))
        attn = F.softmax(score, dim=1)                        # (B, S)
        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # (B, H)
        return context, attn


class Seq2SeqTorch(nn.Module):
    """GRU/LSTM encoder–decoder. attention in {None, 'bahdanau', 'luong'}."""

    def __init__(self, vocab, emb=24, hidden=32, cell="gru", attention=None):
        super().__init__()
        V = N_SPECIAL + vocab
        self.V, self.H, self.cell = V, hidden, cell
        self.attention_kind = attention
        self.emb = nn.Embedding(V, emb, padding_idx=PAD)
        Cell = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.encoder = Cell(emb, hidden, batch_first=True)
        # decoder input = token embedding (+ context if attention)
        dec_in = emb + (hidden if attention else 0)
        self.decoder_cell = {"gru": nn.GRUCell, "lstm": nn.LSTMCell}[cell](dec_in, hidden)
        self.attn = Attention(hidden, attention) if attention else None
        out_in = hidden + (hidden if attention else 0)
        self.out = nn.Linear(out_in, V)

    def _enc_to_dec_state(self, hn):
        # hn from nn.GRU/LSTM is (1, B, H) (or tuple for LSTM); return per-cell state.
        if self.cell == "lstm":
            h, c = hn
            return (h.squeeze(0), c.squeeze(0))
        return hn.squeeze(0)

    def forward(self, src, tgt_in, teacher_forcing=1.0):
        """src: (B,S); tgt_in: (B,Tdec). Returns logits (B,Tdec,V)."""
        B, Tdec = tgt_in.shape
        enc_out, hn = self.encoder(self.emb(src))             # enc_out: (B,S,H)
        state = self._enc_to_dec_state(hn)
        mask = (src != PAD)
        logits = []
        prev = tgt_in[:, 0]                                   # BOS
        for t in range(Tdec):
            e = self.emb(prev)                                # (B, emb)
            h_for_attn = state[0] if self.cell == "lstm" else state
            if self.attn is not None:
                context, _ = self.attn(h_for_attn, enc_out, mask)
                dec_in = torch.cat([e, context], dim=1)
            else:
                dec_in = e
            state = self.decoder_cell(dec_in, state)
            h_t = state[0] if self.cell == "lstm" else state
            feat = torch.cat([h_t, context], dim=1) if self.attn is not None else h_t
            step_logits = self.out(feat)                      # (B, V)
            logits.append(step_logits)
            # teacher forcing: next input is gold token (prob tf) else own prediction
            if t + 1 < Tdec:
                use_tf = torch.rand(1, device=src.device).item() < teacher_forcing
                prev = tgt_in[:, t + 1] if use_tf else step_logits.argmax(1)
        return torch.stack(logits, dim=1)                     # (B, Tdec, V)

    def fit(self, src, tgt, epochs=40, lr=0.01, clip=5.0, teacher_forcing=0.9, batch=64):
        dev = get_device()
        self.to(dev)
        src = torch.as_tensor(np.asarray(src), dtype=torch.long, device=dev)
        tgt = torch.as_tensor(np.asarray(tgt), dtype=torch.long, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        n = src.shape[0]
        self.history = []
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            ep_loss = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                s, full = src[idx], tgt[idx]
                tgt_in, tgt_out = full[:, :-1], full[:, 1:]
                logits = self(s, tgt_in, teacher_forcing=teacher_forcing)
                loss = loss_fn(logits.reshape(-1, self.V), tgt_out.reshape(-1))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), clip)  # gradient clipping
                opt.step()
                ep_loss += loss.item() * len(idx)
            self.history.append(ep_loss / n)
        return self

    @torch.no_grad()
    def greedy_decode(self, src, max_len=12):
        """Inference: NO teacher forcing — feed the model's own predictions."""
        dev = next(self.parameters()).device
        src = torch.as_tensor(np.asarray(src), dtype=torch.long, device=dev)
        if src.dim() == 1:
            src = src.unsqueeze(0)
        B = src.shape[0]
        enc_out, hn = self.encoder(self.emb(src))
        state = self._enc_to_dec_state(hn)
        mask = (src != PAD)
        prev = torch.full((B,), BOS, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        outs = []
        for _ in range(max_len):
            e = self.emb(prev)
            h_for_attn = state[0] if self.cell == "lstm" else state
            if self.attn is not None:
                context, _ = self.attn(h_for_attn, enc_out, mask)
                dec_in = torch.cat([e, context], dim=1)
            else:
                dec_in = e
            state = self.decoder_cell(dec_in, state)
            h_t = state[0] if self.cell == "lstm" else state
            feat = torch.cat([h_t, context], dim=1) if self.attn is not None else h_t
            prev = self.out(feat).argmax(1)
            outs.append(prev.clone())
            done = done | (prev == EOS)
            if bool(done.all()):
                break
        seqs = torch.stack(outs, dim=1).cpu().numpy()         # (B, L)
        result = []
        for row in seqs:
            seq = []
            for tok in row:
                if tok == EOS:
                    break
                seq.append(int(tok))
            result.append(seq)
        return result

    def sequence_accuracy(self, src, tgt):
        preds = self.greedy_decode(src)
        correct = 0
        for pred, full in zip(preds, np.asarray(tgt)):
            gold = [int(x) for x in full[1:-1]]
            correct += int(pred == gold)
        return correct / len(preds)


# ---------------------------------------------------------------------------
# 3. Demo — tiny REVERSE task; compare NumPy, plain-context, and attention.
# ---------------------------------------------------------------------------
def demo():
    seed_everything(SEED)
    vocab, length = 6, 4
    src, tgt = make_reverse_task(n=400, vocab=vocab, length=length)
    tr = slice(0, 320)
    te = slice(320, 400)

    # NumPy encoder-decoder (compact, fixed context, teacher forcing).
    np_model = Seq2SeqNumPy(vocab, emb=16, hidden=24, lr=0.1)
    np_model.fit(src[tr], tgt[tr], epochs=25)
    print(f"NumPy seq2seq (no attn)   exact-match acc = "
          f"{np_model.sequence_accuracy(src[te], tgt[te]):.3f}")

    # PyTorch: no attention vs Bahdanau vs Luong.
    for attn in (None, "bahdanau", "luong"):
        seed_everything(SEED)
        m = Seq2SeqTorch(vocab, emb=24, hidden=32, cell="gru", attention=attn)
        m.fit(src[tr], tgt[tr], epochs=40, teacher_forcing=0.9)
        name = attn if attn else "no-attn"
        print(f"Torch seq2seq ({name:9s}) exact-match acc = "
              f"{m.sequence_accuracy(src[te], tgt[te]):.3f}")

    print("\nTeacher forcing feeds the GOLD previous token while training, turning")
    print("p(y_{1:T}|x)=∏_t p(y_t|y_{<t},x) into per-step classification; at")
    print("inference we decode greedily on the model's OWN outputs. Attention lets")
    print("the decoder read every encoder state instead of one fixed context vector.")


if __name__ == "__main__":
    demo()

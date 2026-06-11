"""
Sequence-to-Sequence with Attention
====================================
A plain encoder-decoder squeezes the entire source sequence into one fixed
vector — a bottleneck that hurts long inputs. **Attention** lets the decoder
look back at *all* encoder states and build, at each output step, a weighted
"context" focused on the relevant source positions. We implement an RNN
encoder-decoder with **both** classic attention variants:

  - **Bahdanau (additive)** attention: a small MLP scores each encoder state
    against the previous decoder state.
  - **Luong (multiplicative / dot)** attention: a bilinear/dot-product score,
    using the *current* decoder state.

We train with **teacher forcing** on a tiny **reverse** task (output = input
reversed), which is impossible without attention but easy with it.

Variants implemented here:
    - Bahdanau additive attention vs Luong general/dot attention (a flag)
    - GRU encoder/decoder, teacher forcing, greedy decoding
    - A from-scratch NumPy attention-scoring reference

Training techniques demonstrated:
    - TEACHER FORCING (see training-techniques/README.md)
    - Attention as a remedy for the fixed-vector bottleneck

References:
    - Bahdanau, Cho & Bengio (2015), "Neural Machine Translation by Jointly
      Learning to Align and Translate"
    - Luong, Pham & Manning (2015), "Effective Approaches to Attention-based NMT"
"""

from __future__ import annotations

import numpy as np

SEED = 0

# Special tokens for the toy task
PAD, BOS, EOS = 0, 1, 2
N_SYMBOLS = 8          # symbols 3..3+N_SYMBOLS-1 are the "content" tokens
VOCAB = 3 + N_SYMBOLS


# ---------------------------------------------------------------------------
# 1. NumPy reference — the attention scoring math, made explicit
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def attention_numpy(dec_state, enc_states, mode="dot", W=None, v=None):
    r"""
    Compute one attention step from scratch.

    Inputs:
        dec_state  : (d,)         the query (decoder hidden state)
        enc_states : (T_src, d)   the keys/values (encoder hidden states)

    Alignment scores e_t (how well source position t matches the query):
        - "dot"      (Luong):  e_t = h_dec · h_t
        - "general"  (Luong):  e_t = h_dec^T W h_t
        - "additive" (Bahdanau): e_t = v^T tanh(W [h_dec ; h_t])
    Attention weights:  a = softmax(e)              (sum to 1 over source)
    Context vector:     c = Σ_t a_t h_t             (weighted sum of values)
    Returns (context, weights).
    """
    if mode == "dot":
        scores = enc_states @ dec_state                       # (T_src,)
    elif mode == "general":
        scores = enc_states @ (W @ dec_state)                 # bilinear
    elif mode == "additive":
        # v^T tanh(W [h_dec ; h_t]) for each t
        cat = np.concatenate([np.broadcast_to(dec_state, enc_states.shape),
                              enc_states], axis=1)            # (T_src, 2d)
        scores = np.tanh(cat @ W.T) @ v                       # (T_src,)
    else:
        raise ValueError(mode)
    weights = softmax(scores)
    context = weights @ enc_states                            # (d,)
    return context, weights


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — encoder-decoder with attention
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)  # keep tiny CPU RNNs fast & deterministic here


def get_device():
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Attention(nn.Module):
    """Bahdanau (additive) or Luong (dot / general) attention."""

    def __init__(self, hidden, mode="bahdanau"):
        super().__init__()
        self.mode = mode
        if mode == "bahdanau":
            self.W = nn.Linear(2 * hidden, hidden)
            self.v = nn.Linear(hidden, 1, bias=False)
        elif mode == "luong_general":
            self.W = nn.Linear(hidden, hidden, bias=False)
        elif mode == "luong_dot":
            pass
        else:
            raise ValueError(mode)

    def forward(self, dec_h, enc_outs):
        # dec_h: (B, H)   enc_outs: (B, T, H)
        B, T, H = enc_outs.shape
        if self.mode == "bahdanau":
            q = dec_h.unsqueeze(1).expand(B, T, H)            # (B,T,H)
            energy = torch.tanh(self.W(torch.cat([q, enc_outs], -1)))
            scores = self.v(energy).squeeze(-1)               # (B,T)
        elif self.mode == "luong_general":
            scores = torch.bmm(self.W(enc_outs), dec_h.unsqueeze(-1)).squeeze(-1)
        else:  # luong_dot
            scores = torch.bmm(enc_outs, dec_h.unsqueeze(-1)).squeeze(-1)
        weights = F.softmax(scores, dim=-1)                   # (B,T)
        context = torch.bmm(weights.unsqueeze(1), enc_outs).squeeze(1)  # (B,H)
        return context, weights


class Encoder(nn.Module):
    def __init__(self, vocab, emb=32, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hidden, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outs, h = self.gru(e)            # outs:(B,T,H)  h:(1,B,H)
        return outs, h


class Decoder(nn.Module):
    def __init__(self, vocab, emb=32, hidden=64, attn_mode="bahdanau"):
        super().__init__()
        self.embed = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.attn = Attention(hidden, attn_mode)
        self.gru = nn.GRU(emb + hidden, hidden, batch_first=True)
        self.out = nn.Linear(2 * hidden, vocab)   # combine GRU state + context

    def step(self, tok, h, enc_outs):
        # one decoding step: tok (B,), h (1,B,H)
        e = self.embed(tok).unsqueeze(1)                      # (B,1,emb)
        context, weights = self.attn(h[-1], enc_outs)         # (B,H),(B,T)
        gru_in = torch.cat([e, context.unsqueeze(1)], -1)     # feed context in
        out, h = self.gru(gru_in, h)                          # (B,1,H)
        logits = self.out(torch.cat([out.squeeze(1), context], -1))
        return logits, h, weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab=VOCAB, emb=32, hidden=64, attn_mode="bahdanau"):
        super().__init__()
        self.enc = Encoder(vocab, emb, hidden)
        self.dec = Decoder(vocab, emb, hidden, attn_mode)
        self.vocab = vocab

    def forward(self, src, tgt, teacher_forcing=0.5):
        """tgt includes BOS ... EOS. Returns logits (B, T_out, vocab)."""
        dev = src.device
        enc_outs, h = self.enc(src)
        B, T = tgt.shape
        logits = []
        tok = tgt[:, 0]                                       # BOS
        for t in range(1, T):
            lg, h, _ = self.dec.step(tok, h, enc_outs)
            logits.append(lg)
            use_tf = torch.rand(1, device=dev).item() < teacher_forcing
            tok = tgt[:, t] if use_tf else lg.argmax(-1)      # teacher forcing
        return torch.stack(logits, 1)                         # (B,T-1,vocab)

    @torch.no_grad()
    def greedy_decode(self, src, max_len=12):
        dev = src.device
        enc_outs, h = self.enc(src)
        B = src.size(0)
        tok = torch.full((B,), BOS, dtype=torch.long, device=dev)
        outs, attns = [], []
        for _ in range(max_len):
            lg, h, w = self.dec.step(tok, h, enc_outs)
            tok = lg.argmax(-1)
            outs.append(tok); attns.append(w)
            if (tok == EOS).all():
                break
        return torch.stack(outs, 1), torch.stack(attns, 1)    # tokens, (B,T_out,T_src)


# ---------------------------------------------------------------------------
# 3. Toy task + training
# ---------------------------------------------------------------------------
def make_reverse_data(n=512, min_len=4, max_len=8, seed=SEED):
    """Source = random symbols; target = the source REVERSED. Needs alignment."""
    rng = np.random.default_rng(seed)
    src, tgt = [], []
    for _ in range(n):
        L = rng.integers(min_len, max_len + 1)
        s = rng.integers(3, VOCAB, size=L).tolist()
        src.append(s + [EOS])
        tgt.append([BOS] + s[::-1] + [EOS])         # reversed content
    return src, tgt


def _pad(batch, length):
    return [seq + [PAD] * (length - len(seq)) for seq in batch]


def train(model, src, tgt, epochs=40, lr=0.01, batch=64, teacher_forcing=0.5):
    dev = get_device(); model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    smax = max(len(s) for s in src); tmax = max(len(t) for t in tgt)
    S = torch.tensor(_pad(src, smax), device=dev)
    T = torch.tensor(_pad(tgt, tmax), device=dev)
    N = len(src); model.history = []
    rng = np.random.default_rng(SEED)
    for _ in range(epochs):
        perm = rng.permutation(N); total = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]
            sb, tb = S[idx], T[idx]
            logits = model(sb, tb, teacher_forcing)
            loss = loss_fn(logits.reshape(-1, model.vocab), tb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); total += loss.item()
        model.history.append(total / (N // batch + 1))
    return model


def sequence_accuracy(model, src, tgt):
    dev = next(model.parameters()).device
    smax = max(len(s) for s in src)
    S = torch.tensor(_pad(src, smax), device=dev)
    preds, _ = model.greedy_decode(S, max_len=max(len(t) for t in tgt))
    correct = 0
    for p, t in zip(preds.cpu().tolist(), tgt):
        gold = t[1:]                                  # strip BOS
        gold = gold[:gold.index(EOS) + 1] if EOS in gold else gold
        pred = p[:p.index(EOS) + 1] if EOS in p else p
        correct += (pred == gold)
    return correct / len(src)


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)

    # --- NumPy attention sanity: weights focus on the matching position ---
    # Distinct (near-orthogonal) encoder states; the query equals state #2, so
    # the dot-product attention should put almost all its mass on position 2.
    enc = np.eye(5, 4) * 3.0                           # 5 distinct source states
    q = enc[2].copy()                                  # query == encoder state #2
    _, w = attention_numpy(q, enc, "dot")
    print(f"NumPy dot      attention argmax = {w.argmax()} (query == pos 2), "
          f"weights={np.round(w, 2)}")
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.5, (4, 8)); v = rng.normal(0, 0.5, 4)
    _, w = attention_numpy(q, enc, "additive", W=W, v=v)
    print(f"NumPy additive attention weights = {np.round(w, 2)} (a distribution "
          f"over source positions)")

    # --- Train both attention variants on the reverse task ---
    src, tgt = make_reverse_data(n=512)
    tr = slice(0, 448); te = slice(448, 512)
    models = {}
    for mode in ("bahdanau", "luong_dot"):
        torch.manual_seed(SEED)
        m = Seq2SeqAttention(attn_mode=mode)
        train(m, src[tr], tgt[tr], epochs=40)
        acc = sequence_accuracy(m, src[te], tgt[te])
        models[mode] = m
        print(f"{mode:11s} reverse-task exact-sequence acc = {acc:.3f}  "
              f"(final loss {m.history[-1]:.3f})")

    # --- Show one decoded example with its attention alignment ---
    m = models["bahdanau"]
    dev = next(m.parameters()).device
    s = src[te][0]
    smax = max(len(x) for x in src[tr])               # pad as during training
    S = torch.tensor(_pad([s], smax), device=dev)
    pred, attn = m.greedy_decode(S, max_len=len(s) + 2)
    out = pred[0].cpu().tolist()
    out = out[:out.index(EOS) + 1] if EOS in out else out
    print(f"\nexample  src={s}  ->  pred={out}   (reverse of the content tokens)")
    print("attention is roughly anti-diagonal: output position i attends to "
          "source position (len-1-i).")


if __name__ == "__main__":
    demo()

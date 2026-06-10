"""
The Transformer ("Attention Is All You Need")
=============================================
A full encoder-decoder built only from attention + feed-forward layers, with
residual connections, layer normalization, positional encodings, and the noam
warmup schedule. The architecture behind BERT, GPT, T5, and modern LLMs.

Components implemented here:
    - Sinusoidal positional encoding
    - Multi-head self/cross attention (PyTorch)
    - Position-wise feed-forward, residual + LayerNorm (pre-norm)
    - Encoder & decoder stacks; greedy decoding
    - The noam learning-rate warmup schedule

Training techniques demonstrated:
    - RESIDUAL CONNECTIONS + LAYER NORM (stable very-deep training)
    - LR WARMUP / scheduling (noam)  — see training-techniques/README.md
    - Dropout; causal masking; teacher forcing

References:
    - Vaswani et al. (2017)
"""

from __future__ import annotations

import math
import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy: positional encoding (the one piece with crisp closed-form math)
# ---------------------------------------------------------------------------
def positional_encoding(L, d_model):
    r"""
    PE[pos, 2i]   = sin(pos / 10000^{2i/d})
    PE[pos, 2i+1] = cos(pos / 10000^{2i/d})
    Each dimension is a sinusoid of a different wavelength; relative positions are
    then linear functions of these, so attention can learn to shift by an offset.
    """
    pos = np.arange(L)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((L, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — a compact but complete Transformer
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.register_buffer("pe", torch.tensor(
            positional_encoding(max_len, d_model), dtype=torch.float32))

    def forward(self, x):                       # x: (B, L, d)
        return x + self.pe[:x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        def split(x, lin): return lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
        Q, K, V = (split(t, lin) for t, lin in zip((q, k, v), self.qkv))
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.drop(scores.softmax(-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x): return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):            # pre-norm + residual
        x = x + self.drop(self.attn(self.n1(x), self.n1(x), self.n1(x), mask))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model); self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask=None, src_mask=None):
        x = x + self.drop(self.self_attn(self.n1(x), self.n1(x), self.n1(x), tgt_mask))
        nx = self.n2(x)
        x = x + self.drop(self.cross_attn(nx, mem, mem, src_mask))   # attend to encoder
        x = x + self.drop(self.ff(self.n3(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, n_heads=4,
                 d_ff=128, n_layers=2, dropout=0.1, max_len=64):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.enc = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout)
                                 for _ in range(n_layers))
        self.dec = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout)
                                 for _ in range(n_layers))
        self.fc = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def encode(self, src, src_mask=None):
        x = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.enc:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, mem, tgt_mask=None, src_mask=None):
        x = self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.dec:
            x = layer(x, mem, tgt_mask, src_mask)
        return self.fc(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.decode(tgt, self.encode(src, src_mask), tgt_mask, src_mask)


def causal_mask(L, device):
    return torch.tril(torch.ones(L, L, device=device)).bool()    # (L,L) lower-tri


def noam_lr(step, d_model, warmup):
    """LR ∝ d_model^-0.5 * min(step^-0.5, step*warmup^-1.5). Warmup then decay."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


# ---------------------------------------------------------------------------
# 3. Demo — copy/reverse task: learn to OUTPUT THE REVERSED input sequence
# ---------------------------------------------------------------------------
def make_reverse_data(n, L, vocab, seed=SEED):
    """src = random tokens; tgt = src reversed. Tokens 0=PAD,1=BOS,2=EOS."""
    rng = np.random.default_rng(seed)
    src = rng.integers(3, vocab, size=(n, L))
    tgt_out = src[:, ::-1]
    bos = np.full((n, 1), 1); eos = np.full((n, 1), 2)
    tgt_in = np.concatenate([bos, tgt_out], 1)          # teacher forcing input
    tgt_out = np.concatenate([tgt_out, eos], 1)         # shifted target
    return src, tgt_in, tgt_out


def demo():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = get_device()
    V, L = 14, 6
    src, tin, tout = make_reverse_data(2000, L, V)
    src = torch.tensor(src, device=dev)
    tin = torch.tensor(tin, device=dev)
    tout = torch.tensor(tout, device=dev)

    model = Transformer(V, V, d_model=64, n_heads=4, n_layers=2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for step in range(1, 1001):
        for g in opt.param_groups:               # noam schedule (warmup + decay)
            g["lr"] = noam_lr(step, 64, warmup=300)
        tmask = causal_mask(tin.size(1), dev)
        logits = model(src, tin, tgt_mask=tmask)
        loss = loss_fn(logits.reshape(-1, V), tout.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}  lr {opt.param_groups[0]['lr']:.4f}")

    # greedy decode one example
    model.eval()
    with torch.no_grad():
        s = src[:1]
        mem = model.encode(s)
        ys = torch.tensor([[1]], device=dev)                     # BOS
        for _ in range(L):
            m = causal_mask(ys.size(1), dev)
            logit = model.decode(ys, mem, tgt_mask=m)
            nxt = logit[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], 1)
    print("\nsrc      :", s[0].tolist())
    print("reversed :", s[0].flip(0).tolist())
    print("predicted:", ys[0, 1:].tolist())


if __name__ == "__main__":
    demo()

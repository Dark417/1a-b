"""
T5 — encoder-decoder "text-to-text" Transformer
================================================
T5 frames *every* NLP problem as text -> text: translation, classification,
summarization, span infilling all become "read an input string, write an output
string". Architecturally it is the full Transformer (bidirectional encoder +
causal decoder with cross-attention). Its pretraining objective is **span
corruption**: contiguous spans of the input are replaced by sentinel tokens and
the decoder must emit the missing spans.

Variants implemented here:
    - Bidirectional encoder + causal decoder with cross-attention (full seq2seq)
    - Span-corruption objective (conceptual builder + demo helper)
    - Sinusoidal absolute positional encoding (self-contained)
    - Greedy decoding for inference

Training techniques demonstrated:
    - TEXT-TO-TEXT framing (one model, many tasks)
    - SPAN CORRUPTION denoising objective (sentinel tokens)
    - Teacher forcing + causal masking in the decoder

References:
    - Raffel et al. (2020), "Exploring the Limits of Transfer Learning … T5"
    - Vaswani et al. (2017), "Attention Is All You Need"
"""

from __future__ import annotations

import math
import numpy as np

SEED = 0

# Toy token convention: 0=PAD, 1=BOS, 2=EOS, sentinels start at 3.
PAD, BOS, EOS = 0, 1, 2
N_SENTINEL = 2                          # number of <extra_id_*> sentinel tokens
SENTINEL0 = 3                          # first sentinel id


# ---------------------------------------------------------------------------
# 1. NumPy: sinusoidal PE + the span-corruption transform (the T5 objective)
# ---------------------------------------------------------------------------
def sinusoidal_encoding(L: int, d_model: int) -> np.ndarray:
    """PE[pos,2i]=sin(pos/10000^{2i/d}); PE[pos,2i+1]=cos(...). See transformer.py."""
    pos = np.arange(L)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((L, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def span_corrupt(tokens: list[int], rng, p: float = 0.4,
                 sentinel0: int = SENTINEL0):
    r"""
    Span corruption (the T5 denoising objective), single sequence.

    Mark ~p of tokens for corruption; merge consecutive marked tokens into spans.
    Each span is replaced *in the input* by a unique sentinel <extra_id_k>, and the
    *target* is the concatenation of sentinels followed by their dropped spans:

        input  :  the <X> walked <Y> dog
        target :  <X> cat <Y> the    (then EOS)

    So the encoder sees a corrupted string and the decoder reconstructs only the
    missing pieces (much cheaper than reproducing the whole input).
    Returns (corrupted_input, target).
    """
    mark = rng.random(len(tokens)) < p
    src, tgt = [], []
    k = 0
    i = 0
    n = len(tokens)
    while i < n:
        if mark[i]:
            sent = sentinel0 + k
            src.append(sent)            # one sentinel for the whole span
            tgt.append(sent)
            while i < n and mark[i]:     # collect the contiguous span
                tgt.append(tokens[i])
                i += 1
            k += 1
        else:
            src.append(tokens[i])
            i += 1
    return src, tgt


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — a compact encoder-decoder Transformer
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
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.tensor(sinusoidal_encoding(max_len, d_model), dtype=torch.float32)
        self.register_buffer("pe", pe)

    def forward(self, x):                       # x: (B, L, d)
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    """Generic MHA usable for self- and cross-attention; additive mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):      # mask: additive (..., Lq, Lk)
        Q, K, V = self._split(self.wq(q)), self._split(self.wk(k)), self._split(self.wv(v))
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        attn = self.drop(scores.softmax(-1))
        o = (attn @ V).transpose(1, 2).reshape(q.size(0), -1, self.h * self.d_k)
        return self.out(o)


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

    def forward(self, x, src_mask=None):        # pre-norm + residual
        x = x + self.attn(self.n1(x), self.n1(x), self.n1(x), src_mask)
        x = x + self.ff(self.n2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)

    def forward(self, x, mem, tgt_mask=None, src_mask=None):
        x = x + self.self_attn(self.n1(x), self.n1(x), self.n1(x), tgt_mask)
        nx = self.n2(x)
        x = x + self.cross_attn(nx, mem, mem, src_mask)         # attend to encoder
        x = x + self.ff(self.n3(x))
        return x


class T5(nn.Module):
    """Encoder-decoder text-to-text Transformer (shared vocab + tied embedding)."""

    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 128, n_layers: int = 2, max_len: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)  # shared enc/dec
        self.pos = PositionalEncoding(d_model, max_len)
        self.enc = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout)
                                 for _ in range(n_layers))
        self.dec = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout)
                                 for _ in range(n_layers))
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight                       # weight tying
        self.d_model = d_model

    def encode(self, src, src_mask=None):
        x = self.pos(self.emb(src) * math.sqrt(self.d_model))
        for layer in self.enc:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, mem, tgt_mask=None, src_mask=None):
        x = self.pos(self.emb(tgt) * math.sqrt(self.d_model))
        for layer in self.dec:
            x = layer(x, mem, tgt_mask, src_mask)
        return self.head(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.decode(tgt, self.encode(src, src_mask), tgt_mask, src_mask)


def causal_mask(L: int, device) -> torch.Tensor:
    """Additive (1, 1, L, L) mask: 0 on/below diagonal, -inf above."""
    m = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
    return torch.zeros(L, L, device=device).masked_fill(m, float("-inf"))[None, None]


# ---------------------------------------------------------------------------
# 3. Demo — a tiny text-to-text task: REVERSE the input sequence
# ---------------------------------------------------------------------------
def make_reverse_data(n: int, L: int, vocab: int, seed: int = SEED):
    """src = random content tokens; tgt = reversed, framed with BOS/EOS.
    A clean instance of the text-to-text view: input string -> output string."""
    rng = np.random.default_rng(seed)
    lo = SENTINEL0 + N_SENTINEL                              # content tokens above sentinels
    src = rng.integers(lo, vocab, size=(n, L))
    rev = src[:, ::-1]
    bos = np.full((n, 1), BOS)
    eos = np.full((n, 1), EOS)
    tgt_in = np.concatenate([bos, rev], axis=1)             # teacher-forcing input
    tgt_out = np.concatenate([rev, eos], axis=1)            # shifted target
    return src.astype(np.int64), tgt_in.astype(np.int64), tgt_out.astype(np.int64)


def demo():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)        # tiny CPU model: 1 thread avoids oversubscription
    dev = get_device()

    # First: illustrate the span-corruption objective on one toy sentence.
    rng = np.random.default_rng(SEED)
    sentence = [10, 11, 12, 13, 14, 15]
    src_sc, tgt_sc = span_corrupt(sentence, rng, p=0.5)
    print("span corruption (objective):")
    print("  original :", sentence)
    print("  input    :", src_sc, "  (sentinels start at id %d)" % SENTINEL0)
    print("  target   :", tgt_sc)

    # Then: train the seq2seq model on the text-to-text REVERSE task.
    V, L = 20, 6
    src, tin, tout = make_reverse_data(512, L, V)
    src = torch.tensor(src, device=dev)
    tin = torch.tensor(tin, device=dev)
    tout = torch.tensor(tout, device=dev)

    model = T5(V, d_model=48, n_heads=4, d_ff=96, n_layers=2,
               max_len=L + 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossfn = nn.CrossEntropyLoss(ignore_index=PAD)

    model.train()
    tmask = causal_mask(tin.size(1), dev)
    print("\ntext-to-text REVERSE task:")
    for step in range(1, 301):
        logits = model(src, tin, tgt_mask=tmask)
        loss = lossfn(logits.reshape(-1, V), tout.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 75 == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    # greedy decode one example
    model.eval()
    with torch.no_grad():
        s = src[:1]
        mem = model.encode(s)
        ys = torch.tensor([[BOS]], device=dev)
        for _ in range(L):
            m = causal_mask(ys.size(1), dev)
            logit = model.decode(ys, mem, tgt_mask=m)
            nxt = logit[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
    print("\n  src      :", s[0].tolist())
    print("  reversed :", s[0].flip(0).tolist())
    print("  predicted:", ys[0, 1:].tolist())


if __name__ == "__main__":
    demo()

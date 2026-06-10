"""
GPT — decoder-only causal Transformer (autoregressive language model)
=====================================================================
GPT keeps only the *decoder* half of the Transformer: a stack of blocks doing
**masked (causal) self-attention** followed by a position-wise MLP. It models a
sequence left-to-right, factorizing the joint probability as a product of
next-token conditionals, and is trained by plain next-token prediction. The same
weights generate text by sampling one token at a time.

Variants implemented here:
    - Causal multi-head self-attention (the core of GPT)
    - Pre-norm Transformer decoder block (GPT-2 style: LN before sublayers)
    - Learned token + learned absolute positional embeddings
    - Weight tying (input embedding == output projection)
    - Autoregressive generation with temperature + top-k sampling

Training techniques demonstrated:
    - CAUSAL MASKING (enforces the autoregressive factorization)
    - Pre-norm residual blocks (stable deep training; see transformer.py)
    - Cross-entropy next-token objective; temperature sampling at inference

References:
    - Radford et al. (2018, 2019), "Improving Language Understanding…" / GPT-2
    - Vaswani et al. (2017), "Attention Is All You Need"
"""

from __future__ import annotations

import math
import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy: the causal mask (the one piece with crisp closed-form math)
# ---------------------------------------------------------------------------
def causal_mask_numpy(L: int) -> np.ndarray:
    r"""
    Boolean lower-triangular mask of shape (L, L).

        keep[i, j] = 1  if  j <= i   (token i may attend to token j)
                     0  if  j >  i   (no peeking at the future)

    Applied as an *additive* mask: scores[i, j] += (0 if keep else -inf) before
    softmax. This is exactly what turns self-attention into a valid
    autoregressive model p(x) = prod_t p(x_t | x_<t).
    """
    return np.tril(np.ones((L, L), dtype=np.int64))


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — a compact but complete GPT
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


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal (look-ahead) mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, L, d)
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                        # (B, h, L, d_k)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)  # (B, h, L, L)
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))        # block the future
        attn = self.drop(scores.softmax(-1))
        o = (attn @ v).transpose(1, 2).reshape(B, L, self.h * self.d_k)
        return self.out(o)


class FeedForward(nn.Module):
    """Position-wise MLP with GELU (GPT-2 style)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """Pre-norm block: x = x + Attn(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only causal language model with weight tying."""

    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 128, n_layers: int = 2, max_len: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)            # learned absolute
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok_emb.weight                   # weight tying

    def forward(self, idx: torch.Tensor) -> torch.Tensor:        # idx: (B, L)
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))                           # (B, L, vocab)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n_new: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressively extend idx by n_new tokens (sampling one at a time)."""
        self.eval()
        for _ in range(n_new):
            idx_cond = idx[:, -self.max_len:]                    # crop to context
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")      # keep only top-k
            probs = logits.softmax(-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------------------
# 3. Demo — learn a deterministic pattern, then generate from it
# ---------------------------------------------------------------------------
def make_pattern_data(n: int, L: int, period: int, seed: int = SEED):
    """Sequences from a fixed repeating cycle 0,1,...,period-1,0,1,... with a
    random phase per sequence. A causal LM should learn 'next = (cur+1) % period'."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, period, size=(n, 1))
    offs = np.arange(L)[None, :]
    return ((starts + offs) % period).astype(np.int64)           # (n, L)


def demo():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)        # tiny CPU model: 1 thread avoids oversubscription
    dev = get_device()

    period, V, L = 5, 5, 12        # vocab == period (the cyclic alphabet)
    data = torch.tensor(make_pattern_data(256, L + 1, period), device=dev)
    x, y = data[:, :-1], data[:, 1:]                             # next-token targets

    model = GPT(V, d_model=48, n_heads=4, d_ff=96, n_layers=2, max_len=L).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossfn = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, 251):
        logits = model(x)
        loss = lossfn(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    # greedy/temperature generation: start from a single token, continue the cycle
    start = torch.tensor([[2]], device=dev)                      # begin at token 2
    out = model.generate(start, n_new=L - 1, temperature=0.5, top_k=3)
    seq = out[0].tolist()
    expected = [(2 + i) % period for i in range(L)]
    print("\ngenerated:", seq)
    print("expected :", expected, "(cyclic +1 mod %d)" % period)
    print("match    :", seq == expected)


if __name__ == "__main__":
    demo()

"""
BERT — encoder-only Transformer with masked-language-model pretraining
======================================================================
BERT keeps only the *encoder* of the Transformer and trains it **bidirectionally**:
every token attends to every other token (no causal mask). Pretraining masks out
~15% of tokens and asks the model to reconstruct them (the Masked Language Model
objective), which forces deep bidirectional context. A special [CLS] token
summarizes the sequence for classification, and Next-Sentence-Prediction (NSP)
is included conceptually via segment embeddings.

Variants implemented here:
    - Bidirectional multi-head self-attention (no causal mask; padding mask only)
    - Token + learned positional + segment embeddings ([CLS]/[SEP] structure)
    - Masked-Language-Model head (the pretraining objective)
    - Pooled [CLS] head for sentence-level tasks / NSP (conceptual)

Training techniques demonstrated:
    - MASKED LANGUAGE MODELING (the 80/10/10 masking scheme)
    - Bidirectional attention with a PADDING mask
    - Pre-norm residual blocks, GELU MLP (stable deep training)

References:
    - Devlin et al. (2019), "BERT: Pre-training of Deep Bidirectional Transformers"
    - Vaswani et al. (2017), "Attention Is All You Need"
"""

from __future__ import annotations

import math
import numpy as np

SEED = 0

# Reserved special-token ids used throughout (toy convention).
PAD, CLS, SEP, MASK = 0, 1, 2, 3
N_SPECIAL = 4


# ---------------------------------------------------------------------------
# 1. NumPy: the MLM masking scheme (the heart of BERT pretraining)
# ---------------------------------------------------------------------------
def mlm_mask_numpy(tokens: np.ndarray, vocab: int, mask_id: int = MASK,
                   n_special: int = N_SPECIAL, p: float = 0.15,
                   seed: int = SEED):
    r"""
    BERT's 80/10/10 masking. Pick 15% of non-special positions; of those:
        - 80%  -> replace with [MASK]
        - 10%  -> replace with a random token
        - 10%  -> keep unchanged
    The model must predict the ORIGINAL token at every chosen position. The 10%
    "keep" / "random" splits stop the model from only ever seeing [MASK] at train
    time (which never appears at fine-tuning time) and force it to model every
    token, not just the masked slots.

    Returns (corrupted_tokens, labels) where labels = original id at chosen
    positions and -100 elsewhere (so cross-entropy ignores them).
    """
    rng = np.random.default_rng(seed)
    corrupted = tokens.copy()
    labels = np.full_like(tokens, -100)
    for b in range(tokens.shape[0]):
        for t in range(tokens.shape[1]):
            if tokens[b, t] < n_special:        # never mask [PAD]/[CLS]/[SEP]
                continue
            if rng.random() < p:
                labels[b, t] = tokens[b, t]      # remember the answer
                r = rng.random()
                if r < 0.8:
                    corrupted[b, t] = mask_id            # 80% -> [MASK]
                elif r < 0.9:
                    corrupted[b, t] = rng.integers(n_special, vocab)  # 10% random
                # else 10% -> leave unchanged
    return corrupted, labels


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — a compact but complete BERT
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


class BidirectionalSelfAttention(nn.Module):
    """Multi-head self-attention with an optional *padding* mask (no causal mask)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, h, L, d_k)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        if pad_mask is not None:                               # (B, L): 1=keep, 0=pad
            scores = scores.masked_fill(pad_mask[:, None, None, :] == 0, float("-inf"))
        attn = self.drop(scores.softmax(-1))
        o = (attn @ v).transpose(1, 2).reshape(B, L, self.h * self.d_k)
        return self.out(o)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x): return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm encoder block: x = x + Attn(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.ln1(x), pad_mask)
        x = x + self.ff(self.ln2(x))
        return x


class BERT(nn.Module):
    """Encoder-only Transformer with MLM head + pooled [CLS] head (NSP)."""

    def __init__(self, vocab: int, d_model: int = 64, n_heads: int = 4,
                 d_ff: int = 128, n_layers: int = 2, max_len: int = 64,
                 n_segments: int = 2, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)          # learned absolute
        self.seg_emb = nn.Embedding(n_segments, d_model)       # sentence A / B
        self.ln_in = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        # MLM head (tied to input embedding) and a pooled NSP head.
        self.mlm = nn.Linear(d_model, vocab, bias=True)
        self.mlm.weight = self.tok_emb.weight                  # weight tying
        self.pool = nn.Linear(d_model, d_model)
        self.nsp = nn.Linear(d_model, 2)

    def encode(self, idx, seg=None, pad_mask=None):
        B, L = idx.shape
        pos = torch.arange(L, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        if seg is not None:
            x = x + self.seg_emb(seg)
        x = self.drop(self.ln_in(x))
        for block in self.blocks:
            x = block(x, pad_mask)
        return x                                               # (B, L, d)

    def forward(self, idx, seg=None, pad_mask=None):
        h = self.encode(idx, seg, pad_mask)
        mlm_logits = self.mlm(h)                               # (B, L, vocab)
        pooled = torch.tanh(self.pool(h[:, 0]))                # [CLS] summary
        nsp_logits = self.nsp(pooled)                          # (B, 2)
        return mlm_logits, nsp_logits


# ---------------------------------------------------------------------------
# 3. Demo — tiny MLM: reconstruct masked tokens of toy sequences
# ---------------------------------------------------------------------------
def make_sequences(n: int, L: int, vocab: int, seed: int = SEED):
    """[CLS] w1 w2 ... wL [SEP]  with random content words (ids >= N_SPECIAL)."""
    rng = np.random.default_rng(seed)
    words = rng.integers(N_SPECIAL, vocab, size=(n, L))
    cls = np.full((n, 1), CLS)
    sep = np.full((n, 1), SEP)
    return np.concatenate([cls, words, sep], axis=1).astype(np.int64)


def demo():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = get_device()

    V, L, n = 24, 8, 1024
    seqs = make_sequences(n, L, V)                             # (n, L+2)
    corrupt, labels = mlm_mask_numpy(seqs, V, p=0.20)          # mask ~20% to learn fast
    seg = np.zeros_like(seqs)                                  # single segment here
    pad_mask = (seqs != PAD).astype(np.int64)

    X = torch.tensor(corrupt, device=dev)
    Y = torch.tensor(labels, device=dev)
    S = torch.tensor(seg, device=dev)
    M = torch.tensor(pad_mask, device=dev)

    model = BERT(V, d_model=64, n_heads=4, d_ff=128, n_layers=2,
                 max_len=L + 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossfn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for step in range(1, 301):
        mlm_logits, _ = model(X, S, M)
        loss = lossfn(mlm_logits.reshape(-1, V), Y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 75 == 0:
            # accuracy on the masked positions
            with torch.no_grad():
                pred = mlm_logits.argmax(-1)
                sel = Y != -100
                acc = (pred[sel] == Y[sel]).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.4f}  masked-acc {acc:.2f}")

    # show one example: original vs corrupted vs reconstruction at masked slots
    model.eval()
    with torch.no_grad():
        mlm_logits, _ = model(X[:1], S[:1], M[:1])
        pred = mlm_logits[0].argmax(-1)
    masked_pos = (Y[0] != -100).nonzero(as_tuple=True)[0].tolist()
    print("\noriginal   :", seqs[0].tolist())
    print("corrupted  :", X[0].tolist(), "  (id %d = [MASK])" % MASK)
    print("masked pos :", masked_pos)
    print("true @pos  :", [int(seqs[0][p]) for p in masked_pos])
    print("pred @pos  :", [int(pred[p]) for p in masked_pos])


if __name__ == "__main__":
    demo()

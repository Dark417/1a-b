"""
Attention (scaled dot-product & multi-head)
============================================
Attention lets every position look directly at every other position and pull in
the information it needs — a content-based, differentiable lookup. It replaced
recurrence as the core sequence operation and powers all modern LLMs.

Variants implemented here:
    - Scaled dot-product attention (the primitive)
    - Multi-head attention (several attention "views" in parallel)
    - Self-attention vs cross-attention
    - Causal (look-ahead) masking for autoregressive models

Training techniques demonstrated:
    - The 1/sqrt(d_k) scaling (keeps softmax gradients healthy)
    - Masking (causal & padding)

References:
    - Vaswani et al. (2017), "Attention Is All You Need"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# 1. NumPy implementation
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    r"""
    Attention(Q,K,V) = softmax( Q Kᵀ / sqrt(d_k) ) V

    Q:(..., Lq, d_k)  K:(..., Lk, d_k)  V:(..., Lk, d_v)
    mask: additive (0 keep, -inf block) broadcast to scores shape.
    Returns (output, attention_weights).
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)     # (..., Lq, Lk)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)                     # rows sum to 1
    return weights @ V, weights


def causal_mask(L):
    """Upper-triangular -inf so position i can't attend to j>i."""
    m = np.triu(np.ones((L, L)), k=1)
    return np.where(m == 1, -1e9, 0.0)


class MultiHeadAttentionNumPy:
    r"""
    Project Q,K,V into h heads, attend in each (d_k = d_model/h), concat, project.
        head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
        MHA    = Concat(head_1..head_h) W^O
    Multiple heads let the model attend to different relations simultaneously.
    """

    def __init__(self, d_model, n_heads, seed=SEED):
        assert d_model % n_heads == 0
        self.d_model, self.h = d_model, n_heads
        self.d_k = d_model // n_heads
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(d_model)
        self.Wq = rng.normal(0, s, (d_model, d_model))
        self.Wk = rng.normal(0, s, (d_model, d_model))
        self.Wv = rng.normal(0, s, (d_model, d_model))
        self.Wo = rng.normal(0, s, (d_model, d_model))

    def _split(self, x):                # (B,L,d) -> (B,h,L,d_k)
        B, L, _ = x.shape
        return x.reshape(B, L, self.h, self.d_k).transpose(0, 2, 1, 3)

    def _merge(self, x):                # (B,h,L,d_k) -> (B,L,d)
        B, h, L, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, L, h * d_k)

    def __call__(self, query, key, value, mask=None):
        Q = self._split(query @ self.Wq)
        K = self._split(key @ self.Wk)
        V = self._split(value @ self.Wv)
        out, self.weights = scaled_dot_product_attention(Q, K, V, mask)
        return self._merge(out) @ self.Wo


# ---------------------------------------------------------------------------
# 2. PyTorch implementation
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionTorch(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def _split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        Q, K, V = self._split(self.Wq(q)), self._split(self.Wk(k)), self._split(self.Wv(v))
        scores = Q @ K.transpose(-1, -2) / self.d_k ** 0.5
        if mask is not None:
            scores = scores + mask
        self.weights = scores.softmax(-1)
        out = self.weights @ V
        B, h, L, d_k = out.shape
        out = out.transpose(1, 2).reshape(B, L, h * d_k)
        return self.Wo(out)


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    B, L, d_model, h = 2, 6, 16, 4
    x = np.random.randn(B, L, d_model).astype(np.float32)

    mha = MultiHeadAttentionNumPy(d_model, h)
    out = mha(x, x, x)                                # self-attention
    print(f"NumPy MHA self-attention out shape: {out.shape}")
    print(f"attention weights row sums (≈1): {mha.weights[0,0,0].sum():.3f}")

    # causal masking: position 0 should only attend to itself
    m = causal_mask(L)
    _ = mha(x, x, x, mask=m)
    upper = mha.weights[0, 0][np.triu_indices(L, k=1)]
    print(f"causal mask: max weight above diagonal = {upper.max():.2e} (≈0)")

    tor = MultiHeadAttentionTorch(d_model, h)
    ot = tor(torch.tensor(x), torch.tensor(x), torch.tensor(x))
    print(f"Torch MHA out shape: {tuple(ot.shape)}")

    # cross-attention: query length differs from key/value length
    q = np.random.randn(B, 3, d_model).astype(np.float32)
    oc = mha(q, x, x)
    print(f"cross-attention (Lq=3, Lk=6) out shape: {oc.shape}")


if __name__ == "__main__":
    demo()

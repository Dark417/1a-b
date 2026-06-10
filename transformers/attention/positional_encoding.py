"""
Positional Encodings — sinusoidal, learned, RoPE, ALiBi
=======================================================
Self-attention is permutation-equivariant: shuffle the input tokens and the set
of outputs just shuffles the same way — the model has *no* notion of order. We
therefore inject position information. Four classic schemes are implemented here,
each with the NumPy math made explicit alongside the idiomatic PyTorch module.

Variants implemented here:
    - Sinusoidal (absolute, fixed)        — Vaswani et al. 2017
    - Learned absolute embeddings          — BERT/GPT style
    - RoPE (rotary position embedding)     — Su et al. 2021 (relative via rotation)
    - ALiBi (attention with linear biases) — Press et al. 2022 (relative via bias)

Training techniques demonstrated:
    - Absolute vs relative position encoding and their length-extrapolation behaviour
    - How RoPE injects RELATIVE position purely through rotations of Q and K
    - How ALiBi biases attention scores with a distance-proportional penalty

References:
    - Vaswani et al. (2017), "Attention Is All You Need"
    - Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - Press, Smith, Lewis (2022), "Train Short, Test Long: ALiBi"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementations — the closed-form math
# ---------------------------------------------------------------------------
def sinusoidal_encoding(L: int, d_model: int) -> np.ndarray:
    r"""
    Absolute sinusoidal encoding (added to token embeddings).

        PE[pos, 2i]   = sin(pos / 10000^{2i/d})
        PE[pos, 2i+1] = cos(pos / 10000^{2i/d})

    Each dimension is a sinusoid whose wavelength grows geometrically from 2*pi to
    10000*2*pi. Because sin/cos of a shifted angle are LINEAR combinations of the
    unshifted ones, a fixed offset k maps PE[pos] -> PE[pos+k] by a position-
    independent linear transform, so attention can learn to attend by relative
    offset.
    """
    pos = np.arange(L)[:, None]                 # (L,1)
    i = np.arange(d_model)[None, :]             # (1,d)
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((L, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])        # even dims -> sin
    pe[:, 1::2] = np.cos(angle[:, 1::2])        # odd dims  -> cos
    return pe


def rope_angles(L: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    r"""
    Per-(position, dim-pair) rotation angle for RoPE.

        theta_j = base^{-2j/d}  for j = 0..d/2-1
        angle[pos, j] = pos * theta_j

    RoPE splits the feature vector into d/2 pairs and rotates pair j of the query
    and key by angle (pos * theta_j). The dot product of two rotated vectors then
    depends only on the DIFFERENCE of their positions — relative position emerges
    from absolute rotation.
    """
    j = np.arange(d_model // 2)                 # (d/2,)
    theta = base ** (-2.0 * j / d_model)        # decreasing frequencies
    pos = np.arange(L)[:, None]                 # (L,1)
    return pos * theta[None, :]                 # (L, d/2)


def apply_rope_numpy(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    r"""
    Apply rotary embedding to x of shape (L, d) (d even).

    For pair j with components (x_{2j}, x_{2j+1}) at position pos:
        [ cos(pos*theta_j)  -sin(pos*theta_j) ] [ x_{2j}   ]
        [ sin(pos*theta_j)   cos(pos*theta_j) ] [ x_{2j+1} ]
    i.e. a 2-D rotation of each consecutive pair by a position-dependent angle.
    """
    L, d = x.shape
    ang = rope_angles(L, d, base)               # (L, d/2)
    cos, sin = np.cos(ang), np.sin(ang)
    x1 = x[:, 0::2]                             # even components of each pair
    x2 = x[:, 1::2]                             # odd  components of each pair
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


def alibi_slopes(n_heads: int) -> np.ndarray:
    r"""
    ALiBi per-head slopes: a geometric sequence m_h = 2^{-8h/H} for h=1..H.

    Head h adds a bias -m_h * |i-j| (here we use the causal form -m_h*(i-j) for
    j<=i) to the attention score, gently penalizing distant keys. Different slopes
    let different heads have different effective context windows.
    """
    # Standard recipe: ratio start 2^{-8/H}, geometric.
    start = 2 ** (-8.0 / n_heads)
    return start ** (np.arange(1, n_heads + 1))


def alibi_bias(n_heads: int, L: int, causal: bool = True) -> np.ndarray:
    r"""
    Additive bias tensor of shape (n_heads, L, L).

        bias[h, i, j] = -m_h * (i - j)   (causal, j <= i)
        bias[h, i, j] = -m_h * |i - j|   (bidirectional)

    Added to QK^T/sqrt(d) before softmax; no learned parameters, no embeddings.
    """
    slopes = alibi_slopes(n_heads)[:, None, None]      # (H,1,1)
    i = np.arange(L)[None, :, None]
    j = np.arange(L)[None, None, :]
    if causal:
        dist = (i - j).astype(float)
        dist = np.where(j <= i, dist, np.inf)          # forbid future (->-inf bias)
        bias = -slopes * dist
        bias = np.where(np.isfinite(bias), bias, -1e9)
    else:
        bias = -slopes * np.abs(i - j).astype(float)
    return bias


# ---------------------------------------------------------------------------
# 2. PyTorch implementations (idiomatic nn.Module)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding added to embeddings (no learned parameters)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.tensor(sinusoidal_encoding(max_len, d_model), dtype=torch.float32)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, L, d)
        return x + self.pe[: x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """A trainable embedding table indexed by position (BERT/GPT style)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, L, d)
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.emb(pos)[None]


class RotaryPositionalEncoding(nn.Module):
    r"""
    RoPE: rotate query/key feature pairs by a position-dependent angle.

    Call `apply(q)` and `apply(k)` *inside* attention (RoPE is not added to the
    embedding; it multiplies Q and K). The cached cos/sin are (max_len, d).
    """

    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, "RoPE needs an even feature dim"
        ang = torch.tensor(rope_angles(max_len, d_model, base), dtype=torch.float32)
        # interleave so cos/sin line up with (x0,x1,x2,...) pairs
        cos = torch.repeat_interleave(torch.cos(ang), 2, dim=-1)   # (L,d)
        sin = torch.repeat_interleave(torch.sin(ang), 2, dim=-1)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """(x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...) — the 'swap & negate' for pairs."""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, h, L, d). Returns rotated tensor of the same shape."""
        L = x.size(-2)
        cos = self.cos[:L]
        sin = self.sin[:L]
        return x * cos + self._rotate_half(x) * sin


class ALiBiBias(nn.Module):
    """Produces the additive ALiBi bias (n_heads, L, L); no learned parameters."""

    def __init__(self, n_heads: int, max_len: int = 512, causal: bool = True):
        super().__init__()
        bias = torch.tensor(alibi_bias(n_heads, max_len, causal), dtype=torch.float32)
        self.register_buffer("bias", bias)

    def forward(self, L: int) -> torch.Tensor:
        return self.bias[:, :L, :L]


# A tiny self-attention block that can use any of the four schemes, so the demo
# can compare them on a real (toy) task.
class PosAwareAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 64,
                 scheme: str = "sinusoidal", causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.scheme, self.causal = scheme, causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        if scheme == "sinusoidal":
            self.abs_pe = SinusoidalPositionalEncoding(d_model, max_len)
        if scheme == "learned":
            self.abs_pe = LearnedPositionalEncoding(d_model, max_len)
        if scheme == "rope":
            self.rope = RotaryPositionalEncoding(self.d_k, max_len)
        if scheme == "alibi":
            self.alibi = ALiBiBias(n_heads, max_len, causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, L, d)
        if self.scheme in ("sinusoidal", "learned"):
            x = self.abs_pe(x)                            # add absolute position
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # (B,h,L,d_k)
        if self.scheme == "rope":
            q, k = self.rope.apply(q), self.rope.apply(k)
        scores = q @ k.transpose(-1, -2) / self.d_k ** 0.5
        if self.scheme == "alibi":
            scores = scores + self.alibi(L)[None]
        if self.causal and self.scheme != "alibi":
            mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
            scores = scores.masked_fill(mask, -1e9)
        attn = scores.softmax(-1)
        o = (attn @ v).transpose(1, 2).reshape(B, L, self.h * self.d_k)
        return self.out(o)


# ---------------------------------------------------------------------------
# 3. Demo — verify the math + a tiny task where position matters
# ---------------------------------------------------------------------------
def _rope_relative_check():
    """Show <RoPE(q,m), RoPE(k,n)> depends only on (m-n)."""
    rng = np.random.default_rng(SEED)
    d = 8
    q = rng.normal(size=d)
    k = rng.normal(size=d)
    # rotate q at position m, k at position n; compare two pairs with same m-n
    def rot(vec, pos):
        x = vec[None].copy()
        # build a length-(pos+1) array, take last row = position `pos`
        big = np.repeat(x, pos + 1, axis=0)
        return apply_rope_numpy(big)[pos]
    a = rot(q, 5) @ rot(k, 3)      # m-n = 2
    b = rot(q, 9) @ rot(k, 7)      # m-n = 2 too
    return a, b


def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dev = get_device()

    # --- (a) sinusoidal encoding basics
    pe = sinusoidal_encoding(20, 16)
    print(f"sinusoidal PE shape {pe.shape}, values in [{pe.min():.2f},{pe.max():.2f}]")

    # --- (b) RoPE relative-position property (the key idea)
    a, b = _rope_relative_check()
    print(f"RoPE: <q@5,k@3>={a:.4f}  <q@9,k@7>={b:.4f}  "
          f"(equal => depends only on m-n: diff={abs(a-b):.2e})")

    # --- (c) ALiBi slopes & bias
    print(f"ALiBi slopes (H=4): {np.round(alibi_slopes(4), 4).tolist()}")
    bias = alibi_bias(4, 5)
    print(f"ALiBi bias head0 row(i=4): {np.round(bias[0,4],2).tolist()} (distant keys penalized)")

    # --- (d) tiny task: predict whether token at position 0 equals token at the
    # LAST position (a position-sensitive parity/match task). All four schemes
    # should learn it; this just confirms each block trains end-to-end fast.
    V, L = 6, 8
    n = 1024
    rng = np.random.default_rng(SEED)
    seqs = rng.integers(1, V, size=(n, L))
    labels = (seqs[:, 0] == seqs[:, -1]).astype(np.int64)   # does first==last?
    X = torch.tensor(seqs, device=dev)
    y = torch.tensor(labels, device=dev)

    emb = nn.Embedding(V, 32).to(dev)

    def train_scheme(scheme):
        torch.manual_seed(SEED)
        block = PosAwareAttention(32, 4, max_len=L, scheme=scheme, causal=False).to(dev)
        head = nn.Linear(32, 2).to(dev)
        params = list(emb.parameters()) + list(block.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=3e-3)
        lossfn = nn.CrossEntropyLoss()
        for step in range(300):
            h = block(emb(X))           # (B,L,32)
            logits = head(h.mean(1))    # pool then classify
            loss = lossfn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        acc = (logits.argmax(-1) == y).float().mean().item()
        return loss.item(), acc

    print("\ntiny 'first==last?' task (each block trained 300 steps):")
    for sch in ["sinusoidal", "learned", "rope", "alibi"]:
        loss, acc = train_scheme(sch)
        print(f"  {sch:10s}  final loss {loss:.3f}  train acc {acc:.2f}")


if __name__ == "__main__":
    demo()

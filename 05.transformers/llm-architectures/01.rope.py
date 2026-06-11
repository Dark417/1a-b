"""
RoPE — Rotary Position Embeddings (+ NTK / linear scaling, YaRN idea)
====================================================================
Instead of *adding* a position vector, RoPE *rotates* the query/key vectors by
an angle proportional to their absolute position. Because a rotation by angle
m θ followed by a rotation by −n θ leaves only the relative angle (m−n)θ, the
dot product q_m · k_n depends only on the relative offset (m − n). This gives
attention an inductive bias for relative position while costing nothing extra
in parameters, and it extrapolates / interpolates gracefully to longer contexts.

The trick: pair up the d head dimensions into d/2 2-D planes. In plane i use
frequency θ_i = base^{-2i/d} (base = 10000). Rotating the pair (x_{2i}, x_{2i+1})
by m·θ_i is a 2-D rotation:
    [x'_{2i}  ]   [cos(mθ_i)  -sin(mθ_i)] [x_{2i}  ]
    [x'_{2i+1}] = [sin(mθ_i)   cos(mθ_i)] [x_{2i+1}]

Context-length extension (frequencies are the only thing that "knows" length):
    - Linear position interpolation (PI, Chen 2023): divide positions by a scale
      s = L_new / L_train, i.e. m -> m / s. Squeezes new positions into the
      trained range. Hurts high-frequency (local) detail.
    - NTK-aware scaling (bloc97 2023): instead of scaling positions, scale the
      *base*: base' = base · s^{d/(d-2)}. Stretches low frequencies a lot, high
      frequencies almost none — preserves local resolution. "NTK" because it
      mirrors neural-tangent-kernel intuition about high vs low frequencies.
    - YaRN (Peng et al. 2023): interpolate *per-wavelength*. Wavelengths shorter
      than the context get NO interpolation (keep local detail); long
      wavelengths get full linear interpolation; a ramp blends the middle. Also
      multiplies attention logits by a temperature 1/t (≈0.1 ln s + 1) to keep
      the softmax entropy stable. Best length-extension quality of the three.

What introduced it: RoFormer (Su et al. 2021); adopted by LLaMA, GPT-NeoX,
PaLM, Qwen, Mistral, DeepSeek, and essentially every modern decoder LLM.

References:
    - Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position
      Embedding"
    - Chen et al. (2023), "Extending Context Window of LLMs via Position
      Interpolation"
    - bloc97 (2023), "NTK-Aware Scaled RoPE" (community report)
    - Peng et al. (2023), "YaRN: Efficient Context Window Extension of LLMs"
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def rope_frequencies(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Inverse frequencies θ_i = base^{-2i/dim} for i = 0..dim/2-1."""
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    return 1.0 / (base ** (i / dim))                      # (dim/2,)


def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0,
                     linear_scale: float = 1.0, ntk_scale: float = 1.0):
    r"""
    Precompute cos/sin tables of shape (seq_len, dim).

    linear_scale s>1  : positions m -> m / s         (position interpolation)
    ntk_scale    s>1  : base -> base * s^{dim/(dim-2)} (NTK-aware, scales freqs)
    """
    if ntk_scale != 1.0:
        base = base * (ntk_scale ** (dim / (dim - 2)))    # NTK-aware base shift
    inv_freq = rope_frequencies(dim, base)                # (dim/2,)
    m = torch.arange(seq_len, dtype=torch.float32) / linear_scale  # PI: m/s
    angles = torch.outer(m, inv_freq)                     # (L, dim/2)
    angles = torch.cat([angles, angles], dim=-1)          # (L, dim) — duplicate
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(-x2, x1) split so that x*cos + rotate_half(x)*sin == the 2-D rotation."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (..., L, dim). cos/sin: (L, dim). Returns rotated x.
    Uses the standard "rotate_half" formulation (GPT-NeoX layout):
        x' = x * cos + rotate_half(x) * sin
    """
    return x * cos + _rotate_half(x) * sin


def yarn_rope_cache(seq_len: int, dim: int, scale: float, base: float = 10000.0,
                    orig_max: int = 2048, beta_fast: float = 32.0,
                    beta_slow: float = 1.0):
    r"""
    YaRN: per-wavelength interpolation ramp + attention temperature.

    For each frequency, decide how much to interpolate based on how many full
    rotations it completes inside the *original* context window:
        rotations(i) = orig_max * inv_freq_i / (2π)
    High-frequency dims (>beta_fast rotations) -> no interpolation (mask 0);
    low-frequency dims (<beta_slow rotations) -> full linear interp (mask 1);
    a linear ramp blends between. inv_freq blends between the un-scaled and the
    PI-scaled inverse frequency by that mask.
    """
    pos = torch.arange(seq_len, dtype=torch.float32)
    inv_freq = rope_frequencies(dim, base)                # original freqs
    inv_freq_interp = inv_freq / scale                    # fully interpolated

    # rotations completed in the original window -> ramp factor in [0,1]
    rotations = orig_max * inv_freq / (2 * math.pi)
    ramp = (rotations - beta_slow) / (beta_fast - beta_slow)
    ramp = ramp.clamp(0.0, 1.0)                           # 1=keep, 0=interpolate
    inv_freq = inv_freq * ramp + inv_freq_interp * (1.0 - ramp)

    angles = torch.outer(pos, inv_freq)
    angles = torch.cat([angles, angles], dim=-1)
    # attention temperature: scale logits by 1/t to keep softmax entropy stable
    mscale = 0.1 * math.log(scale) + 1.0 if scale > 1 else 1.0
    return (angles.cos() * mscale, angles.sin() * mscale)


class RotaryEmbedding(nn.Module):
    """Drop-in module that rotates q and k. dim = per-head dimension."""

    def __init__(self, dim, max_len=512, base=10000.0,
                 linear_scale=1.0, ntk_scale=1.0):
        super().__init__()
        cos, sin = build_rope_cache(max_len, dim, base, linear_scale, ntk_scale)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q, k, offset=0):
        L = q.size(-2)
        cos = self.cos[offset:offset + L]
        sin = self.sin[offset:offset + L]
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


# ---------------------------------------------------------------------------
# Demo — verify the relative-position property
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    dim, L = 16, 12
    cos, sin = build_rope_cache(L, dim)

    # Two fixed random vectors used as a query and a key.
    base_q = torch.randn(dim)
    base_k = torch.randn(dim)

    # KEY CLAIM: after rotation, <rot(q, m), rot(k, n)> depends only on (m - n).
    # Compare pairs with the SAME offset placed at different absolute positions.
    def dot_at(m, n):
        q = apply_rope(base_q.unsqueeze(0), cos[m:m + 1], sin[m:m + 1])
        k = apply_rope(base_k.unsqueeze(0), cos[n:n + 1], sin[n:n + 1])
        return (q * k).sum().item()

    print("Relative-position property (q·k should match for equal m-n):")
    for (m, n) in [(3, 1), (7, 5), (10, 8)]:        # all have offset = 2
        print(f"  m={m:2d} n={n:2d}  (m-n={m-n})  q.k = {dot_at(m, n):+.6f}")
    d1, d2, d3 = dot_at(3, 1), dot_at(7, 5), dot_at(10, 8)
    assert abs(d1 - d2) < 1e-4 and abs(d2 - d3) < 1e-4, "relative prop broken"
    print("  -> all equal: dot product is a function of (m-n) only.  PASS")

    # Different offsets give different dot products (it really uses relative pos).
    print(f"\noffset 2 -> {dot_at(3,1):+.4f}   offset 5 -> {dot_at(7,2):+.4f}")

    # Length-extension variants change the frequency tables.
    print("\nContext extension (cos[8,0] for different schemes; same position):")
    base_c, _ = build_rope_cache(L, dim)
    pi_c, _ = build_rope_cache(L, dim, linear_scale=4.0)         # position interp
    ntk_c, _ = build_rope_cache(L, dim, ntk_scale=4.0)           # NTK-aware
    yarn_c, _ = yarn_rope_cache(L, dim, scale=4.0, orig_max=8)   # YaRN
    print(f"  base={base_c[8,0]:+.4f}  PI={pi_c[8,0]:+.4f}  "
          f"NTK={ntk_c[8,0]:+.4f}  YaRN={yarn_c[8,0]:+.4f}")

    # Module path: rotate q,k inside attention shape (B, H, L, d).
    rot = RotaryEmbedding(dim, max_len=L)
    q = torch.randn(2, 4, L, dim); k = torch.randn(2, 4, L, dim)
    rq, rk = rot(q, k)
    print(f"\nRotaryEmbedding module: q {tuple(q.shape)} -> {tuple(rq.shape)}")
    # rotation preserves norms (it's orthogonal):
    print(f"norm preserved: {torch.allclose(q.norm(dim=-1), rq.norm(dim=-1), atol=1e-4)}")


if __name__ == "__main__":
    demo()

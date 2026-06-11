"""
SwiGLU / GeGLU — Gated Feed-Forward Networks
============================================
The vanilla Transformer FFN is  FFN(x) = W2 · act(W1·x).  GLU-family FFNs replace
the single activated projection with a *gate*: one linear branch is passed through
an activation and multiplies a second (ungated) linear branch element-wise. The
gate lets the network modulate information flow per-dimension, and empirically
GLU variants train better and reach lower loss at equal compute.

    Vanilla (ReLU/GELU):   h = act(x W1) ;        y = h W2
    GLU family:            h = act(x W_gate) ⊙ (x W_up) ;   y = h W_down

The activation names the variant:
    GeGLU   : act = GELU      (used by Gemma, T5-v1.1, PaLM's FFN experiments)
    SwiGLU  : act = SiLU/Swish (x·σ(x))   (used by LLaMA, Mistral, Qwen, ...)
    ReGLU   : act = ReLU
    Bilinear: act = identity

THE 2/3 TRICK. A GLU FFN has THREE weight matrices (gate, up, down) instead of
two, so to keep the parameter / FLOP budget equal to a vanilla FFN with hidden
size d_ff, the GLU hidden size is shrunk to ~2/3·d_ff:
        d_ff_glu = (2/3) · d_ff,  usually rounded to a multiple of 256.
LLaMA reports d_ff ≈ (2/3)·4·d_model = (8/3)·d_model. So "param-matched" SwiGLU
uses a smaller hidden dim but three matrices, landing at the same total params.

What introduced it: Shazeer (2020) "GLU Variants Improve Transformer." SwiGLU is
the default FFN in LLaMA, Mistral, Qwen, DeepSeek, Gemma (GeGLU), and most LLMs.

References:
    - Shazeer (2020), "GLU Variants Improve Transformer"
    - Dauphin et al. (2017), "Language Modeling with Gated Convolutional Networks"
      (original GLU)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaFFN(nn.Module):
    """Classic two-matrix FFN: y = W2 · act(W1·x)."""

    def __init__(self, d_model, d_ff, act=F.gelu):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = act

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class GLUFFN(nn.Module):
    """
    Gated FFN: y = W_down ( act(W_gate·x) ⊙ W_up·x ).
    activation 'silu' -> SwiGLU, 'gelu' -> GeGLU, 'relu' -> ReGLU.
    Pass match_params=True to apply the 2/3 hidden-dim rule.
    """

    def __init__(self, d_model, d_ff, activation="silu", match_params=True,
                 multiple_of=8):
        super().__init__()
        if match_params:
            d_ff = int(2 * d_ff / 3)
            # round up to a hardware-friendly multiple (LLaMA uses 256)
            d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.d_ff = d_ff
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.act = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu}[activation]

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Demo — param-matched comparison
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    d_model, d_ff = 64, 256                 # vanilla 4x expansion
    x = torch.randn(2, 6, d_model)

    vanilla = VanillaFFN(d_model, d_ff)
    swiglu_naive = GLUFFN(d_model, d_ff, "silu", match_params=False)
    swiglu_match = GLUFFN(d_model, d_ff, "silu", match_params=True)
    geglu = GLUFFN(d_model, d_ff, "gelu", match_params=True)

    def n(m): return sum(p.numel() for p in m.parameters())

    print(f"d_model={d_model}, vanilla d_ff={d_ff}")
    print(f"{'FFN':18} {'hidden':>7} {'params':>8}")
    print(f"{'Vanilla GELU':18} {d_ff:>7} {n(vanilla):>8}")
    print(f"{'SwiGLU (naive)':18} {swiglu_naive.d_ff:>7} {n(swiglu_naive):>8}"
          f"   (3 matrices -> ~1.5x params)")
    print(f"{'SwiGLU (2/3)':18} {swiglu_match.d_ff:>7} {n(swiglu_match):>8}"
          f"   <- param-matched to vanilla")
    print(f"{'GeGLU  (2/3)':18} {geglu.d_ff:>7} {n(geglu):>8}")

    # the 2/3 rule brings the gated FFN back near the vanilla budget
    ratio = n(swiglu_match) / n(vanilla)
    print(f"\nparam-matched SwiGLU / vanilla = {ratio:.2f}x (close to 1.0)")
    assert 0.7 < ratio < 1.15, "2/3 trick should roughly match params"

    for m, name in [(vanilla, "vanilla"), (swiglu_match, "swiglu"),
                    (geglu, "geglu")]:
        y = m(x)
        assert y.shape == x.shape
    print(f"\nall FFNs map {tuple(x.shape)} -> {tuple(y.shape)}.  PASS")


if __name__ == "__main__":
    demo()

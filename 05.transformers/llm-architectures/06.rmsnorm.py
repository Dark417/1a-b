"""
RMSNorm — Root Mean Square Layer Normalization
==============================================
LayerNorm normalizes activations to zero mean and unit variance, then rescales
and shifts (γ, β). RMSNorm drops the mean-centering entirely: it only rescales by
the root-mean-square of the activations. The hypothesis (and empirical finding)
is that the *re-centering* part of LayerNorm contributes little; the *re-scaling*
is what stabilizes training. Removing the mean (and usually β) makes the op
cheaper and slightly faster while matching quality.

    LayerNorm(x) = (x - μ) / sqrt(σ² + ε) · γ + β,   μ,σ² over the feature dim
    RMSNorm(x)   = x / sqrt(mean(x²) + ε) · γ        (no μ, no β)

PRE-NORM vs POST-NORM. Where you place the norm matters for deep stacks:
    Post-norm (original Transformer): x -> Sublayer -> Add -> Norm.
        Strong signal but unstable past ~12 layers without careful warmup.
    Pre-norm (GPT-2, LLaMA, ...):     x -> Norm -> Sublayer -> Add.
        The residual path is "clean" (an identity highway), so gradients flow to
        the bottom unimpeded — this is why very deep LLMs train stably. Trade-off:
        representations can grow along depth, so a final norm before the head is
        added.

DEEPNORM. A way to keep POST-norm stable at extreme depth (1000 layers): scale
the residual branch up by α before adding, and down-scale the sublayer weights by
β at init:  x_{l+1} = Norm(α·x_l + Sublayer(x_l)),  with α=(2N)^{1/4}, β=(8N)^{-1/4}
for an N-layer encoder. This bounds the update magnitude so post-norm no longer
blows up. (Wang et al. 2022, "DeepNet".)

What introduced it: Zhang & Sennrich (2019). RMSNorm is the default in LLaMA,
Mistral, Qwen, DeepSeek, Gemma, T5 (a variant), and most modern LLMs.

References:
    - Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
    - Ba, Kiros & Hinton (2016), "Layer Normalization"
    - Xiong et al. (2020), "On Layer Normalization in the Transformer Architecture"
    - Wang et al. (2022), "DeepNet: Scaling Transformers to 1,000 Layers"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * gamma   (no mean subtraction, no bias)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma (scale only)

    def forward(self, x):
        # compute in float32 for stability (LLaMA does this), then cast back
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def deepnorm_alpha_beta(n_layers, kind="encoder"):
    """DeepNorm residual scale alpha and init scale beta for stable post-norm."""
    N = n_layers
    if kind == "encoder":
        return (2 * N) ** 0.25, (8 * N) ** -0.25
    # decoder: alpha = (3N)^{1/4}-ish per DeepNet paper recipe
    return (3 * N) ** 0.25, (12 * N) ** -0.25


class PreNormBlock(nn.Module):
    """x -> Norm -> sublayer -> +residual. Clean residual highway (GPT-2/LLaMA)."""

    def __init__(self, dim, sublayer):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.sublayer = sublayer

    def forward(self, x):
        return x + self.sublayer(self.norm(x))


class PostNormBlock(nn.Module):
    """x -> sublayer -> +residual -> Norm. Optional DeepNorm alpha on residual."""

    def __init__(self, dim, sublayer, alpha=1.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.sublayer = sublayer
        self.alpha = alpha

    def forward(self, x):
        return self.norm(self.alpha * x + self.sublayer(x))


# ---------------------------------------------------------------------------
# Demo — numerical check vs LayerNorm behaviour
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    dim = 8
    x = torch.randn(3, dim) * 2 + 5            # non-zero mean, non-unit var

    rms = RMSNorm(dim)
    ln = nn.LayerNorm(dim)

    yr = rms(x)
    yl = ln(x)
    print(f"input mean per row : {x.mean(-1).tolist()}")
    print(f"RMSNorm  out mean  : {[round(v,3) for v in yr.mean(-1).tolist()]}"
          f"  (NOT centered -> nonzero)")
    print(f"LayerNorm out mean : {[round(v,3) for v in yl.mean(-1).tolist()]}"
          f"  (centered -> ~0)")

    # RMSNorm fixes the scale: each row's RMS becomes ~1 (gamma=1 at init)
    rms_of_out = yr.pow(2).mean(-1).sqrt()
    print(f"\nRMSNorm output RMS per row: {[round(v,3) for v in rms_of_out.tolist()]}"
          f"  (~1.0)")
    assert torch.allclose(rms_of_out, torch.ones(3), atol=1e-3)

    # KEY EQUIVALENCE: on already-centered input, RMSNorm == LayerNorm (no β).
    xc = x - x.mean(-1, keepdim=True)
    yr2 = (xc / xc.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt())
    ln_nobias = nn.LayerNorm(dim, elementwise_affine=False)
    yl2 = ln_nobias(xc)
    print(f"\nOn centered input, RMSNorm ≈ LayerNorm(no affine): "
          f"max diff {(yr2 - yl2).abs().max():.2e}")
    assert torch.allclose(yr2, yl2, atol=1e-3)

    # DeepNorm scaling factors for a deep post-norm stack.
    a, b = deepnorm_alpha_beta(100, "encoder")
    print(f"\nDeepNorm @100 layers: residual alpha={a:.3f}, init beta={b:.3f}")

    # Pre-norm block: clean residual highway preserves identity at init.
    blk = PreNormBlock(dim, nn.Linear(dim, dim))
    nn.init.zeros_(blk.sublayer.weight); nn.init.zeros_(blk.sublayer.bias)
    z = torch.randn(2, dim)
    assert torch.allclose(blk(z), z), "pre-norm identity at zero-init sublayer"
    print("Pre-norm residual highway is identity at zero-init sublayer.  PASS")


if __name__ == "__main__":
    demo()

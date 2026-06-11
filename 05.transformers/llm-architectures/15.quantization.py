"""
Quantization — Low-Bit Weights for Inference
============================================
LLM weights are usually fp16/bf16 (2 bytes each). Quantization stores them in
fewer bits (int8, int4) to cut memory and bandwidth — the dominant cost of
inference. The idea: map a float range to a small integer grid, keeping a scale
(and maybe a zero-point) so you can DEQUANTIZE back approximately.

SYMMETRIC vs ASYMMETRIC (per the affine quantization scheme):
    Symmetric (int8 in [-127,127]):  s = max|w| / 127,   q = round(w / s)
                                     dequant: ŵ = q · s        (zero maps to 0)
    Asymmetric:  uses the true [min,max]; q = round(w/s) + z with a zero-point z,
                 better when the distribution is one-sided (e.g. post-ReLU acts).
        s = (max - min) / (2^b - 1),  z = round(-min / s)

GROUP-WISE (int4). A single scale for a whole tensor wastes precision when a few
outliers stretch the range. Group quantization splits each row into small GROUPS
(e.g. 128 weights) with their own scale (and zero) — far lower error for int4,
at the cost of storing one scale per group. This is what GGUF's "Q4_K", GPTQ, and
AWQ all do under the hood.

THE METHODS (conceptual):
    - bitsandbytes (LLM.int8 / NF4): on-the-fly quantization; LLM.int8 keeps
      outlier activation channels in fp16 (mixed-precision matmul); NF4 (QLoRA)
      uses a 4-bit "NormalFloat" grid matched to a normal distribution.
    - GPTQ: one-shot POST-TRAINING quantization that minimizes the layer-wise
      output error using approximate second-order (Hessian) information, quantizing
      weights column-by-column with error compensation. Great for int4/int3.
    - AWQ (Activation-aware Weight Quantization): protects the ~1% of weight
      channels that matter most (identified via activation magnitudes) by scaling
      them before quantization — no backprop needed, fast, accurate at int4.
    - GGUF: the llama.cpp file FORMAT + a family of k-quant schemes (Q4_K_M, Q5_K,
      Q6_K, ...) using group-wise quant with extra super-block scales; CPU-friendly.

Quantization-aware training (QAT) instead simulates quantization in the forward
pass during training; the above are all POST-TRAINING (PTQ) — no/low retraining.

References:
    - Dettmers et al. (2022), "LLM.int8(): 8-bit Matrix Multiplication"
    - Dettmers et al. (2023), "QLoRA: Efficient Finetuning of Quantized LLMs" (NF4)
    - Frantar et al. (2023), "GPTQ: Accurate Post-Training Quantization"
    - Lin et al. (2023), "AWQ: Activation-aware Weight Quantization"
"""

from __future__ import annotations

import torch
import torch.nn as nn


def quantize_symmetric_int8(w):
    """Per-tensor symmetric int8. Returns (q, scale). dequant = q * scale."""
    s = w.abs().max() / 127.0
    s = s.clamp_min(1e-8)
    q = torch.round(w / s).clamp(-127, 127).to(torch.int8)
    return q, s


def dequantize_symmetric(q, s):
    return q.float() * s


def quantize_asymmetric_int8(w):
    """Per-tensor asymmetric (affine) uint8. Returns (q, scale, zero_point)."""
    lo, hi = w.min(), w.max()
    s = (hi - lo) / 255.0
    s = s.clamp_min(1e-8)
    z = torch.round(-lo / s)                          # zero-point in [0,255]
    q = torch.round(w / s + z).clamp(0, 255).to(torch.uint8)
    return q, s, z


def dequantize_asymmetric(q, s, z):
    return (q.float() - z) * s


def quantize_int4_group(w, group_size=64):
    """
    Group-wise symmetric int4 (range [-7,7]) along the last dim. Each group of
    `group_size` weights gets its own scale. Returns (q, scales, group_size).
    """
    *lead, n = w.shape
    pad = (-n) % group_size
    if pad:
        w = torch.cat([w, torch.zeros(*lead, pad)], dim=-1)
    g = w.reshape(*lead, -1, group_size)              # (..., n_groups, group_size)
    s = g.abs().amax(-1, keepdim=True) / 7.0
    s = s.clamp_min(1e-8)
    q = torch.round(g / s).clamp(-7, 7).to(torch.int8)
    return q, s, n                                    # keep original n to unpad


def dequantize_int4_group(q, s, n):
    g = q.float() * s
    flat = g.reshape(*q.shape[:-2], -1)[..., :n]
    return flat


class QuantizedLinear(nn.Module):
    """A Linear whose weights are stored int4 group-quantized and dequantized
    on the fly for the matmul (the simplest 'weight-only' quant inference path)."""

    def __init__(self, linear: nn.Linear, group_size=64):
        super().__init__()
        w = linear.weight.data                         # (out, in)
        self.q, self.s, self.n = quantize_int4_group(w, group_size)
        self.bias = linear.bias
        self.out_features, self.in_features = w.shape

    def forward(self, x):
        w = dequantize_int4_group(self.q, self.s, self.n)
        return x @ w.t() + (self.bias if self.bias is not None else 0)


# ---------------------------------------------------------------------------
# Demo — quantize a linear layer, show MSE + size reduction
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    w = torch.randn(64, 256) * 0.5                     # a weight matrix

    # --- int8 symmetric vs asymmetric ---
    q8, s8 = quantize_symmetric_int8(w)
    w8 = dequantize_symmetric(q8, s8)
    qa, sa, za = quantize_asymmetric_int8(w)
    wa = dequantize_asymmetric(qa, sa, za)
    mse8 = (w - w8).pow(2).mean().item()
    msea = (w - wa).pow(2).mean().item()
    print("Dequant error (MSE), lower = better:")
    print(f"  int8 symmetric : {mse8:.3e}")
    print(f"  int8 asymmetric: {msea:.3e}")

    # --- int4: per-tensor vs group-wise (group should win) ---
    qg, sg, n = quantize_int4_group(w, group_size=64)
    wg = dequantize_int4_group(qg, sg, n)
    qg1, sg1, n1 = quantize_int4_group(w, group_size=256)   # whole row = 1 group
    wg1 = dequantize_int4_group(qg1, sg1, n1)
    mse4_grp = (w - wg).pow(2).mean().item()
    mse4_one = (w - wg1).pow(2).mean().item()
    print(f"\n  int4 per-row (1 group)   : {mse4_one:.3e}")
    print(f"  int4 group=64            : {mse4_grp:.3e}  (groups -> less error)")
    assert mse4_grp < mse4_one, "group quant should beat single-scale int4"

    # --- size reduction ---
    fp16 = w.numel() * 2
    i8 = w.numel() * 1
    n_groups = (w.shape[1] + 63) // 64
    i4 = w.numel() * 0.5 + w.shape[0] * n_groups * 2   # 4-bit + fp16 group scales
    print(f"\nStorage for a {tuple(w.shape)} weight:")
    print(f"  fp16     : {fp16:6d} B  (1.00x)")
    print(f"  int8     : {i8:6d} B  ({fp16/i8:.1f}x smaller)")
    print(f"  int4 grp : {int(i4):6d} B  ({fp16/i4:.1f}x smaller, incl. scales)")

    # --- end-to-end: quantized Linear stays close to fp ---
    lin = nn.Linear(256, 64)
    qlin = QuantizedLinear(lin, group_size=64)
    x = torch.randn(4, 256)
    rel = (lin(x) - qlin(x)).norm() / lin(x).norm()
    print(f"\nQuantizedLinear (int4 group) output rel-error: {rel:.3%}")
    assert rel < 0.1, "int4 group-quant linear should stay within ~10%"
    print("Quantize -> dequantize with small error and big size win.  PASS")


if __name__ == "__main__":
    demo()

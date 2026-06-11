"""
MobileNet — depthwise-separable convolutions for cheap CNNs
===========================================================
MobileNet (Howard et al., 2017) factorizes a standard convolution into two much
cheaper steps: a **depthwise** conv that filters each input channel
independently (spatial mixing, no cross-channel mixing), followed by a
**pointwise** 1x1 conv that combines channels. This **depthwise-separable**
factorization cuts parameters and FLOPs by roughly $1/N + 1/k^2$ versus a full
conv, with little accuracy loss — the key to running CNNs on phones. This file
implements the separable block from scratch (NumPy parameter/FLOP accounting),
the idiomatic PyTorch block, and a small MobileNet.

Variants implemented here:
    - depthwise_separable_savings(): from-scratch param/FLOP comparison vs a
      standard conv (the core teaching arithmetic)
    - DepthwiseSeparableConv: depthwise (grouped) conv + pointwise 1x1, each
      with BN + ReLU (MobileNet-v1 block)
    - TinyMobileNet: a stem + a stack of separable blocks + classifier head
    - width_multiplier alpha (thins every layer, the MobileNet hyperparameter)

Training techniques demonstrated:
    - Factorized convolutions (depthwise + pointwise) for efficiency
    - Parameter / FLOP budgeting (the separable-conv math)
    - BatchNorm + ReLU after each conv

References:
    - Howard et al. (2017), "MobileNets: Efficient Convolutional Neural Networks
      for Mobile Vision Applications"
    - Sifre & Mallat (2014); Chollet (2017, Xception) — separable convolutions
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 0. Teaching snippet (NumPy/arithmetic): standard vs depthwise-separable cost
# ---------------------------------------------------------------------------
def depthwise_separable_savings(c_in: int = 32, c_out: int = 64, k: int = 3,
                                hw: int = 16) -> dict:
    r"""Parameters and FLOPs of a standard k x k conv vs its depthwise-separable
    factorization, mapping (c_in, hw, hw) -> (c_out, hw, hw) with 'same' padding.

    Standard conv:
        params = k*k * c_in * c_out
        flops  = k*k * c_in * c_out * (hw*hw)            # one MAC per output px

    Depthwise-separable = depthwise (k x k, per-channel) + pointwise (1x1):
        depthwise params = k*k * c_in
        pointwise params = c_in * c_out
        depthwise flops  = k*k * c_in * (hw*hw)
        pointwise flops  = c_in * c_out * (hw*hw)

    The ratio is the famous MobileNet formula:
        separable / standard = 1/c_out + 1/k^2.
    """
    std_params = k * k * c_in * c_out
    dw_params = k * k * c_in
    pw_params = c_in * c_out
    sep_params = dw_params + pw_params

    px = hw * hw
    std_flops = std_params * px
    sep_flops = (dw_params + pw_params) * px

    ratio = 1.0 / c_out + 1.0 / (k * k)
    return {
        "standard_params": std_params,
        "depthwise_params": dw_params,
        "pointwise_params": pw_params,
        "separable_params": sep_params,
        "standard_flops": std_flops,
        "separable_flops": sep_flops,
        "param_ratio": sep_params / std_params,
        "flop_ratio": sep_flops / std_flops,
        "formula_ratio": ratio,            # 1/N + 1/k^2 (should match the above)
    }


# ---------------------------------------------------------------------------
# 1. PyTorch implementation — depthwise-separable block + a small MobileNet
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DepthwiseSeparableConv(nn.Module):
    """MobileNet-v1 block: a depthwise conv (groups=in_c -> one filter per input
    channel, spatial mixing only) followed by a pointwise 1x1 conv (cross-channel
    mixing). BN + ReLU after each, as in the paper.

    `stride` is applied in the depthwise step (downsampling). Setting groups=in_c
    is what makes the conv 'depthwise': each input channel is convolved by its own
    k x k kernel, with NO summation across channels."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride,
                                   padding=1, groups=in_c, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_c)
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.depthwise(x)))    # spatial, per-channel
        x = self.relu(self.bn_pw(self.pointwise(x)))    # mix channels (1x1)
        return x


class TinyMobileNet(nn.Module):
    """A miniature MobileNet-v1: a standard-conv stem, a stack of
    depthwise-separable blocks (some with stride 2 to downsample), global average
    pooling, and a linear classifier. `alpha` is the width multiplier that thins
    every layer uniformly (the MobileNet efficiency knob)."""

    def __init__(self, in_c: int = 1, n_classes: int = 10, alpha: float = 1.0):
        super().__init__()
        def w(c: int) -> int:                # apply width multiplier, keep >= 1
            return max(1, int(c * alpha))

        self.stem = nn.Sequential(
            nn.Conv2d(in_c, w(8), 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(w(8)), nn.ReLU(inplace=True),
        )
        # (in, out, stride) for each separable block
        cfg = [(w(8), w(16), 1), (w(16), w(32), 2), (w(32), w(32), 1)]
        self.blocks = nn.Sequential(
            *[DepthwiseSeparableConv(i, o, s) for i, o, s in cfg]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg[-1][1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 2. Demo — FAST on CPU
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)        # tiny ops: avoid thread-thrashing on big CPUs
    dev = get_device()

    # --- (a) The depthwise-separable saving (arithmetic, from scratch) --------
    s = depthwise_separable_savings(c_in=32, c_out=64, k=3, hw=16)
    print("Standard 3x3 conv (32->64) vs depthwise-separable, on a 16x16 map:")
    print(f"  standard  : params = {s['standard_params']:>8,}  "
          f"flops = {s['standard_flops']:>12,}")
    print(f"  separable : params = {s['separable_params']:>8,}  "
          f"flops = {s['separable_flops']:>12,}  "
          f"(depthwise {s['depthwise_params']} + pointwise {s['pointwise_params']})")
    print(f"  -> ratio = {s['param_ratio']:.3f} params, {s['flop_ratio']:.3f} flops "
          f"= ~{s['param_ratio']:.1%} of the cost")
    print(f"  -> matches the formula 1/N + 1/k^2 = {s['formula_ratio']:.3f}")

    # --- (b) One standard conv vs one separable block: param count + shapes ---
    x = torch.randn(8, 32, 16, 16, device=dev)
    std = nn.Conv2d(32, 64, 3, padding=1, bias=False).to(dev)
    sep = DepthwiseSeparableConv(32, 64, stride=1).to(dev)
    p_std = sum(p.numel() for p in std.parameters())
    p_sep = sum(p.numel() for p in sep.parameters())   # incl. BN params
    print(f"\nOn {tuple(x.shape)}:")
    print(f"  standard Conv2d        -> {tuple(std(x).shape)}, params = {p_std:,}")
    print(f"  DepthwiseSeparableConv -> {tuple(sep(x).shape)}, params = {p_sep:,}")

    # --- (c) Build TinyMobileNet (alpha=1 and 0.5), count params, train steps -
    x1 = torch.randn(8, 1, 16, 16, device=dev)
    yb = torch.randint(0, 10, (8,), device=dev)
    for alpha in (1.0, 0.5):
        net = TinyMobileNet(in_c=1, n_classes=10, alpha=alpha).to(dev)
        out = net(x1)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"\nTinyMobileNet(alpha={alpha}): {tuple(x1.shape)} -> "
              f"{tuple(out.shape)}, params = {n_params:,}")
        if alpha == 1.0:
            opt = torch.optim.Adam(net.parameters(), lr=1e-2)
            loss_fn = nn.CrossEntropyLoss()
            losses = []
            for _ in range(20):
                opt.zero_grad()
                loss = loss_fn(net(x1), yb)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"  training loss: {losses[0]:.3f} -> {losses[-1]:.3f} (going down)")


if __name__ == "__main__":
    demo()

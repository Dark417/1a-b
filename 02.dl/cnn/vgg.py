"""
VGG — very deep networks from stacked 3x3 convolutions
======================================================
VGG (Simonyan & Zisserman, 2014) made one design choice ruthlessly uniform:
build the whole network out of tiny **3x3 convolutions** (stride 1, pad 1) and
2x2 max-pools. The key insight is that a *stack* of small filters has the same
**receptive field** as a single large filter but with **fewer parameters** and
**more nonlinearity** (a ReLU between each conv). This file builds VGG-style
blocks, a small "VGG-mini" net, and quantifies the 3x3-vs-5x5 trade-off.

Variants implemented here:
    - vgg_block(n_convs): a stack of 3x3 conv-ReLU layers + 2x2 max-pool
    - VGGMini: a tiny VGG (config-driven, like the real "VGG-A..E" tables)
    - Receptive-field / parameter-count comparison: two 3x3 vs one 5x5
      (and three 3x3 vs one 7x7) — a from-scratch arithmetic teaching snippet

Training techniques demonstrated:
    - Depth via repeated 3x3 stacks (the VGG philosophy)
    - Receptive-field arithmetic & parameter budgeting
    - Weight init (He) — see training-techniques/README.md

References:
    - Simonyan & Zisserman (2014), "Very Deep Convolutional Networks for
      Large-Scale Image Recognition" (VGG)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 0. Teaching snippet (pure arithmetic): why stacked 3x3 beats one big conv
# ---------------------------------------------------------------------------
def receptive_field(kernels: list[int], strides: list[int] | None = None) -> int:
    r"""Receptive field of a stack of conv layers (all stride 1 unless given).

    For layer l with kernel k_l and stride s_l, the RF grows as
        RF_l = RF_{l-1} + (k_l - 1) * prod_{j<l} s_j.
    With all strides 1 this is RF = 1 + sum_l (k_l - 1). So two 3x3 convs give
    RF = 1 + 2 + 2 = 5 (a 5x5 field); three give 7.
    """
    if strides is None:
        strides = [1] * len(kernels)
    rf, jump = 1, 1
    for k, s in zip(kernels, strides):
        rf += (k - 1) * jump
        jump *= s
    return rf


def conv_params(k: int, c_in: int, c_out: int, bias: bool = True) -> int:
    """Weight count of one conv layer: k*k*c_in*c_out (+ c_out biases)."""
    return k * k * c_in * c_out + (c_out if bias else 0)


def small_vs_large_filter_table(channels: int = 64) -> dict:
    r"""Compare equal-receptive-field designs for C in/out channels (no bias).

    Two 3x3 convs vs one 5x5:    2 * 9 C^2  =  18 C^2   <  25 C^2.
    Three 3x3 convs vs one 7x7:  3 * 9 C^2  =  27 C^2   <  49 C^2.
    The 3x3 stack also inserts an extra ReLU per layer -> more nonlinearity.
    """
    C = channels
    rows = {
        "two_3x3":  (receptive_field([3, 3]),   2 * conv_params(3, C, C, bias=False)),
        "one_5x5":  (receptive_field([5]),          conv_params(5, C, C, bias=False)),
        "three_3x3": (receptive_field([3, 3, 3]), 3 * conv_params(3, C, C, bias=False)),
        "one_7x7":  (receptive_field([7]),          conv_params(7, C, C, bias=False)),
    }
    return rows


# ---------------------------------------------------------------------------
# 1. PyTorch implementation — VGG blocks + a small configurable VGG
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def vgg_block(n_convs: int, in_c: int, out_c: int) -> nn.Sequential:
    """`n_convs` stacked 3x3 conv-ReLU layers (pad 1, so spatial size is kept),
    followed by a 2x2 max-pool that halves the spatial resolution.

    This is the repeating motif of every VGG configuration (A through E differ
    only in how many convs sit in each block)."""
    layers: list[nn.Module] = []
    c = in_c
    for _ in range(n_convs):
        layers.append(nn.Conv2d(c, out_c, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        c = out_c
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# A tiny "VGG" config: list of (n_convs, out_channels) per block. The real VGG-16
# is ((2,64),(2,128),(3,256),(3,512),(3,512)); we shrink it so it runs on 16x16.
VGG_MINI_CFG = [(2, 8), (2, 16)]


class VGGMini(nn.Module):
    """A miniature VGG: stacked 3x3-conv blocks (each halving resolution via
    max-pool), then a small classifier head. Config-driven exactly like the
    VGG-A..E table in the paper."""

    def __init__(self, cfg=VGG_MINI_CFG, in_c: int = 1, n_classes: int = 10):
        super().__init__()
        blocks: list[nn.Module] = []
        c = in_c
        for n_convs, out_c in cfg:
            blocks.append(vgg_block(n_convs, c, out_c))
            c = out_c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)        # makes head input-size agnostic
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, 32), nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )
        self._he_init()

    def _he_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(self.pool(x))


# ---------------------------------------------------------------------------
# 2. Demo — FAST on CPU
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)        # tiny ops: avoid thread-thrashing on big CPUs
    dev = get_device()

    # --- (a) Receptive-field & parameter arithmetic: 3x3 stacks vs big convs --
    print("Equal-receptive-field designs (C=64 channels, weights only):")
    for name, (rf, p) in small_vs_large_filter_table(64).items():
        print(f"  {name:10s}: receptive field = {rf}x{rf}, params = {p:>8,}")
    two, one5 = small_vs_large_filter_table(64)["two_3x3"], small_vs_large_filter_table(64)["one_5x5"]
    print(f"  -> two 3x3 match a 5x5 RF with {one5[1] / two[1]:.2f}x FEWER... "
          f"actually {1 - two[1] / one5[1]:.0%} fewer params, + an extra ReLU.")

    # --- (b) Build VGGMini, show shapes & per-layer param counts --------------
    x = torch.randn(8, 1, 16, 16, device=dev)
    yb = torch.randint(0, 10, (8,), device=dev)
    net = VGGMini(in_c=1, n_classes=10).to(dev)
    out = net(x)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\nVGGMini: input {tuple(x.shape)} -> output {tuple(out.shape)}, "
          f"params = {n_params:,}")
    # trace feature-map shapes block by block
    h = x
    for i, blk in enumerate(net.features):
        h = blk(h)
        print(f"  after block {i}: {tuple(h.shape)}")

    # --- (c) A handful of training steps; loss should drop --------------------
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn(net(x), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"\ntraining loss: {losses[0]:.3f} -> {losses[-1]:.3f} (going down)")


if __name__ == "__main__":
    demo()

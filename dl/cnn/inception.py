"""
Inception / GoogLeNet — multi-branch convolutions + 1x1 bottlenecks
===================================================================
Instead of picking one filter size, the **Inception module** (Szegedy et al.,
2014) computes several in parallel — 1x1, 3x3, 5x5, and a pooling branch — and
**concatenates** their outputs along channels, letting the network choose the
scale per feature. The catch: a naive 5x5 branch over many input channels is
expensive. The fix is the **1x1 "bottleneck" convolution**, which cheaply reduces
channel depth *before* the costly spatial convs. This file builds the naive and
dimension-reduced modules, a tiny GoogLeNet-style net, and quantifies the 1x1
parameter savings.

Variants implemented here:
    - NaiveInception: parallel 1x1 / 3x3 / 5x5 / pool branches, concatenated
    - Inception (with 1x1 bottlenecks before the 3x3 and 5x5 branches)
    - TinyGoogLeNet: a small stack of Inception modules + classifier head
    - A from-scratch arithmetic comparison of params with vs without bottlenecks

Training techniques demonstrated:
    - Multi-scale feature extraction via branch concatenation
    - 1x1 convolutions as cheap channel-wise dimensionality reduction
    - Parameter/FLOP budgeting (the bottleneck math)

References:
    - Szegedy et al. (2014), "Going Deeper with Convolutions" (GoogLeNet,
      Inception v1)
    - Lin et al. (2013), "Network in Network" (the 1x1 convolution idea)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 0. Teaching snippet (pure arithmetic): how much do 1x1 bottlenecks save?
# ---------------------------------------------------------------------------
def conv_params(k: int, c_in: int, c_out: int) -> int:
    """Weight count of one conv layer (no bias): k*k*c_in*c_out."""
    return k * k * c_in * c_out


def inception_param_comparison(c_in: int = 192, reduce_3: int = 96,
                               reduce_5: int = 16,
                               out_1: int = 64, out_3: int = 128,
                               out_5: int = 32, out_pool: int = 32) -> dict:
    r"""Parameters of one Inception module with vs without 1x1 bottlenecks.

    Default channel counts are GoogLeNet's "inception (3a)" module. The 5x5 and
    3x3 branches are the expensive ones; a 1x1 conv first squeezes c_in down to a
    small `reduce` width, so the spatial conv runs in a cheap subspace.

    naive 5x5 branch:        5*5 * c_in * out_5
    bottlenecked 5x5 branch: 1*1 * c_in * reduce_5  +  5*5 * reduce_5 * out_5
    """
    naive = {
        "1x1":  conv_params(1, c_in, out_1),
        "3x3":  conv_params(3, c_in, out_3),
        "5x5":  conv_params(5, c_in, out_5),
        "pool_proj": conv_params(1, c_in, out_pool),
    }
    reduced = {
        "1x1":  conv_params(1, c_in, out_1),
        "3x3_reduce": conv_params(1, c_in, reduce_3),
        "3x3":  conv_params(3, reduce_3, out_3),
        "5x5_reduce": conv_params(1, c_in, reduce_5),
        "5x5":  conv_params(5, reduce_5, out_5),
        "pool_proj": conv_params(1, c_in, out_pool),
    }
    return {
        "naive_total": sum(naive.values()),
        "reduced_total": sum(reduced.values()),
        "naive": naive,
        "reduced": reduced,
    }


# ---------------------------------------------------------------------------
# 1. PyTorch implementation — Inception modules + a tiny GoogLeNet
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def conv_bn_relu(in_c: int, out_c: int, k: int, padding: int = 0) -> nn.Sequential:
    """Conv -> BN -> ReLU, the standard GoogLeNet basic conv unit."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, padding=padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class NaiveInception(nn.Module):
    """Parallel 1x1 / 3x3 / 5x5 / (3x3 max-pool) branches, concatenated on the
    channel axis. All branches use padding so spatial size is preserved, so the
    concatenation aligns. This version has NO bottleneck -> expensive."""

    def __init__(self, in_c: int, out_1: int, out_3: int, out_5: int,
                 out_pool: int):
        super().__init__()
        self.b1 = conv_bn_relu(in_c, out_1, 1)
        self.b3 = conv_bn_relu(in_c, out_3, 3, padding=1)
        self.b5 = conv_bn_relu(in_c, out_5, 5, padding=2)
        self.bpool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_bn_relu(in_c, out_pool, 1),
        )
        self.out_channels = out_1 + out_3 + out_5 + out_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bpool(x)], dim=1)


class Inception(nn.Module):
    """GoogLeNet Inception module WITH 1x1 dimension-reduction bottlenecks.

    The 3x3 and 5x5 branches first squeeze the input channels with a cheap 1x1
    conv (`reduce_3`, `reduce_5`) before the expensive spatial conv. The pool
    branch projects with a 1x1 too. Output channels = out_1+out_3+out_5+out_pool.
    """

    def __init__(self, in_c: int, out_1: int, reduce_3: int, out_3: int,
                 reduce_5: int, out_5: int, out_pool: int):
        super().__init__()
        self.b1 = conv_bn_relu(in_c, out_1, 1)
        self.b3 = nn.Sequential(
            conv_bn_relu(in_c, reduce_3, 1),          # 1x1 bottleneck (reduce)
            conv_bn_relu(reduce_3, out_3, 3, padding=1),
        )
        self.b5 = nn.Sequential(
            conv_bn_relu(in_c, reduce_5, 1),          # 1x1 bottleneck (reduce)
            conv_bn_relu(reduce_5, out_5, 5, padding=2),
        )
        self.bpool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_bn_relu(in_c, out_pool, 1),          # 1x1 projection
        )
        self.out_channels = out_1 + out_3 + out_5 + out_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bpool(x)], dim=1)


class TinyGoogLeNet(nn.Module):
    """A miniature GoogLeNet: a conv stem, two Inception modules, global average
    pooling, and a linear classifier. Channel counts shrunk so it runs on tiny
    inputs in seconds."""

    def __init__(self, in_c: int = 1, n_classes: int = 10):
        super().__init__()
        self.stem = conv_bn_relu(in_c, 16, 3, padding=1)
        # tiny analogues of inception (3a)/(3b): in=16 -> 32 -> 48 channels
        self.inc1 = Inception(16, out_1=8, reduce_3=8, out_3=12,
                              reduce_5=2, out_5=4, out_pool=8)   # 8+12+4+8 = 32
        self.inc2 = Inception(self.inc1.out_channels,
                              out_1=12, reduce_3=12, out_3=16,
                              reduce_5=4, out_5=8, out_pool=12)   # 12+16+8+12 = 48
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.inc2.out_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
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

    # --- (a) The 1x1-bottleneck parameter saving (GoogLeNet inception 3a) -----
    cmp = inception_param_comparison()
    print("Inception module (3a) parameters, naive vs 1x1-bottlenecked:")
    print(f"  naive   (no 1x1 reduce): {cmp['naive_total']:>10,} weights")
    print(f"  reduced (with 1x1)     : {cmp['reduced_total']:>10,} weights")
    print(f"  -> bottlenecks cut params to "
          f"{cmp['reduced_total'] / cmp['naive_total']:.0%} "
          f"({1 - cmp['reduced_total'] / cmp['naive_total']:.0%} fewer).")

    # --- (b) Naive vs bottlenecked module on a tiny input: shapes + params ----
    x = torch.randn(8, 16, 16, 16, device=dev)
    naive = NaiveInception(16, out_1=8, out_3=12, out_5=4, out_pool=8).to(dev)
    redu = Inception(16, out_1=8, reduce_3=8, out_3=12,
                     reduce_5=2, out_5=4, out_pool=8).to(dev)
    pn = sum(p.numel() for p in naive.parameters())
    pr = sum(p.numel() for p in redu.parameters())
    print(f"\nTiny module on {tuple(x.shape)}:")
    print(f"  NaiveInception -> {tuple(naive(x).shape)}, params = {pn:,}")
    print(f"  Inception(1x1) -> {tuple(redu(x).shape)}, params = {pr:,}")

    # --- (c) Build TinyGoogLeNet, count params, train a few steps -------------
    x1 = torch.randn(8, 1, 16, 16, device=dev)
    yb = torch.randint(0, 10, (8,), device=dev)
    net = TinyGoogLeNet(in_c=1, n_classes=10).to(dev)
    out = net(x1)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\nTinyGoogLeNet: {tuple(x1.shape)} -> {tuple(out.shape)}, "
          f"params = {n_params:,}")
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

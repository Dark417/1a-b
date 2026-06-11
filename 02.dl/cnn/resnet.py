"""
ResNet — Residual Networks
==========================
Plain deep CNNs *degrade*: past a certain depth, adding layers makes training
error go **up**, not because of overfitting but because the optimizer struggles
to push signal (and gradients) through dozens of nonlinear layers. ResNet's fix
is almost embarrassingly simple: instead of learning a mapping $H(x)$ directly,
each block learns the **residual** $F(x)=H(x)-x$ and outputs $x + F(x)$. The
identity shortcut gives every layer a direct path, so gradients flow even in
100+ layer nets.

Variants implemented here:
    - BasicBlock (two 3x3 convs + skip) — ResNet-18/34 style
    - Bottleneck (1x1 -> 3x3 -> 1x1 + skip) — ResNet-50/101/152 style
    - A small ResNet ("TinyResNet") assembled from stages of these blocks
    - A "PlainNet" twin (same depth, NO skips) used for the gradient comparison

Training techniques demonstrated:
    - SKIP / RESIDUAL CONNECTIONS vs VANISHING GRADIENTS — the canonical demo.
      We measure per-layer gradient norms in a deep PLAIN net vs the same net
      with residual connections, and verify the residual gradient identity
      d(out)/d(x) = I + F'(x)  (a guaranteed +1 path).
    - BatchNorm (keeps activations well-scaled) — see training-techniques/README.md.

References:
    - He, Zhang, Ren, Sun (2015), "Deep Residual Learning for Image Recognition"
    - He et al. (2016), "Identity Mappings in Deep Residual Networks"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 0. NumPy teaching block: WHY the residual gradient never vanishes
# ---------------------------------------------------------------------------
def residual_gradient_demo(n_layers: int = 50, dim: int = 16, seed: int = SEED):
    r"""Measure how the input-gradient magnitude survives depth, with vs without
    a skip connection — using a tiny hand-written backward pass (no autograd).

    Each "layer" is z = tanh(x @ W); a residual layer outputs x + z. We backprop
    a unit upstream gradient through `n_layers` and report ||dL/dx_input||.

    Residual identity (the whole point):
        out = x + F(x)  =>  d(out)/d(x) = I + F'(x).
    The "+ I" guarantees a gradient highway of magnitude ~1 regardless of depth.
    """
    rng = np.random.default_rng(seed)
    # Small weights so the plain net's Jacobian product shrinks (vanishing).
    Ws = [rng.normal(0, 0.5, (dim, dim)) for _ in range(n_layers)]

    def run(x0, residual: bool):
        # forward, caching pre-activations
        xs, zs = [x0], []
        x = x0
        for W in Ws:
            z = x @ W
            zs.append(z)
            a = np.tanh(z)
            x = x + a if residual else a
            xs.append(x)
        # backward: start from a unit upstream gradient on the output
        g = np.ones_like(x)
        norms = [np.linalg.norm(g)]
        for l in reversed(range(n_layers)):
            dz = g * (1.0 - np.tanh(zs[l]) ** 2)     # through tanh
            dx_through_F = dz @ Ws[l].T              # through the weight
            g = dx_through_F + g if residual else dx_through_F  # "+ I" skip path
            norms.append(np.linalg.norm(g))
        norms.reverse()                              # index 0 = input layer
        return np.array(norms)

    x0 = rng.normal(0, 1, (4, dim))
    plain = run(x0, residual=False)
    resid = run(x0, residual=True)
    return plain, resid


# ---------------------------------------------------------------------------
# 1. PyTorch implementation — residual building blocks + a small ResNet
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def conv3x3(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


def conv1x1(in_c: int, out_c: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Two stacked 3x3 convs with a skip: out = ReLU( x_proj + F(x) ).

    F(x) = BN(conv3x3( ReLU( BN(conv3x3(x)) ) )). When the block changes spatial
    size (stride>1) or channel count, the shortcut uses a 1x1 conv to match.
    """

    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_c != out_c * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_c, out_c * self.expansion, stride),
                nn.BatchNorm2d(out_c * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)          # the residual addition (the +x path)
        return F.relu(out)


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 with a skip (ResNet-50+). The 1x1 convs squeeze then
    restore channels, so the expensive 3x3 runs in a low-dimensional space —
    far fewer FLOPs/params for the same representational power."""

    expansion = 4

    def __init__(self, in_c: int, mid_c: int, stride: int = 1):
        super().__init__()
        out_c = mid_c * self.expansion
        self.conv1 = conv1x1(in_c, mid_c)             # reduce
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.conv2 = conv3x3(mid_c, mid_c, stride)    # spatial mixing (cheap)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.conv3 = conv1x1(mid_c, out_c)            # restore
        self.bn3 = nn.BatchNorm2d(out_c)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                conv1x1(in_c, out_c, stride), nn.BatchNorm2d(out_c))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class TinyResNet(nn.Module):
    """A small ResNet for tiny images. `block` is BasicBlock or Bottleneck;
    `layers` gives the number of blocks per stage."""

    def __init__(self, block=BasicBlock, layers=(2, 2), in_c=1,
                 base=8, n_classes=10):
        super().__init__()
        self.in_c = base
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(inplace=True))
        self.stage1 = self._make_stage(block, base, layers[0], stride=1)
        self.stage2 = self._make_stage(block, base * 2, layers[1], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base * 2 * block.expansion, n_classes)

    def _make_stage(self, block, mid_c, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(block(self.in_c, mid_c, s))
            self.in_c = mid_c * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 2. PlainNet twin — identical depth, NO skip connections (for the comparison)
# ---------------------------------------------------------------------------
class _PlainConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class PlainNet(nn.Module):
    """Same conv depth as a BasicBlock TinyResNet but WITHOUT the +x shortcuts.
    Used to expose vanishing gradients in the early layers."""

    def __init__(self, depth=16, in_c=1, width=8, n_classes=10):
        super().__init__()
        layers = [_PlainConv(in_c, width)]
        for _ in range(depth - 1):
            layers.append(_PlainConv(width, width))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, n_classes)

    def forward(self, x):
        x = self.features(x)
        return self.fc(self.pool(x).flatten(1))


class ResidualNet(nn.Module):
    """Twin of PlainNet: same conv layers, but every pair is wrapped in a skip."""

    class _ResPair(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.c1 = _PlainConv(width, width)
            self.c2 = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width))

        def forward(self, x):
            return F.relu(x + self.c2(self.c1(x)))   # identity shortcut

    def __init__(self, depth=16, in_c=1, width=8, n_classes=10):
        super().__init__()
        self.stem = _PlainConv(in_c, width)
        self.blocks = nn.Sequential(*[self._ResPair(width)
                                      for _ in range((depth - 1) // 2)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, n_classes)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.fc(self.pool(x).flatten(1))


def layerwise_grad_norms(model: nn.Module, x: torch.Tensor,
                         y: torch.Tensor) -> list[float]:
    """Backprop one batch and return the gradient norm of each conv weight,
    ordered from input layer to output layer."""
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    norms = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.weight.grad is not None:
            norms.append(m.weight.grad.norm().item())
    return norms


# ---------------------------------------------------------------------------
# 3. Demo — FAST on CPU
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)        # tiny ops: avoid thread-thrashing on big CPUs
    dev = get_device()

    # --- (a) NumPy: the residual gradient identity I + F'(x) ----------------
    plain_g, resid_g = residual_gradient_demo(n_layers=50, dim=16)
    print("NumPy 50-layer tanh net — ||grad w.r.t. activations|| (input ... output):")
    print(f"  plain  : input={plain_g[0]:.2e}  output={plain_g[-1]:.2e}"
          f"  -> shrinks {plain_g[-1] / (plain_g[0] + 1e-30):.1e}x toward input")
    print(f"  residual: input={resid_g[0]:.2e}  output={resid_g[-1]:.2e}"
          f"  -> stays O(1) thanks to the + I path")

    # --- (b) PyTorch: deep PLAIN vs RESIDUAL per-layer conv grad norms -------
    x = torch.randn(8, 1, 8, 8, device=dev)
    yb = torch.randint(0, 10, (8,), device=dev)
    plain = PlainNet(depth=16, width=8).to(dev)
    resnet_twin = ResidualNet(depth=16, width=8).to(dev)
    pn = layerwise_grad_norms(plain, x, yb)
    rn = layerwise_grad_norms(resnet_twin, x, yb)
    print("\n16-conv PyTorch nets — first-layer / last-layer conv ||dW||:")
    print(f"  plain   : first={pn[0]:.2e}  last={pn[-1]:.2e}"
          f"  ratio last/first = {pn[-1] / (pn[0] + 1e-30):.1f}x")
    print(f"  residual: first={rn[0]:.2e}  last={rn[-1]:.2e}"
          f"  ratio last/first = {rn[-1] / (rn[0] + 1e-30):.1f}x")
    print("  -> skips keep the early-layer gradient from collapsing.")

    # --- (c) Build the blocks & a small ResNet, count params, train a few steps
    for name, net in [("BasicBlock ResNet", TinyResNet(BasicBlock, (2, 2))),
                      ("Bottleneck ResNet", TinyResNet(Bottleneck, (2, 2)))]:
        net = net.to(dev)
        out = net(x)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"\n{name}: output {tuple(out.shape)}, params = {n_params:,}")
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        losses = []
        for _ in range(15):
            opt.zero_grad()
            loss = loss_fn(net(x), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"  loss: {losses[0]:.3f} -> {losses[-1]:.3f} (going down)")


if __name__ == "__main__":
    demo()

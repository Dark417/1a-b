"""
Vision Transformer (ViT) — images as sequences of patches
=========================================================
ViT shows that the Transformer encoder, with almost no vision-specific inductive
bias, classifies images well once you treat an image as a **sequence of patches**.
Split the image into non-overlapping P x P patches, flatten and linearly project
each into a token, prepend a learnable [CLS] token, add learned positional
embeddings, run a standard Transformer encoder, and classify from the final [CLS]
representation.

Variants implemented here:
    - Patch embedding via an unfolded linear projection (and the Conv2d equivalent)
    - Learnable [CLS] token + learned positional embeddings
    - Pre-norm Transformer encoder (GELU MLP) + classification head

Training techniques demonstrated:
    - PATCHIFY: image -> sequence of patch tokens (the key ViT idea)
    - [CLS] token aggregation for classification
    - Pre-norm residual blocks, LayerNorm, dropout

References:
    - Dosovitskiy et al. (2021), "An Image is Worth 16x16 Words"
    - Vaswani et al. (2017), "Attention Is All You Need"
"""

from __future__ import annotations

import math
import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy: patchify (the one operation with crisp closed-form math)
# ---------------------------------------------------------------------------
def patchify_numpy(img: np.ndarray, patch: int) -> np.ndarray:
    r"""
    Split a (C, H, W) image into a sequence of flattened patches.

    With H = W = N*patch, there are (H/patch)*(W/patch) patches, each of size
    C*patch*patch. We reshape so each row of the output is one flattened patch in
    row-major (raster) order:

        n_patches = (H // P) * (W // P)
        out[k]    = flatten(img[:, i*P:(i+1)*P, j*P:(j+1)*P])

    A learnable matrix E in R^{(C*P*P) x d} then maps each patch row to a d-dim
    token: this is the "linear patch embedding". (A Conv2d with kernel=stride=P
    computes exactly the same thing.)
    """
    C, H, W = img.shape
    assert H % patch == 0 and W % patch == 0
    nh, nw = H // patch, W // patch
    # (C, nh, P, nw, P) -> (nh, nw, C, P, P) -> (nh*nw, C*P*P)
    x = img.reshape(C, nh, patch, nw, patch)
    x = x.transpose(1, 3, 0, 2, 4)
    return x.reshape(nh * nw, C * patch * patch)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — a compact but complete ViT
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PatchEmbed(nn.Module):
    """Conv2d(kernel=stride=patch) == linear projection of each flattened patch."""

    def __init__(self, in_ch: int, img_size: int, patch: int, d_model: int):
        super().__init__()
        assert img_size % patch == 0
        self.n_patches = (img_size // patch) ** 2
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, C, H, W)
        x = self.proj(x)                                       # (B, d, H/P, W/P)
        return x.flatten(2).transpose(1, 2)                    # (B, n_patches, d)


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention (ViT encoder, no mask)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, L, d)
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        attn = self.drop(scores.softmax(-1))
        o = (attn @ v).transpose(1, 2).reshape(B, L, self.h * self.d_k)
        return self.out(o)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x): return self.net(x)


class EncoderBlock(nn.Module):
    """Pre-norm: x = x + Attn(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class VisionTransformer(nn.Module):
    """Patchify -> [CLS] + pos-emb -> Transformer encoder -> classify from [CLS]."""

    def __init__(self, in_ch: int, img_size: int, patch: int, n_classes: int,
                 d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, img_size, patch, d_model)
        n = self.patch_embed.n_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))    # learnable [CLS]
        self.pos = nn.Parameter(torch.zeros(1, n + 1, d_model))  # learned positions
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, C, H, W)
        B = x.size(0)
        tokens = self.patch_embed(x)                           # (B, n, d)
        cls = self.cls.expand(B, -1, -1)                       # (B, 1, d)
        x = torch.cat([cls, tokens], dim=1) + self.pos         # prepend [CLS] + pos
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.head(x[:, 0])                              # classify from [CLS]


# ---------------------------------------------------------------------------
# 3. Demo — tiny 8x8 "digits" split into 4x4 patches, few training steps
# ---------------------------------------------------------------------------
def make_toy_images(n: int, size: int = 8, seed: int = SEED):
    """3-class toy images on an 8x8 grid: vertical bar / horizontal bar / diagonal.
    Position varies and noise is added so the task is non-trivial but tiny."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size), dtype=np.float32)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    for i in range(n):
        if y[i] == 0:                                   # vertical bar
            c = rng.integers(1, size - 1)
            X[i, 0, :, c] = 1.0
        elif y[i] == 1:                                 # horizontal bar
            r = rng.integers(1, size - 1)
            X[i, 0, r, :] = 1.0
        else:                                           # diagonal
            for d in range(size):
                X[i, 0, d, d] = 1.0
        X[i] += 0.15 * rng.standard_normal((1, size, size)).astype(np.float32)
    return X, y


def demo():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = get_device()

    size, patch = 8, 4
    X_np, y_np = make_toy_images(900, size)
    X = torch.tensor(X_np, device=dev)
    y = torch.tensor(y_np, device=dev)

    # show the patchify math on one image (8x8 -> 4 patches of 4x4 = 16 dims)
    seq = patchify_numpy(X_np[0], patch)
    print(f"patchify: image {X_np[0].shape} -> sequence {seq.shape} "
          f"({(size//patch)**2} patches of dim {1*patch*patch})")

    model = VisionTransformer(in_ch=1, img_size=size, patch=patch, n_classes=3,
                              d_model=64, n_heads=4, d_ff=128, n_layers=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    lossfn = nn.CrossEntropyLoss()

    model.train()
    for step in range(1, 251):
        logits = model(X)
        loss = lossfn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.4f}  train acc {acc:.2f}")

    # evaluate on a fresh batch
    model.eval()
    Xt_np, yt_np = make_toy_images(300, size, seed=SEED + 1)
    with torch.no_grad():
        pred = model(torch.tensor(Xt_np, device=dev)).argmax(-1).cpu().numpy()
    acc = float((pred == yt_np).mean())
    names = ["vertical", "horizontal", "diagonal"]
    print(f"\ntest accuracy: {acc:.2f}")
    print("sample true :", [names[i] for i in yt_np[:6]])
    print("sample pred :", [names[i] for i in pred[:6]])


if __name__ == "__main__":
    demo()

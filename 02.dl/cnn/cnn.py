"""
Convolutional Neural Network (CNN)
==================================
Convolutions share weights across space and exploit locality, making them the
workhorse for images. We implement a forward/backward conv layer from scratch
with the **im2col** trick (turns convolution into one big matrix multiply), then
build a small **LeNet** in PyTorch.

Variants implemented here:
    - Conv2D + max-pool from scratch (im2col forward & backward)
    - A tiny ConvNet trained on digits in NumPy
    - LeNet-5-style network in PyTorch

Training techniques demonstrated:
    - Parameter sharing / local receptive fields (the core CNN idea)
    - im2col vectorization
    - (ResNet file covers skip connections vs vanishing gradients)

References:
    - LeCun et al. (1998), "Gradient-based learning applied to document recognition"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# im2col / col2im — the trick that makes conv a matmul
# ---------------------------------------------------------------------------
def im2col(X, kh, kw, stride=1, pad=0):
    """X:(N,C,H,W) -> columns:(N*OH*OW, C*kh*kw)."""
    N, C, H, W = X.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    cols = np.zeros((N, C, kh, kw, OH, OW))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = Xp[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1), OH, OW


def col2im(cols, X_shape, kh, kw, stride=1, pad=0, OH=None, OW=None):
    """Inverse of im2col, scattering gradients back (accumulating overlaps)."""
    N, C, H, W = X_shape
    Xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    cols = cols.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    for i in range(kh):
        for j in range(kw):
            Xp[:, :, i:i + stride * OH:stride, j:j + stride * OW:stride] += cols[:, :, i, j]
    return Xp[:, :, pad:pad + H, pad:pad + W] if pad else Xp


# ---------------------------------------------------------------------------
# 1. NumPy layers
# ---------------------------------------------------------------------------
class Conv2D:
    def __init__(self, in_c, out_c, k, stride=1, pad=0, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / (in_c * k * k)), (out_c, in_c, k, k))  # He
        self.b = np.zeros(out_c)
        self.k, self.stride, self.pad = k, stride, pad

    def forward(self, X):
        self.X_shape = X.shape
        cols, OH, OW = im2col(X, self.k, self.k, self.stride, self.pad)
        self.cols, self.OH, self.OW = cols, OH, OW
        Wc = self.W.reshape(self.W.shape[0], -1)          # (out_c, C*k*k)
        out = cols @ Wc.T + self.b                        # matmul = convolution
        N = X.shape[0]
        return out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

    def backward(self, dout, lr):
        N, OC, OH, OW = dout.shape
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, OC)
        Wc = self.W.reshape(OC, -1)
        dW = (dout_r.T @ self.cols).reshape(self.W.shape)
        db = dout_r.sum(0)
        dcols = dout_r @ Wc
        dX = col2im(dcols, self.X_shape, self.k, self.k, self.stride, self.pad, OH, OW)
        self.W -= lr * dW; self.b -= lr * db
        return dX


class MaxPool2D:
    def __init__(self, k=2, stride=2):
        self.k, self.stride = k, stride

    def forward(self, X):
        N, C, H, W = X.shape
        self.X_shape = X.shape
        OH = (H - self.k) // self.stride + 1
        OW = (W - self.k) // self.stride + 1
        Xr = X.reshape(N * C, 1, H, W)
        cols, _, _ = im2col(Xr, self.k, self.k, self.stride)
        self.argmax = cols.argmax(1)
        out = cols.max(1).reshape(N, C, OH, OW)
        self.cols_shape, self.OH, self.OW = cols.shape, OH, OW
        return out

    def backward(self, dout, lr=None):
        N, C, H, W = self.X_shape
        dcols = np.zeros(self.cols_shape)
        dcols[np.arange(dcols.shape[0]), self.argmax] = dout.transpose(0, 2, 3, 1).ravel()
        dXr = col2im(dcols, (N * C, 1, H, W), self.k, self.k, self.stride, 0, self.OH, self.OW)
        return dXr.reshape(N, C, H, W)


class ReLU:
    def forward(self, X): self.mask = X > 0; return X * self.mask
    def backward(self, d, lr=None): return d * self.mask


class Flatten:
    def forward(self, X): self.shape = X.shape; return X.reshape(X.shape[0], -1)
    def backward(self, d, lr=None): return d.reshape(self.shape)


class Dense:
    def __init__(self, nin, nout, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / nin), (nin, nout)); self.b = np.zeros(nout)

    def forward(self, X): self.X = X; return X @ self.W + self.b
    def backward(self, d, lr):
        dW = self.X.T @ d; db = d.sum(0); dX = d @ self.W.T
        self.W -= lr * dW; self.b -= lr * db
        return dX


class ConvNetNumPy:
    """Conv -> ReLU -> MaxPool -> Flatten -> Dense -> softmax."""

    def __init__(self, in_c=1, n_classes=10, img=8, lr=0.05):
        self.lr = lr
        self.conv = Conv2D(in_c, 4, 3, pad=1)
        self.relu = ReLU(); self.pool = MaxPool2D(2, 2); self.flat = Flatten()
        pooled = (img // 2)
        self.fc = Dense(4 * pooled * pooled, n_classes)
        self.layers = [self.conv, self.relu, self.pool, self.flat, self.fc]

    def forward(self, X):
        for l in self.layers: X = l.forward(X)
        return softmax(X)

    def fit(self, X, y, epochs=8, batch=32, seed=SEED):
        rng = np.random.default_rng(seed); self.history = []
        for _ in range(epochs):
            idx = rng.permutation(len(X)); loss = 0.0
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                p = self.forward(X[b])
                Y = np.eye(p.shape[1])[y[b]]
                loss += -np.sum(np.log(p[np.arange(len(b)), y[b]] + 1e-12))
                d = (p - Y) / len(b)
                for l in reversed(self.layers):
                    d = l.backward(d, self.lr)
            self.history.append(loss / len(X))
        return self

    def predict(self, X):
        return self.forward(X).argmax(1)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — LeNet-5 style
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class LeNetTorch(nn.Module):
    def __init__(self, in_c=1, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_c, 6, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

    def fit(self, X, y, epochs=8, batch=32, lr=0.01):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(y, dtype=torch.long, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=dev)
            for s in range(0, len(X), batch):
                idx = perm[s:s + batch]
                opt.zero_grad(); loss = loss_fn(self(X[idx]), y[idx])
                loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return self(X).argmax(1).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo — 8x8 digits
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.images[:, None, :, :] / 16.0           # (N,1,8,8) in [0,1]
    y = d.target
    n_tr = 1400
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    net = ConvNetNumPy(img=8, lr=0.1).fit(Xtr, ytr, epochs=8)
    print(f"NumPy ConvNet test acc = {np.mean(net.predict(Xte) == yte):.3f}")

    lenet = LeNetTorch().fit(Xtr, ytr, epochs=8, lr=0.01)
    print(f"Torch LeNet   test acc = {np.mean(lenet.predict(Xte) == yte):.3f}")


if __name__ == "__main__":
    demo()

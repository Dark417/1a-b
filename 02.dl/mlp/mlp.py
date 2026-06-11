"""
Multi-Layer Perceptron (MLP)
============================
A stack of linear layers + nonlinearities, trained by backpropagation. This is
"the" neural network — once you can derive and code backprop here, every later
architecture is a variation. We implement the full forward/backward pass in pure
NumPy, then mirror it in PyTorch.

Variants implemented here:
    - Arbitrary depth/width, choice of activation (ReLU / tanh / sigmoid)
    - Classification (softmax + cross-entropy) and the building blocks for regression
    - Weight init: He (ReLU) vs Xavier (tanh) vs naive

Training techniques demonstrated:
    - Backpropagation (the chain rule, made explicit)
    - Weight initialization (Xavier/He) — see training-techniques/README.md
    - VANISHING GRADIENTS: deep sigmoid nets vs ReLU + He init (measured!)

References:
    - Rumelhart, Hinton, Williams (1986); Glorot & Bengio (2010); He et al. (2015)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# --- activations and their derivatives --------------------------------------
def relu(z):            return np.maximum(0, z)
def drelu(z):           return (z > 0).astype(z.dtype)
def tanh(z):            return np.tanh(z)
def dtanh(z):           return 1.0 - np.tanh(z) ** 2
def sigmoid(z):         return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
def dsigmoid(z):        s = sigmoid(z); return s * (1 - s)

ACT = {"relu": (relu, drelu), "tanh": (tanh, dtanh), "sigmoid": (sigmoid, dsigmoid)}


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z); return E / E.sum(1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. NumPy implementation — forward + backprop by hand
# ---------------------------------------------------------------------------
class MLPNumPy:
    r"""
    Layer l:  z[l] = a[l-1] W[l] + b[l];   a[l] = f(z[l]);   a[0] = x.
    Output: softmax; loss: cross-entropy.

    Backprop (the chain rule):
        δ[L] = a[L] - y_onehot                       (softmax+CE gradient)
        δ[l] = (δ[l+1] W[l+1]^T) ⊙ f'(z[l])          (propagate backward)
        dW[l] = a[l-1]^T δ[l] / n ;  db[l] = mean(δ[l])
    """

    def __init__(self, sizes, activation="relu", init="he", lr=0.1, seed=SEED):
        self.sizes = sizes                       # e.g. [in, h1, h2, out]
        self.act_name = activation
        self.f, self.df = ACT[activation]
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for nin, nout in zip(sizes[:-1], sizes[1:]):
            if init == "he":         scale = np.sqrt(2.0 / nin)      # ReLU
            elif init == "xavier":   scale = np.sqrt(1.0 / nin)      # tanh/sigmoid
            else:                    scale = 1.0                     # naive (bad)
            self.W.append(rng.normal(0, scale, size=(nin, nout)))
            self.b.append(np.zeros(nout))

    def forward(self, X):
        self.z, self.a = [], [X]                 # cache for backprop
        h = X
        for l in range(len(self.W)):
            z = h @ self.W[l] + self.b[l]
            self.z.append(z)
            h = softmax(z) if l == len(self.W) - 1 else self.f(z)
            self.a.append(h)
        return h

    def backward(self, y):
        n = len(y)
        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        Y = np.eye(self.sizes[-1])[y]
        delta = (self.a[-1] - Y) / n             # softmax + CE
        self.grad_norms = []                     # track per-layer for vanishing demo
        for l in reversed(range(len(self.W))):
            grads_W[l] = self.a[l].T @ delta
            grads_b[l] = delta.sum(0)
            self.grad_norms.append(np.linalg.norm(grads_W[l]))
            if l > 0:                            # propagate to previous layer
                delta = (delta @ self.W[l].T) * self.df(self.z[l - 1])
        self.grad_norms.reverse()
        return grads_W, grads_b

    def step(self, gW, gb):
        for l in range(len(self.W)):
            self.W[l] -= self.lr * gW[l]
            self.b[l] -= self.lr * gb[l]

    def fit(self, X, y, epochs=200, batch=64, seed=SEED):
        rng = np.random.default_rng(seed)
        self.history = []
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                self.forward(X[b])
                self.step(*self.backward(y[b]))
            p = self.forward(X)
            self.history.append(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
        return self

    def predict(self, X):
        return self.forward(X).argmax(1)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPTorch(nn.Module):
    def __init__(self, sizes, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation]
        layers = []
        for i, (nin, nout) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(nin, nout))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y, epochs=200, batch=64, lr=0.1):
        dev = get_device(); self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(y, dtype=torch.long, device=dev)
        opt = torch.optim.SGD(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                loss = loss_fn(self(X[idx]), y[idx])
                loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return self(X).argmax(1).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo  (incl. the vanishing-gradient measurement)
# ---------------------------------------------------------------------------
def vanishing_gradient_demo():
    """Compare per-layer gradient magnitude in a DEEP net: sigmoid vs ReLU+He."""
    np.random.seed(SEED)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=20, n_informative=10,
                               n_classes=3, random_state=SEED)
    X = (X - X.mean(0)) / X.std(0)
    deep = [20, 64, 64, 64, 64, 64, 3]           # 6 layers deep

    print("Per-layer ||dW|| right after init (input layer ... output layer):")
    for act, init in [("sigmoid", "xavier"), ("relu", "he")]:
        net = MLPNumPy(deep, activation=act, init=init)
        net.forward(X); net.backward(y)
        norms = np.array(net.grad_norms)
        ratio = norms[-1] / (norms[0] + 1e-12)
        print(f"  {act:7s}: " + " ".join(f"{g:7.1e}" for g in norms) +
              f"   (output/input = {ratio:6.1f}x)")
    print("  -> sigmoid gradients shrink toward the input layer (vanishing);")
    print("     ReLU+He keep them comparable, so deep nets actually train.")


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=600, noise=0.2, random_state=SEED)
    X = (X - X.mean(0)) / X.std(0)
    Xtr, ytr, Xte, yte = X[:480], y[:480], X[480:], y[480:]

    net = MLPNumPy([2, 32, 32, 2], activation="relu", init="he", lr=0.2)
    net.fit(Xtr, ytr, epochs=300)
    print(f"NumPy MLP  test acc = {np.mean(net.predict(Xte) == yte):.3f}")

    tnet = MLPTorch([2, 32, 32, 2], activation="relu").fit(Xtr, ytr, epochs=300, lr=0.2)
    print(f"Torch MLP  test acc = {np.mean(tnet.predict(Xte) == yte):.3f}")

    print()
    vanishing_gradient_demo()


if __name__ == "__main__":
    demo()

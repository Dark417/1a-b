"""
Logistic Regression
====================
Linear classification by squashing a linear score through a sigmoid and
training with cross-entropy (= maximum likelihood for a Bernoulli model). The
bridge from regression to classification, and a single-neuron neural net.

Variants implemented here:
    - Binary logistic regression (sigmoid + BCE)
    - Multinomial / softmax regression (K classes + cross-entropy)
    - L2 regularization

Training techniques demonstrated:
    - Cross-entropy loss and its remarkably clean gradient
    - Feature standardization

References:
    - Bishop, "Pattern Recognition and Machine Learning", ch. 4.3
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _sigmoid(z):
    # numerically stable sigmoid
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)        # stability
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. NumPy implementation
# ---------------------------------------------------------------------------
class LogisticRegressionNumPy:
    r"""
    Binary:   p = σ(Xw + b),  loss = -Σ[y log p + (1-y) log(1-p)] / n
              gradient:  dL/dw = (1/n) X^T (p - y)   <- same shape as linear reg!
    Softmax:  P = softmax(XW + b),  loss = -Σ log P[i, y_i] / n
              gradient:  dL/dW = (1/n) X^T (P - Y_onehot)
    """

    def __init__(self, multi_class=False, n_classes=None,
                 lam=0.0, lr=0.1, n_iters=2000):
        self.multi_class = multi_class
        self.n_classes = n_classes
        self.lam, self.lr, self.n_iters = lam, lr, n_iters
        self.W = None; self.b = None
        self.history = []

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y)
        n, d = X.shape
        if self.multi_class:
            K = self.n_classes or int(y.max() + 1)
            self.W = np.zeros((d, K)); self.b = np.zeros(K)
            Y = np.eye(K)[y]                                  # one-hot
            for _ in range(self.n_iters):
                P = _softmax(X @ self.W + self.b)
                gW = X.T @ (P - Y) / n + self.lam * self.W
                gb = (P - Y).mean(0)
                self.W -= self.lr * gW; self.b -= self.lr * gb
                self.history.append(-np.mean(np.log(P[np.arange(n), y] + 1e-12)))
        else:
            self.W = np.zeros(d); self.b = 0.0
            y = y.astype(float)
            for _ in range(self.n_iters):
                p = _sigmoid(X @ self.W + self.b)
                gW = X.T @ (p - y) / n + self.lam * self.W
                gb = (p - y).mean()
                self.W -= self.lr * gW; self.b -= self.lr * gb
                eps = 1e-12
                self.history.append(-np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if self.multi_class:
            return _softmax(X @ self.W + self.b)
        return _sigmoid(X @ self.W + self.b)

    def predict(self, X):
        P = self.predict_proba(X)
        return P.argmax(1) if self.multi_class else (P >= 0.5).astype(int)


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


class LogisticRegressionTorch(nn.Module):
    def __init__(self, in_features, n_classes=2):
        super().__init__()
        self.multi = n_classes > 2
        out = n_classes if self.multi else 1
        self.linear = nn.Linear(in_features, out)

    def forward(self, x):
        z = self.linear(x)
        return z if self.multi else z.squeeze(-1)

    def fit(self, X, y, lr=0.1, n_iters=2000):
        dev = get_device(); self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        if self.multi:
            y = torch.as_tensor(y, dtype=torch.long, device=dev)
            loss_fn = nn.CrossEntropyLoss()
        else:
            y = torch.as_tensor(y, dtype=torch.float32, device=dev)
            loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.SGD(self.parameters(), lr=lr)
        for _ in range(n_iters):
            opt.zero_grad()
            loss = loss_fn(self(X), y)
            loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        z = self(X)
        if self.multi:
            return z.argmax(1).cpu().numpy()
        return (torch.sigmoid(z) >= 0.5).long().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_classification, make_blobs

    # --- binary ---
    Xb, yb = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                 n_clusters_per_class=1, random_state=SEED)
    mu, sd = Xb.mean(0), Xb.std(0); Xb = (Xb - mu) / sd
    m = LogisticRegressionNumPy(lr=0.5, n_iters=2000).fit(Xb, yb)
    acc = np.mean(m.predict(Xb) == yb)
    t = LogisticRegressionTorch(2, n_classes=2).fit(Xb, yb, lr=0.5, n_iters=2000)
    acc_t = np.mean(t.predict(Xb) == yb)
    print(f"[binary]  NumPy acc={acc:.3f}   Torch acc={acc_t:.3f}")

    # --- multiclass (softmax) ---
    Xm, ym = make_blobs(n_samples=450, centers=3, n_features=2, random_state=SEED)
    mu, sd = Xm.mean(0), Xm.std(0); Xm = (Xm - mu) / sd
    ms = LogisticRegressionNumPy(multi_class=True, n_classes=3, lr=0.5, n_iters=2000).fit(Xm, ym)
    acc_s = np.mean(ms.predict(Xm) == ym)
    ts = LogisticRegressionTorch(2, n_classes=3).fit(Xm, ym, lr=0.5, n_iters=2000)
    acc_ts = np.mean(ts.predict(Xm) == ym)
    print(f"[softmax] NumPy acc={acc_s:.3f}   Torch acc={acc_ts:.3f}")


if __name__ == "__main__":
    demo()

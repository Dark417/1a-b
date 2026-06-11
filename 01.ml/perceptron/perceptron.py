"""
Perceptron
==========
The original (1958) trainable neuron: a linear threshold unit updated by a
mistake-driven rule. Historically the seed of neural networks; conceptually the
limit case that motivates the SVM (max margin) and logistic regression (soft
probabilities).

Variants implemented here:
    - Classic Rosenblatt perceptron
    - Pocket algorithm (keeps best weights — for non-separable data)
    - Averaged perceptron (better generalization)
    - Multiclass via one-vs-rest

Training techniques demonstrated:
    - The perceptron convergence behaviour (separable vs not)
    - Why a single linear unit cannot solve XOR (needs a hidden layer → MLP)

References:
    - Rosenblatt (1958); Novikoff (1962) convergence theorem
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation
# ---------------------------------------------------------------------------
class PerceptronNumPy:
    r"""
    Prediction:  ŷ = sign(w·x + b),  labels in {-1, +1}.
    Update on a mistake (y·(w·x+b) ≤ 0):
        w ← w + η y x ,   b ← b + η y
    This is stochastic (sub)gradient descent on the hinge-at-0 loss
        max(0, -y (w·x + b)).
    """

    def __init__(self, lr=1.0, n_epochs=20, mode="vanilla", seed=SEED):
        self.lr, self.n_epochs, self.mode, self.seed = lr, n_epochs, mode, seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.where(np.asarray(y) <= 0, -1, 1)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        w = np.zeros(d); b = 0.0
        w_sum = np.zeros(d); b_sum = 0.0; count = 0
        best_w, best_b, best_err = w.copy(), b, n + 1
        self.errors_ = []
        for _ in range(self.n_epochs):
            errs = 0
            for i in rng.permutation(n):
                if y[i] * (w @ X[i] + b) <= 0:        # misclassified
                    w += self.lr * y[i] * X[i]
                    b += self.lr * y[i]
                    errs += 1
                w_sum += w; b_sum += b; count += 1     # for averaged
            self.errors_.append(errs)
            if errs < best_err:                        # pocket: remember best
                best_err, best_w, best_b = errs, w.copy(), b
            if errs == 0:
                break
        if self.mode == "pocket":
            self.w, self.b = best_w, best_b
        elif self.mode == "averaged":
            self.w, self.b = w_sum / count, b_sum / count
        else:
            self.w, self.b = w, b
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class MulticlassPerceptron:
    """One-vs-rest wrapper around the binary perceptron."""

    def __init__(self, n_classes, **kw):
        self.n_classes, self.kw = n_classes, kw

    def fit(self, X, y):
        self.models = []
        for c in range(self.n_classes):
            yc = (np.asarray(y) == c).astype(int)
            self.models.append(PerceptronNumPy(**self.kw).fit(X, yc))
        return self

    def predict(self, X):
        scores = np.stack([m.decision_function(X) for m in self.models], 1)
        return scores.argmax(1)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (single linear unit, perceptron loss)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


class PerceptronTorch(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.lin = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)

    def fit(self, X, y, lr=0.1, n_epochs=50):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(np.where(np.asarray(y) <= 0, -1, 1),
                            dtype=torch.float32, device=dev)
        opt = torch.optim.SGD(self.parameters(), lr=lr)
        for _ in range(n_epochs):
            opt.zero_grad()
            margin = y * self(X)
            loss = torch.clamp(-margin, min=0).mean()   # perceptron loss
            loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return torch.where(self(X) >= 0, 1, -1).cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=SEED)
    p = PerceptronNumPy(n_epochs=20).fit(X, y)
    converged = p.errors_[-1] == 0
    print(f"ran {len(p.errors_)} epochs (zero-error reached: {converged}); "
          f"final mistakes/epoch={p.errors_[-1]}, "
          f"acc={np.mean((p.predict(X) > 0).astype(int) == y):.3f}")

    pt = PerceptronTorch(2).fit(X, y)
    print(f"torch acc={np.mean((pt.predict(X) > 0).astype(int) == y):.3f}")

    # XOR: not linearly separable — the perceptron cannot fit it (motivates MLP)
    Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    yx = np.array([0, 1, 1, 0])
    px = PerceptronNumPy(n_epochs=100, mode="pocket").fit(Xx, yx)
    print(f"XOR best accuracy a single perceptron reaches: "
          f"{np.mean((px.predict(Xx) > 0).astype(int) == yx):.2f} (max 0.75 — needs a hidden layer)")


if __name__ == "__main__":
    demo()

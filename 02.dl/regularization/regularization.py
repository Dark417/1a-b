"""
Regularization (the techniques that fight overfitting)
======================================================
A model with enough capacity will happily memorize the training set. Regularization
trades a little training fit for a lot of generalization. This is the **canonical
home** in the repo for Dropout, BatchNorm and LayerNorm (see
training-techniques/README.md); we implement each from scratch with an explicit
forward AND backward pass, plus weight decay (L1/L2), early stopping and label
smoothing, then show on a small overfitting-prone MLP that dropout + weight decay
shrink the train/test gap.

Variants implemented here:
    - L2 weight decay and L1 (their gradients)
    - Dropout (inverted) — forward + backward
    - BatchNorm (1D) — forward + backward (the full chain rule)
    - LayerNorm — forward + backward
    - Early stopping (patience on a validation metric)
    - Label smoothing (soft cross-entropy targets)

Training techniques demonstrated:
    - Dropout                       (CANONICAL DEMO)
    - Batch / Layer normalization   (CANONICAL DEMO)
    - L1/L2 weight decay, early stopping, label smoothing
      (see training-techniques/README.md)

References:
    - Srivastava et al. (2014, Dropout); Ioffe & Szegedy (2015, BatchNorm)
    - Ba et al. (2016, LayerNorm); Szegedy et al. (2016, label smoothing)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementations (from scratch) — forward + backward where it matters
# ---------------------------------------------------------------------------
def l2_penalty(W: np.ndarray, lam: float):
    r"""L2 (ridge) weight decay. Loss term lam/2 * ||W||^2; gradient lam*W.
    Returns (penalty_value, grad_wrt_W)."""
    return 0.5 * lam * np.sum(W ** 2), lam * W


def l1_penalty(W: np.ndarray, lam: float):
    r"""L1 (lasso) weight decay. Loss term lam*||W||_1; subgradient lam*sign(W).
    L1 drives weights exactly to 0 (sparsity); L2 only shrinks them."""
    return lam * np.sum(np.abs(W)), lam * np.sign(W)


class Dropout:
    r"""Inverted dropout. TRAIN: keep each unit w.p. (1-p), and SCALE survivors by
    1/(1-p) so the expected activation is unchanged; TEST: identity (no scaling
    needed thanks to the train-time scaling).

        mask ~ Bernoulli(1-p)/(1-p)
        forward:  y = x * mask
        backward: dx = dy * mask        (the mask is a constant w.r.t. x)
    """

    def __init__(self, p: float = 0.5, seed: int = SEED):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward(self, x, train: bool = True):
        if not train or self.p == 0.0:
            self.mask = None                                   # identity: no gating
            return x
        keep = 1.0 - self.p
        self.mask = (self.rng.random(x.shape) < keep) / keep   # inverted scaling
        return x * self.mask

    def backward(self, dy):
        if self.mask is None:
            return dy                                          # identity case
        return dy * self.mask                                  # gradient gated by mask


class BatchNorm1d:
    r"""Batch normalization over a (N, D) batch, normalizing each feature across
    the batch dimension.
        mu = mean_n x ;  var = var_n x
        x_hat = (x - mu)/sqrt(var + eps)
        y = gamma * x_hat + beta
    At test time we use running estimates of mu/var (momentum EMA).

    Backward (the famous chain rule). With N samples, let dy be upstream:
        dgamma = sum_n dy * x_hat ;  dbeta = sum_n dy
        dx_hat = dy * gamma
        dvar   = sum_n dx_hat*(x-mu) * -0.5*(var+eps)^{-3/2}
        dmu    = sum_n dx_hat * -1/sqrt(var+eps) + dvar * mean(-2(x-mu))
        dx     = dx_hat/sqrt(var+eps) + dvar*2(x-mu)/N + dmu/N
    """

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps, self.momentum = eps, momentum
        self.run_mu = np.zeros(dim)
        self.run_var = np.ones(dim)

    def forward(self, x, train: bool = True):
        if train:
            mu = x.mean(0)
            var = x.var(0)
            self.run_mu = (1 - self.momentum) * self.run_mu + self.momentum * mu
            self.run_var = (1 - self.momentum) * self.run_var + self.momentum * var
        else:
            mu, var = self.run_mu, self.run_var
        self.std = np.sqrt(var + self.eps)
        self.x_mu = x - mu
        self.x_hat = self.x_mu / self.std
        return self.gamma * self.x_hat + self.beta

    def backward(self, dy):
        N = dy.shape[0]
        self.dgamma = np.sum(dy * self.x_hat, axis=0)
        self.dbeta = np.sum(dy, axis=0)
        dx_hat = dy * self.gamma
        dvar = np.sum(dx_hat * self.x_mu, axis=0) * -0.5 * self.std ** (-3)
        dmu = np.sum(-dx_hat / self.std, axis=0) + dvar * np.mean(-2.0 * self.x_mu, axis=0)
        dx = dx_hat / self.std + dvar * 2.0 * self.x_mu / N + dmu / N
        return dx


class LayerNorm:
    r"""Layer normalization: normalize over the FEATURE axis, per sample (no batch
    statistics, no train/test difference). This is what Transformers use.
        mu = mean over features ;  var = var over features (per row)
        x_hat = (x - mu)/sqrt(var + eps) ;  y = gamma*x_hat + beta
    Backward has the same structure as BatchNorm but reduces over the feature axis:
        dx = (1/std) * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))
    where dx_hat = dy * gamma and the means are over the feature axis.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def forward(self, x, train: bool = True):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        self.std = np.sqrt(var + self.eps)
        self.x_hat = (x - mu) / self.std
        return self.gamma * self.x_hat + self.beta

    def backward(self, dy):
        D = dy.shape[-1]
        self.dgamma = np.sum(dy * self.x_hat, axis=0)
        self.dbeta = np.sum(dy, axis=0)
        dx_hat = dy * self.gamma
        # compact LayerNorm backward (means over the feature axis)
        mean_dxhat = dx_hat.mean(-1, keepdims=True)
        mean_dxhat_xhat = (dx_hat * self.x_hat).mean(-1, keepdims=True)
        dx = (dx_hat - mean_dxhat - self.x_hat * mean_dxhat_xhat) / self.std
        return dx


def label_smoothing_targets(y: np.ndarray, n_classes: int, eps: float = 0.1):
    r"""Soft targets: put 1-eps on the true class, eps/(K-1) spread on the rest.
        q_k = (1-eps) [k=y] + eps/(K-1) [k!=y].
    Cross-entropy against these soft targets discourages over-confident logits
    (a calibration + regularization effect)."""
    q = np.full((len(y), n_classes), eps / (n_classes - 1))
    q[np.arange(len(y)), y] = 1.0 - eps
    return q


class EarlyStopping:
    r"""Stop when a monitored validation metric stops improving for `patience`
    epochs. Returns True from `step()` when training should halt. Keeps the best
    weights so you can restore them."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = np.inf
        self.bad = 0
        self.best_state = None

    def step(self, val_loss: float, state=None) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad = 0
            self.best_state = state
            return False
        self.bad += 1
        return self.bad >= self.patience


# ---------------------------------------------------------------------------
# A small MLP (NumPy) used to DEMONSTRATE that dropout + weight decay reduce
# overfitting. Forward/backward for a 1-hidden-layer net with optional dropout
# and L2 decay; this is intentionally minimal (the MLP file is the full version).
# ---------------------------------------------------------------------------
def relu(z): return np.maximum(0.0, z)
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


class RegMLP:
    """1 hidden layer: x -> ReLU -> (optional Dropout) -> softmax, SGD trained."""

    def __init__(self, d_in, d_hid, d_out, lr=0.1, l2=0.0, dropout=0.0, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hid))
        self.b1 = np.zeros(d_hid)
        self.W2 = rng.normal(0, np.sqrt(2.0 / d_hid), (d_hid, d_out))
        self.b2 = np.zeros(d_out)
        self.lr, self.l2, self.d_out = lr, l2, d_out
        self.drop = Dropout(dropout, seed=seed)

    def forward(self, X, train=True):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.a1d = self.drop.forward(self.a1, train=train)
        self.p = softmax(self.a1d @ self.W2 + self.b2)
        return self.p

    def step(self, y):
        n = len(y)
        Y = np.eye(self.d_out)[y]
        d2 = (self.p - Y) / n                         # softmax+CE
        dW2 = self.a1d.T @ d2 + self.l2 * self.W2     # + L2 gradient
        db2 = d2.sum(0)
        da1 = self.drop.backward(d2 @ self.W2.T)      # through dropout mask
        d1 = da1 * (self.z1 > 0)                       # ReLU'
        dW1 = self.X.T @ d1 + self.l2 * self.W1
        db1 = d1.sum(0)
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def loss(self, X, y):
        p = self.forward(X, train=False)
        return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

    def fit(self, X, y, epochs=300, batch=32, seed=SEED):
        # Mini-batch SGD: a FRESH dropout mask per batch is what makes dropout an
        # implicit ensemble over sub-networks (a single full-batch mask would not).
        rng = np.random.default_rng(seed)
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                self.forward(X[b], train=True)
                self.step(y[b])
        return self

    def acc(self, X, y):
        return float(np.mean(self.forward(X, train=False).argmax(1) == y))


def numeric_grad_check(layer, x, eps=1e-6):
    """Verify a norm layer's backward against finite differences on a scalar loss
    L = sum(forward(x)). Returns the max abs error between analytic and numeric dx."""
    y = layer.forward(x, train=True)
    dy = np.ones_like(y)
    dx = layer.backward(dy)
    num = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; lp = layer.forward(x, train=True).sum()
        x[i] = old - eps; lm = layer.forward(x, train=True).sum()
        x[i] = old
        num[i] = (lp - lm) / (2 * eps)
        it.iternext()
    return float(np.max(np.abs(dx - num)))


# ---------------------------------------------------------------------------
# 2. PyTorch usage — the idiomatic versions of the same techniques
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

# Keep the CPU demo fast & deterministic: avoid thread oversubscription, which can
# make these tiny full-batch problems paradoxically slow.
torch.set_num_threads(1)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RegMLPTorch(nn.Module):
    """nn.Dropout + (optional) nn.BatchNorm1d; weight decay via the optimizer."""

    def __init__(self, d_in, d_hid, d_out, dropout=0.0, batchnorm=False):
        super().__init__()
        layers = [nn.Linear(d_in, d_hid)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(d_hid))
        layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hid, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def fit(self, X, y, epochs=300, lr=0.1, weight_decay=0.0, label_smoothing=0.0):
        dev = get_device(); self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(y, dtype=torch.long, device=dev)
        # weight_decay in the optimizer == DECOUPLED L2 (AdamW-style); SGD couples it
        opt = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        for _ in range(epochs):
            self.train()
            opt.zero_grad()
            loss = loss_fn(self(X), y)
            loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def acc(self, X, y):
        self.eval()
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return float((self(X).argmax(1).cpu().numpy() == y).mean())


# ---------------------------------------------------------------------------
# 3. Demo — verify norm backprop, then SHOW overfitting reduced by reg
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)

    # (a) numerically verify BatchNorm & LayerNorm backward passes
    print("Norm-layer backward vs finite differences (max abs error in dx):")
    bn = BatchNorm1d(5); x = np.random.randn(8, 5)
    print(f"  BatchNorm1d: {numeric_grad_check(bn, x.copy()):.2e}")
    ln = LayerNorm(5); x = np.random.randn(8, 5)
    print(f"  LayerNorm  : {numeric_grad_check(ln, x.copy()):.2e}")

    # (b) dropout keeps the expected activation unchanged (inverted scaling)
    drop = Dropout(0.5, seed=SEED); a = np.ones((2000, 10))
    out = drop.forward(a, train=True)
    print(f"\nInverted dropout: E[output] = {out.mean():.3f} (target 1.0; scaling preserves mean)")

    # (c) THE overfitting demo: a big net on FEW, label-noisy digits. With only 80
    # training images (20% of labels corrupted) a 256-unit net memorizes the train
    # set perfectly; regularization trades that away for better generalization.
    from sklearn.datasets import load_digits
    d = load_digits()
    Xall = d.data / 16.0
    yall = d.target.copy()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xall))
    Xall, yall = Xall[perm], yall[perm]
    Xtr, ytr = Xall[:80].copy(), yall[:80].copy()
    Xte, yte = Xall[200:700], yall[200:700]
    noisy = rng.random(len(ytr)) < 0.20                 # corrupt 20% of train labels
    ytr[noisy] = rng.integers(0, 10, noisy.sum())

    print("\nOverfitting demo (8x8 digits, 80 noisy train imgs, 256-unit hidden):")
    print(f"  {'config':22s} {'train acc':>10s} {'test acc':>10s} {'gap':>7s}")
    configs = [
        ("no regularization",  dict(l2=0.0,    dropout=0.0)),
        ("L2 weight decay",    dict(l2=5e-3,   dropout=0.0)),
        ("dropout p=0.6",      dict(l2=0.0,    dropout=0.6)),
        ("dropout + L2",       dict(l2=5e-3,   dropout=0.6)),
    ]
    for name, kw in configs:
        net = RegMLP(64, 256, 10, lr=0.2, seed=SEED, **kw).fit(
            Xtr, ytr, epochs=200, batch=16)
        tr, te = net.acc(Xtr, ytr), net.acc(Xte, yte)
        print(f"  {name:22s} {tr:10.3f} {te:10.3f} {tr-te:7.3f}")
    print("  -> regularization lowers TRAIN accuracy but RAISES test accuracy:")
    print("     the train/test gap (overfitting) shrinks.")

    # (d) early stopping on a held-out split (use a val split distinct from test)
    Xval, yval = Xall[700:900], yall[700:900]
    es = EarlyStopping(patience=20)
    net = RegMLP(64, 256, 10, lr=0.2, l2=0.0, dropout=0.0, seed=SEED)
    stop_epoch = -1
    brng = np.random.default_rng(SEED)
    for ep in range(300):
        idx = brng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 16):
            b = idx[s:s + 16]
            net.forward(Xtr[b], train=True); net.step(ytr[b])
        if es.step(net.loss(Xval, yval)):
            stop_epoch = ep; break
    print(f"\nEarly stopping fired at epoch {stop_epoch} (best val loss {es.best:.3f}),")
    print("  halting before the net memorizes the noisy labels.")

    # (e) label smoothing: softens targets to curb over-confidence. We report the
    # mean confidence (max softmax prob) on the test set: smoothing should LOWER it
    # (better-calibrated, less over-confident) while keeping accuracy comparable.
    print("\nLabel smoothing (torch): mean test confidence (max prob) & accuracy.")
    base = RegMLPTorch(64, 128, 10, dropout=0.0).fit(Xtr, ytr, epochs=200, lr=0.2)
    smot = RegMLPTorch(64, 128, 10, dropout=0.0).fit(Xtr, ytr, epochs=200, lr=0.2,
                                                     label_smoothing=0.1)

    @torch.no_grad()
    def mean_conf(model, X):
        model.eval()
        t = torch.as_tensor(X, dtype=torch.float32)
        return float(torch.softmax(model(t), 1).max(1).values.mean())

    print(f"  plain CE     : conf={mean_conf(base, Xte):.3f}  test acc={base.acc(Xte, yte):.3f}")
    print(f"  smoothed CE  : conf={mean_conf(smot, Xte):.3f}  test acc={smot.acc(Xte, yte):.3f}")
    print("  -> smoothing reduces over-confidence (lower mean prob) -> better calibration.")


if __name__ == "__main__":
    demo()

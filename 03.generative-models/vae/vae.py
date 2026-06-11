"""
Variational Autoencoder (VAE)
=============================
A probabilistic autoencoder: an encoder maps x to a *distribution* over a latent
z, a decoder reconstructs x from z, and we train by maximizing a tractable lower
bound on the data likelihood (the ELBO). The headline trick is the
**reparameterization** that lets gradients flow through a sampling step.

Variants implemented here:
    - Vanilla VAE (Gaussian latent, Bernoulli/Gaussian decoder)
    - beta-VAE (weight the KL term to encourage disentanglement)
    - Conditional VAE (condition encoder & decoder on a label)

Training techniques demonstrated:
    - The REPARAMETERIZATION TRICK (see training-techniques/README.md)
    - The ELBO = reconstruction - KL trade-off

References:
    - Kingma & Welling (2013), "Auto-Encoding Variational Bayes"
    - Higgins et al. (2017), beta-VAE
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation — tiny VAE, full forward + backward by hand
# ---------------------------------------------------------------------------
def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
def _relu(z):    return np.maximum(0, z)


class VAENumPy:
    r"""
    Encoder q(z|x)=N(mu, diag(sigma^2)); decoder p(x|z) Bernoulli (sigmoid out).

    ELBO (maximize):  E_q[log p(x|z)] - KL(q(z|x) || p(z)),   p(z)=N(0, I)
    We MINIMIZE the negative ELBO:
        L = BCE(x, x_hat)  +  beta * KL
        KL = -1/2 * sum(1 + log sigma^2 - mu^2 - sigma^2)
    Reparameterization:  z = mu + sigma * eps,  eps ~ N(0, I)  (eps is an INPUT,
    so d z/d mu = 1 and d z/d sigma = eps — gradients flow).
    """

    def __init__(self, in_dim, hidden=64, latent=2, beta=1.0, lr=1e-3, seed=SEED):
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.normal(0, np.sqrt(2.0 / a), (a, b))
        # encoder
        self.W1 = s(in_dim, hidden); self.b1 = np.zeros(hidden)
        self.Wmu = s(hidden, latent); self.bmu = np.zeros(latent)
        self.Wlv = s(hidden, latent); self.blv = np.zeros(latent)   # log-variance
        # decoder
        self.W2 = s(latent, hidden); self.b2 = np.zeros(hidden)
        self.W3 = s(hidden, in_dim); self.b3 = np.zeros(in_dim)
        self.latent, self.beta, self.lr = latent, beta, lr

    def encode(self, X):
        self.h1 = _relu(X @ self.W1 + self.b1)
        mu = self.h1 @ self.Wmu + self.bmu
        logvar = self.h1 @ self.Wlv + self.blv
        return mu, logvar

    def decode(self, Z):
        self.h2 = _relu(Z @ self.W2 + self.b2)
        return _sigmoid(self.h2 @ self.W3 + self.b3)

    def forward(self, X, rng):
        self.X = X
        self.mu, self.logvar = self.encode(X)
        self.eps = rng.normal(size=self.mu.shape)
        self.std = np.exp(0.5 * self.logvar)
        self.z = self.mu + self.std * self.eps          # reparameterization
        self.xhat = self.decode(self.z)
        return self.xhat

    def loss(self):
        n = len(self.X); eps = 1e-8
        bce = -np.sum(self.X * np.log(self.xhat + eps) +
                      (1 - self.X) * np.log(1 - self.xhat + eps)) / n
        kl = -0.5 * np.sum(1 + self.logvar - self.mu ** 2 - np.exp(self.logvar)) / n
        return bce + self.beta * kl, bce, kl

    def backward(self):
        n = len(self.X)
        # decoder grads (Bernoulli + sigmoid -> dL/dlogits = xhat - x)
        dlog = (self.xhat - self.X) / n
        dW3 = self.h2.T @ dlog; db3 = dlog.sum(0)
        dh2 = dlog @ self.W3.T; dh2[self.h2 <= 0] = 0
        dW2 = self.z.T @ dh2; db2 = dh2.sum(0)
        dz = dh2 @ self.W2.T
        # reparam: z = mu + std*eps
        dmu = dz + self.beta * self.mu / n              # +KL term d/dmu (mu^2/2)
        dstd = dz * self.eps
        # std = exp(0.5 logvar) -> dlogvar = dstd*0.5*std + KL term
        dlogvar = dstd * 0.5 * self.std \
            + self.beta * 0.5 * (np.exp(self.logvar) - 1) / n
        dWmu = self.h1.T @ dmu; dbmu = dmu.sum(0)
        dWlv = self.h1.T @ dlogvar; dblv = dlogvar.sum(0)
        dh1 = dmu @ self.Wmu.T + dlogvar @ self.Wlv.T; dh1[self.h1 <= 0] = 0
        dW1 = self.X.T @ dh1; db1 = dh1.sum(0)
        # SGD step
        for p, g in [(self.W1, dW1), (self.b1, db1), (self.Wmu, dWmu), (self.bmu, dbmu),
                     (self.Wlv, dWlv), (self.blv, dblv), (self.W2, dW2), (self.b2, db2),
                     (self.W3, dW3), (self.b3, db3)]:
            p -= self.lr * g

    def fit(self, X, epochs=60, batch=64, seed=SEED):
        rng = np.random.default_rng(seed); self.history = []
        for _ in range(epochs):
            idx = rng.permutation(len(X)); tot = 0.0
            for s in range(0, len(X), batch):
                self.forward(X[idx[s:s + batch]], rng)
                l, _, _ = self.loss(); tot += l
                self.backward()
            self.history.append(tot / (len(X) / batch))
        return self

    def generate(self, n, seed=SEED):
        rng = np.random.default_rng(seed)
        return self.decode(rng.normal(size=(n, self.latent)))


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (+ conditional option)
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAETorch(nn.Module):
    def __init__(self, in_dim, hidden=128, latent=2, beta=1.0, n_classes=0):
        super().__init__()
        self.latent, self.beta, self.n_classes = latent, beta, n_classes
        c = n_classes
        self.enc = nn.Sequential(nn.Linear(in_dim + c, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_lv = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent + c, hidden), nn.ReLU(),
                                 nn.Linear(hidden, in_dim))

    def _cat(self, x, y):
        if self.n_classes and y is not None:
            return torch.cat([x, F.one_hot(y, self.n_classes).float()], 1)
        return x

    def encode(self, x, y=None):
        h = self.enc(self._cat(x, y)); return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)   # the trick

    def decode(self, z, y=None):
        return torch.sigmoid(self.dec(self._cat(z, y)))

    def forward(self, x, y=None):
        mu, lv = self.encode(x, y); z = self.reparam(mu, lv)
        return self.decode(z, y), mu, lv

    def elbo_loss(self, x, xhat, mu, lv):
        bce = F.binary_cross_entropy(xhat, x, reduction="sum") / len(x)
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / len(x)
        return bce + self.beta * kl, bce, kl

    def fit(self, X, y=None, epochs=60, batch=128, lr=1e-3):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = None if y is None else torch.as_tensor(y, dtype=torch.long, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.history = []
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=dev); tot = 0.0
            for s in range(0, len(X), batch):
                idx = perm[s:s + batch]
                yb = None if y is None else y[idx]
                xhat, mu, lv = self(X[idx], yb)
                loss, _, _ = self.elbo_loss(X[idx], xhat, mu, lv)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            self.history.append(tot / (len(X) / batch))
        return self


# ---------------------------------------------------------------------------
# 3. Demo — 8x8 digits flattened to 64-d
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import load_digits
    X = (load_digits().data / 16.0).astype(np.float32)   # (1797, 64) in [0,1]

    v = VAENumPy(64, hidden=64, latent=2, lr=2e-3).fit(X, epochs=40)
    print(f"NumPy VAE  final -ELBO = {v.history[-1]:.3f}")
    print(f"           generated {v.generate(8).shape[0]} samples from N(0,I)")

    t = VAETorch(64, latent=2).fit(X, epochs=40)
    print(f"Torch VAE  final -ELBO = {t.history[-1]:.3f}")

    tc = VAETorch(64, latent=2, n_classes=10).fit(X, load_digits().target, epochs=40)
    print(f"Cond VAE   final -ELBO = {tc.history[-1]:.3f} (can generate a chosen digit)")


if __name__ == "__main__":
    demo()

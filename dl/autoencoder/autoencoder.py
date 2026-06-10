"""
Autoencoders
============
An autoencoder learns to copy its input through a narrow bottleneck: an encoder
$z=f(x)$ compresses, a decoder $\\hat x=g(z)$ reconstructs, and the loss is the
reconstruction error. The bottleneck forces it to discover the structure of the
data (a nonlinear cousin of PCA). The interesting part is the **regularizers** that
shape what the code learns: denoising, sparsity, and contraction. We implement a
small one-hidden-layer autoencoder from scratch in NumPy (forward + manual
backward, including the sparse-KL and contractive-Jacobian penalties) and a deeper
idiomatic PyTorch version, trained on 8x8 digits.

Variants implemented here:
    - Vanilla (undercomplete) autoencoder
    - Denoising autoencoder (reconstruct clean input from a corrupted one)
    - Sparse autoencoder (KL or L1 penalty on hidden activations)
    - Contractive autoencoder (penalize the encoder Jacobian Frobenius norm)

Training techniques demonstrated:
    - Reconstruction objective (MSE) and its gradient by hand
    - Activation-sparsity penalties (KL divergence, L1)
    - Jacobian (contractive) penalty — robustness of the code to input changes

References:
    - Hinton & Salakhutdinov (2006); Vincent et al. (2008, denoising)
    - Ng (2011, sparse AE notes); Rifai et al. (2011, contractive AE)
"""

from __future__ import annotations

import numpy as np

SEED = 0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


# ---------------------------------------------------------------------------
# 1. NumPy autoencoder (from scratch) — one hidden layer, all four variants
# ---------------------------------------------------------------------------
class AutoencoderNumPy:
    r"""x -> [encoder: sigmoid(xW1+b1)=h] -> [decoder: sigmoid(hW2+b2)=xhat].

    Loss = reconstruction (MSE) + optional regularizer:
        - sparse  : + beta * KL(rho || rho_hat)   (or L1 on h if l1>0)
        - contractive : + lam * ||J_h(x)||_F^2  with J = dh/dx
        - denoising: feed a corrupted x_tilde but score against the clean x.

    We tie nothing (separate W1, W2) and derive every gradient explicitly.
    """

    def __init__(self, d_in, d_hid, lr=0.5, mode="vanilla",
                 beta=3.0, rho=0.05, l1=0.0, lam=1e-3, noise=0.3, seed=SEED):
        rng = np.random.default_rng(seed)
        # small symmetric init (sigmoid units)
        self.W1 = rng.normal(0, np.sqrt(1.0 / d_in), (d_in, d_hid))
        self.b1 = np.zeros(d_hid)
        self.W2 = rng.normal(0, np.sqrt(1.0 / d_hid), (d_hid, d_in))
        self.b2 = np.zeros(d_in)
        self.lr, self.mode = lr, mode
        self.beta, self.rho, self.l1, self.lam, self.noise = beta, rho, l1, lam, noise
        self.rng = rng

    # --- forward -----------------------------------------------------------
    def encode(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.h = sigmoid(self.z1)
        return self.h

    def decode(self, h):
        self.z2 = h @ self.W2 + self.b2
        self.xhat = sigmoid(self.z2)
        return self.xhat

    def forward(self, x):
        return self.decode(self.encode(x))

    # --- one gradient step on a batch -------------------------------------
    def step(self, x_clean):
        n = len(x_clean)
        # denoising: corrupt the INPUT but reconstruct the CLEAN target
        if self.mode == "denoising":
            x_in = x_clean * (self.rng.random(x_clean.shape) > self.noise)
        else:
            x_in = x_clean
        self.x_in = x_in

        h = self.encode(x_in)            # (n, d_hid)
        xhat = self.decode(h)            # (n, d_in)

        # --- reconstruction gradient (MSE: 0.5||xhat - x||^2 per sample) ----
        # dL/dz2 = (xhat - x) * sigmoid'(z2),  sigmoid' = xhat(1-xhat)
        dz2 = (xhat - x_clean) * xhat * (1 - xhat) / n
        dW2 = h.T @ dz2
        db2 = dz2.sum(0)
        dh = dz2 @ self.W2.T             # back to hidden activations
        sp = h * (1 - h)                 # sigmoid'(z1), reused below

        # --- sparsity penalty on hidden activations -------------------------
        if self.mode == "sparse":
            if self.l1 > 0.0:
                # L1: d/dh |h| = sign(h); h>0 for sigmoid so sign=+1
                dh = dh + (self.l1 / n) * np.sign(h)
            else:
                # KL(rho || rho_hat): rho_hat = mean activation per hidden unit
                rho_hat = h.mean(0) + 1e-8
                dkl = (-self.rho / rho_hat + (1 - self.rho) / (1 - rho_hat))  # dKL/drho_hat
                dh = dh + (self.beta / n) * dkl              # broadcast over batch

        dz1 = dh * sp
        dW1 = x_in.T @ dz1
        db1 = dz1.sum(0)

        # --- contractive penalty: lam * ||J||_F^2, J_jk = h_j(1-h_j) W1_kj ---
        # ||J||_F^2 = sum_j (h_j(1-h_j))^2 sum_k W1_kj^2.  Its gradients add to W1,b1.
        if self.mode == "contractive":
            s = sp                                  # (n, d_hid) = h(1-h)
            w2col = np.sum(self.W1 ** 2, axis=0)     # (d_hid,) sum_k W1_kj^2
            # d/dz1 of s^2 = 2 s s' , s' = s(1-2h);  per-sample, summed
            dpen_dz1 = 2.0 * s * (s * (1 - 2 * h)) * w2col      # (n, d_hid)
            dz1_c = dpen_dz1
            dW1 = dW1 + (self.lam / n) * (x_in.T @ dz1_c
                                          + 2.0 * (s ** 2).sum(0) * self.W1)
            db1 = db1 + (self.lam / n) * dz1_c.sum(0)

        # --- SGD update --------------------------------------------------
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def recon_error(self, x):
        xhat = self.forward(x)
        return float(np.mean(np.sum((xhat - x) ** 2, axis=1)))

    def fit(self, X, epochs=60, batch=32, seed=SEED):
        rng = np.random.default_rng(seed)
        self.history = []
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                self.step(X[idx[s:s + batch]])
            self.history.append(self.recon_error(X))
        return self


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (idiomatic) — same four variants, deeper
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

torch.set_num_threads(1)   # keep the tiny CPU demo fast (avoid oversubscription)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AutoencoderTorch(nn.Module):
    """Encoder/decoder MLP. `mode` selects the regularizer used during `fit`."""

    def __init__(self, d_in=64, d_hid=32, d_code=16, mode="vanilla",
                 beta=1e-3, rho=0.05, lam=1e-3, noise=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_code), nn.Sigmoid(),   # code in (0,1) -> sparsity well-defined
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_code, d_hid), nn.ReLU(),
            nn.Linear(d_hid, d_in), nn.Sigmoid(),
        )
        self.mode = mode
        self.beta, self.rho, self.lam, self.noise = beta, rho, lam, noise

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def _kl(self, code):
        rho_hat = code.mean(0).clamp(1e-6, 1 - 1e-6)
        rho = torch.full_like(rho_hat, self.rho)
        return (rho * torch.log(rho / rho_hat)
                + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))).sum()

    def fit(self, X, epochs=40, batch=64, lr=1e-2):
        dev = get_device(); self.to(dev)
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        mse = nn.MSELoss()
        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, batch):
                xb = X[perm[s:s + batch]]
                x_in = xb
                if self.mode == "denoising":
                    x_in = xb * (torch.rand_like(xb) > self.noise)
                if self.mode == "contractive":
                    x_in = x_in.requires_grad_(True)
                code = self.encoder(x_in)
                xhat = self.decoder(code)
                loss = mse(xhat, xb)
                if self.mode == "sparse":
                    loss = loss + self.beta * self._kl(code)
                if self.mode == "contractive":
                    # ||dcode/dx||_F^2 via autograd (sum of squared grads)
                    g = torch.autograd.grad(code.sum(), x_in, create_graph=True)[0]
                    loss = loss + self.lam * (g ** 2).sum() / len(xb)
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def recon_error(self, X):
        self.eval()
        dev = next(self.parameters()).device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        xhat = self(X)
        return float(((xhat - X) ** 2).sum(1).mean())


# ---------------------------------------------------------------------------
# 3. Demo — train each variant on 8x8 digits; report recon error & sparsity
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data / 16.0                       # (1797, 64) in [0,1] -> sigmoid output range
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X)); X = X[perm]
    Xtr, Xte = X[:1400], X[1400:]

    print("NumPy autoencoder on 8x8 digits (64 -> 16 -> 64).")
    print(f"  {'variant':14s} {'test recon err':>15s} {'mean code activ':>17s}")
    trained = {}
    for mode in ["vanilla", "denoising", "sparse", "contractive"]:
        lam = 5e-3 if mode == "contractive" else 1e-3
        ae = AutoencoderNumPy(64, 16, lr=0.5, mode=mode, beta=3.0, rho=0.05,
                              lam=lam, noise=0.3, seed=SEED).fit(Xtr, epochs=40)
        trained[mode] = ae
        code = ae.encode(Xte)
        print(f"  {mode:14s} {ae.recon_error(Xte):15.4f} {code.mean():17.4f}")
    print("  -> all reconstruct well; the SPARSE code has the lowest mean activation")
    print("     (few units fire per input); denoising/contractive trade a little")
    print("     reconstruction for robustness of the learned code.")

    # contractive sanity check: the code should change LESS under input noise.
    # Reuse the models already trained above (no extra training).
    print("\nCode robustness to input perturbation (||dz|| for a small input jitter):")
    jitter = 0.1 * rng.standard_normal(Xte.shape)
    for name in ["vanilla", "contractive"]:
        m = trained[name]
        dz = np.linalg.norm(m.encode(Xte + jitter) - m.encode(Xte), axis=1).mean()
        print(f"  {name:12s}: mean ||z(x+dx) - z(x)|| = {dz:.4f}")
    print("  -> the contractive penalty makes the code less sensitive to input noise.")

    # PyTorch cross-check (deeper net, Adam): all four variants
    print("\nPyTorch autoencoder (64 -> 32 -> 16 -> 32 -> 64), test recon error:")
    for mode in ["vanilla", "denoising", "sparse", "contractive"]:
        ae = AutoencoderTorch(64, 32, 16, mode=mode, beta=1e-2, rho=0.05,
                              lam=1e-3, noise=0.3).fit(Xtr, epochs=30, lr=1e-2)
        print(f"  {mode:14s}: {ae.recon_error(Xte):.4f}")


if __name__ == "__main__":
    demo()

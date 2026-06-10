"""
t-SNE — t-Distributed Stochastic Neighbor Embedding
===================================================
A nonlinear visualization method that places points in 2D/3D so that the
*neighborhood structure* of the high-dimensional data is preserved. It models
similarities as probabilities: a Gaussian kernel in the high-dim space, a
heavy-tailed Student-t kernel in the low-dim map, and minimizes the KL
divergence between the two distributions by gradient descent. The heavy tail is
the trick that fixes the "crowding problem" and yields the famous well-separated
clusters.

Variants implemented here:
    - From-scratch NumPy t-SNE (perplexity-calibrated P, Student-t Q, KL grad)
    - PyTorch / autograd t-SNE (the same objective, gradients via backprop)
    - Early exaggeration (multiply P early to form tight clusters)

Training techniques demonstrated:
    - Perplexity calibration via binary search on per-point Gaussian bandwidth
    - Early exaggeration and momentum for gradient descent on a non-convex loss

References:
    - van der Maaten & Hinton (2008), "Visualizing Data using t-SNE"
    - Hinton & Roweis (2002), "Stochastic Neighbor Embedding"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _pairwise_sq_dists(X):
    """||x_i - x_j||^2 for all pairs, via the (a-b)^2 = a^2 - 2ab + b^2 identity."""
    sum_sq = (X ** 2).sum(1)
    D = sum_sq[:, None] - 2 * X @ X.T + sum_sq[None, :]
    return np.maximum(D, 0)


def _binary_search_perplexity(D, perplexity, tol=1e-5, max_iter=50):
    r"""For each point, find the Gaussian precision (beta = 1/2sigma^2) so the
    conditional distribution has the target perplexity.

    Perplexity = 2^{H(P_i)} where H is the Shannon entropy of the row P_{j|i}.
    We binary-search beta_i so that H matches log2(perplexity)."""
    n = D.shape[0]
    P = np.zeros((n, n))
    target = np.log(perplexity)                 # work with entropy in nats
    for i in range(n):
        beta_min, beta_max = -np.inf, np.inf
        beta = 1.0
        Di = np.delete(D[i], i)                  # distances to other points
        for _ in range(max_iter):
            Pi = np.exp(-Di * beta)
            sum_Pi = Pi.sum() + 1e-12
            # entropy H = -sum p log p, with p = Pi/sum_Pi
            H = np.log(sum_Pi) + beta * (Di * Pi).sum() / sum_Pi
            diff = H - target
            if abs(diff) < tol:
                break
            if diff > 0:                         # entropy too high -> increase beta
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2
        Pi = Pi / sum_Pi
        P[i, np.arange(n) != i] = Pi
    return P


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
class TSNENumPy:
    r"""
    High-dim affinities: symmetric joint P_{ij} = (P_{j|i}+P_{i|j})/(2n), where
    P_{j|i} is a Gaussian whose bandwidth is set per point by perplexity.

    Low-dim affinities: a Student-t (1 degree of freedom) kernel
        q_{ij} = (1+||y_i-y_j||^2)^{-1} / sum_{k!=l} (1+||y_k-y_l||^2)^{-1}.

    Objective: KL(P || Q) = sum_ij p_ij log(p_ij / q_ij).
    Gradient:  dC/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j)(1+||y_i-y_j||^2)^{-1}.
    """

    def __init__(self, n_components=2, perplexity=30.0, n_iter=500, lr=200.0,
                 early_exaggeration=12.0, exaggerate_iter=100, seed=SEED):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.lr = lr
        self.early_exaggeration = early_exaggeration
        self.exaggerate_iter = exaggerate_iter
        self.seed = seed

    def _high_dim_P(self, X):
        D = _pairwise_sq_dists(X)
        P_cond = _binary_search_perplexity(D, self.perplexity)
        P = (P_cond + P_cond.T)                 # symmetrize
        P = P / (P.sum() + 1e-12)               # joint distribution, sums to 1
        return np.maximum(P, 1e-12)

    def fit_transform(self, X):
        X = np.asarray(X, float)
        rng = np.random.default_rng(self.seed)
        n = len(X)
        P = self._high_dim_P(X)

        # init the embedding from a small Gaussian
        Y = rng.normal(0, 1e-4, (n, self.n_components))
        velocity = np.zeros_like(Y)             # momentum buffer

        for it in range(self.n_iter):
            # early exaggeration: inflate P so clusters separate early on
            P_eff = P * self.early_exaggeration if it < self.exaggerate_iter else P

            # low-dim Student-t affinities
            num = 1.0 / (1.0 + _pairwise_sq_dists(Y))   # (1 + d^2)^{-1}
            np.fill_diagonal(num, 0.0)
            Q = num / (num.sum() + 1e-12)
            Q = np.maximum(Q, 1e-12)

            # gradient of KL(P||Q): 4 * sum_j (p-q) * num * (y_i - y_j)
            PQ = (P_eff - Q) * num                       # (n,n)
            grad = 4.0 * (np.diag(PQ.sum(1)) - PQ) @ Y   # vectorized form

            momentum = 0.5 if it < 250 else 0.8
            velocity = momentum * velocity - self.lr * grad
            Y = Y + velocity
            Y = Y - Y.mean(0)                            # recenter (translation-invariant)

        # store final KL for diagnostics
        num = 1.0 / (1.0 + _pairwise_sq_dists(Y))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / (num.sum() + 1e-12), 1e-12)
        self.kl_divergence_ = float((P * np.log(P / Q)).sum())
        return Y


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (autograd does the KL gradient for us)
# ---------------------------------------------------------------------------
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tsne_torch(X, n_components=2, perplexity=30.0, n_iter=500, lr=200.0,
               early_exaggeration=12.0, exaggerate_iter=100, seed=SEED):
    """Same objective, but the KL gradient is obtained by autograd. The high-dim
    P is built once with the NumPy helper; only the low-dim map is optimized."""
    dev = get_device()
    X = np.asarray(X, float)
    n = len(X)
    D = _pairwise_sq_dists(X)
    P = _binary_search_perplexity(D, perplexity)
    P = P + P.T
    P = P / (P.sum() + 1e-12)
    P = np.maximum(P, 1e-12)
    Pt = torch.as_tensor(P, dtype=torch.float32, device=dev)

    g = torch.Generator(device="cpu").manual_seed(seed)
    Y = (torch.randn(n, n_components, generator=g) * 1e-4).to(dev).requires_grad_(True)
    opt = torch.optim.SGD([Y], lr=lr, momentum=0.5)
    eye = torch.eye(n, device=dev, dtype=torch.bool)

    for it in range(n_iter):
        for grp in opt.param_groups:
            grp["momentum"] = 0.5 if it < 250 else 0.8
        P_eff = Pt * early_exaggeration if it < exaggerate_iter else Pt
        # squared pairwise distances via ||a-b||^2 = |a|^2 - 2 a.b + |b|^2
        # (much faster than torch.cdist under autograd on CPU)
        sq = (Y ** 2).sum(1)
        d2 = (sq[:, None] - 2.0 * (Y @ Y.T) + sq[None, :]).clamp_min(0.0)
        num = 1.0 / (1.0 + d2)
        num = num.masked_fill(eye, 0.0)
        Q = (num / (num.sum() + 1e-12)).clamp_min(1e-12)
        loss = (P_eff * (P_eff.clamp_min(1e-12) / Q).log()).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            Y -= Y.mean(0)
    return Y.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    from sklearn.datasets import load_digits
    from sklearn.manifold import trustworthiness

    digits = load_digits()
    # subsample to ~300 points so the O(n^2) loop stays fast on CPU
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(digits.data), 300, replace=False)
    X, y = digits.data[idx], digits.target[idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    ts = TSNENumPy(n_components=2, perplexity=30, n_iter=300)
    Y = ts.fit_transform(X)
    tw = trustworthiness(X, Y, n_neighbors=5)
    print(f"NumPy t-SNE  KL={ts.kl_divergence_:.3f}  trustworthiness={tw:.3f}")

    # The torch path optimizes via autograd, which is heavier per step on CPU.
    # Run it on a smaller subset with fewer iterations to stay in budget; it
    # demonstrates the identical objective, just at smaller scale.
    Xs, ys = X[:150], y[:150]
    Yt = tsne_torch(Xs, perplexity=30, n_iter=250, exaggerate_iter=60)
    twt = trustworthiness(Xs, Yt, n_neighbors=5)
    print(f"Torch t-SNE (n=150)  trustworthiness={twt:.3f}")

    # A good embedding keeps same-digit points close: compare mean intra- vs
    # inter-class distances in the map.
    from itertools import combinations
    cen = np.stack([Y[y == c].mean(0) for c in np.unique(y)])
    intra = np.mean([np.linalg.norm(Y[i] - cen[y[i]]) for i in range(len(Y))])
    inter = np.mean([np.linalg.norm(cen[a] - cen[b]) for a, b in combinations(range(len(cen)), 2)])
    print(f"map intra-class={intra:.2f}  inter-class={inter:.2f}  ratio={inter/intra:.2f}")


if __name__ == "__main__":
    demo()

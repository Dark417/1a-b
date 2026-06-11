"""
Gaussian Mixture Model (GMM) via Expectation-Maximization
=========================================================
Model the data as a weighted mixture of K Gaussians and fit it by EM. Unlike
k-means (hard, spherical assignments), a GMM gives *soft* responsibilities and
*elliptical* clusters with their own covariance — k-means is the limiting case
of a GMM with shared spherical covariance and hardened responsibilities.

Variants implemented here:
    - Full covariance per component
    - Diagonal covariance per component (fewer parameters, axis-aligned)

Training techniques demonstrated:
    - EM as monotone ascent on the (incomplete-data) log-likelihood
    - The ELBO / Jensen lower bound and why EM never decreases the likelihood
    - Covariance regularization (reg_covar) for numerical stability
    - k-means++-style initialization of means

References:
    - Dempster, Laird & Rubin (1977), "Maximum Likelihood from Incomplete Data via EM"
    - Bishop, PRML, ch. 9
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch — the EM algorithm by hand)
# ---------------------------------------------------------------------------
class GMMNumPy:
    r"""
    A mixture of K Gaussians:

        p(x) = sum_k  pi_k  N(x | mu_k, Sigma_k),     sum_k pi_k = 1.

    EM introduces latent assignments z_i in {1..K} and alternates:

      E-step: responsibilities  gamma_ik = p(z_i=k | x_i)
                = pi_k N(x_i|mu_k,Sig_k) / sum_j pi_j N(x_i|mu_j,Sig_j).

      M-step: weighted MLE with weights gamma_ik:
                N_k  = sum_i gamma_ik
                mu_k = (1/N_k) sum_i gamma_ik x_i
                Sig_k= (1/N_k) sum_i gamma_ik (x_i-mu_k)(x_i-mu_k)^T
                pi_k = N_k / n.

    Each full E+M step never decreases the data log-likelihood.
    """

    def __init__(self, n_components=3, covariance_type="full", n_iters=200,
                 tol=1e-6, reg_covar=1e-6, n_init=3, seed=SEED):
        self.K = n_components
        self.covariance_type = covariance_type
        self.n_iters, self.tol = n_iters, tol
        self.reg_covar, self.n_init, self.seed = reg_covar, n_init, seed

    # --- log N(x | mu, Sigma) for every point/component, shape (n, K) ----
    def _log_gauss(self, X, means, covs):
        n, d = X.shape
        out = np.empty((n, self.K))
        log2pi = d * np.log(2 * np.pi)
        for k in range(self.K):
            diff = X - means[k]                       # (n, d)
            if self.covariance_type == "diag":
                var = covs[k]                         # (d,) diagonal variances
                # (x-mu)^T Sig^{-1} (x-mu) = sum_j diff_j^2 / var_j
                maha = (diff * diff / var).sum(1)
                logdet = np.log(var).sum()
            else:                                     # full covariance
                # solve via Cholesky for stability:  Sig = L L^T
                L = np.linalg.cholesky(covs[k])
                # solve L y = diff^T  -> maha = ||y||^2 , logdet = 2 sum log L_ii
                y = np.linalg.solve(L, diff.T)        # (d, n)
                maha = (y * y).sum(0)
                logdet = 2.0 * np.log(np.diag(L)).sum()
            out[:, k] = -0.5 * (log2pi + logdet + maha)
        return out

    def _init_means_kpp(self, X, rng):
        # k-means++ seeding: spread the initial means out by D^2 sampling.
        means = [X[rng.integers(len(X))]]
        for _ in range(1, self.K):
            d2 = np.min([((X - m) ** 2).sum(1) for m in means], axis=0)
            probs = d2 / d2.sum()
            means.append(X[rng.choice(len(X), p=probs)])
        return np.array(means, float)

    def _init_params(self, X, rng):
        n, d = X.shape
        means = self._init_means_kpp(X, rng)
        weights = np.full(self.K, 1.0 / self.K)
        gvar = X.var(0) + self.reg_covar               # global spread
        if self.covariance_type == "diag":
            covs = np.tile(gvar, (self.K, 1))
        else:
            covs = np.array([np.diag(gvar) for _ in range(self.K)])
        return weights, means, covs

    def _e_step(self, X, weights, means, covs):
        # log-responsibilities, normalized with the log-sum-exp trick.
        log_w = np.log(weights + 1e-300)
        log_p = self._log_gauss(X, means, covs) + log_w   # (n, K) joint log p(x,z)
        log_norm = _logsumexp(log_p, axis=1, keepdims=True)  # log p(x)
        log_gamma = log_p - log_norm
        ll = log_norm.sum()                               # incomplete-data LL
        return np.exp(log_gamma), ll

    def _m_step(self, X, gamma):
        n, d = X.shape
        Nk = gamma.sum(0) + 1e-300                         # (K,) soft counts
        weights = Nk / n
        means = (gamma.T @ X) / Nk[:, None]                # (K, d)
        if self.covariance_type == "diag":
            covs = np.empty((self.K, d))
            for k in range(self.K):
                diff = X - means[k]
                covs[k] = (gamma[:, k, None] * diff * diff).sum(0) / Nk[k] + self.reg_covar
        else:
            covs = np.empty((self.K, d, d))
            for k in range(self.K):
                diff = X - means[k]                        # (n, d)
                covs[k] = (gamma[:, k, None] * diff).T @ diff / Nk[k]
                covs[k] += self.reg_covar * np.eye(d)      # regularize
        return weights, means, covs

    def _fit_once(self, X, rng):
        weights, means, covs = self._init_params(X, rng)
        prev_ll = -np.inf
        history = []
        for _ in range(self.n_iters):
            gamma, ll = self._e_step(X, weights, means, covs)
            history.append(ll)
            weights, means, covs = self._m_step(X, gamma)
            if ll - prev_ll < self.tol and prev_ll > -np.inf:
                break
            prev_ll = ll
        # final responsibilities/LL after last M-step
        gamma, ll = self._e_step(X, weights, means, covs)
        return weights, means, covs, gamma, ll, history

    def fit(self, X):
        X = np.asarray(X, float)
        best_ll = -np.inf
        for i in range(self.n_init):
            rng = np.random.default_rng(self.seed + i)
            w, m, c, g, ll, hist = self._fit_once(X, rng)
            if ll > best_ll:
                best_ll = ll
                self.weights_, self.means_, self.covariances_ = w, m, c
                self.responsibilities_ = g
                self.log_likelihood_ = ll
                self.history_ = hist
        self.labels_ = self.responsibilities_.argmax(1)
        return self

    def predict_proba(self, X):
        g, _ = self._e_step(np.asarray(X, float), self.weights_,
                            self.means_, self.covariances_)
        return g

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def score(self, X):
        _, ll = self._e_step(np.asarray(X, float), self.weights_,
                            self.means_, self.covariances_)
        return ll / len(X)

    def bic(self, X):
        # BIC = -2 LL + p log n ; rewards fit, penalizes #free parameters p.
        X = np.asarray(X, float)
        n, d = X.shape
        if self.covariance_type == "diag":
            cov_params = self.K * d
        else:
            cov_params = self.K * d * (d + 1) // 2
        p = (self.K - 1) + self.K * d + cov_params          # weights+means+cov
        ll = self.log_likelihood_
        return -2 * ll + p * np.log(n)


def _logsumexp(a, axis=None, keepdims=False):
    """Numerically stable log(sum(exp(a)))."""
    amax = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    out = out + amax
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (vectorized, GPU-friendly EM)
# ---------------------------------------------------------------------------
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GMMTorch:
    """Full-covariance EM on torch tensors (no autograd; closed-form updates)."""

    def __init__(self, n_components=3, n_iters=200, tol=1e-6, reg_covar=1e-6, seed=SEED):
        self.K, self.n_iters = n_components, n_iters
        self.tol, self.reg_covar, self.seed = tol, reg_covar, seed
        self.device = get_device()

    def _log_gauss(self, X, means, covs):
        n, d = X.shape
        log2pi = d * np.log(2 * np.pi)
        out = torch.empty((n, self.K), device=X.device, dtype=X.dtype)
        for k in range(self.K):
            diff = X - means[k]
            L = torch.linalg.cholesky(covs[k])
            y = torch.linalg.solve_triangular(L, diff.T, upper=False)  # (d,n)
            maha = (y * y).sum(0)
            logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
            out[:, k] = -0.5 * (log2pi + logdet + maha)
        return out

    def fit(self, X):
        torch.manual_seed(self.seed)
        Xt = torch.as_tensor(np.asarray(X, float), dtype=torch.float64, device=self.device)
        n, d = Xt.shape
        # init means at random distinct points; covs = global covariance.
        idx = torch.randperm(n, device=self.device)[:self.K]
        means = Xt[idx].clone()
        gvar = Xt.var(0, unbiased=False)
        covs = torch.stack([torch.diag(gvar) for _ in range(self.K)])
        weights = torch.full((self.K,), 1.0 / self.K, dtype=Xt.dtype, device=self.device)
        prev = -float("inf")
        for _ in range(self.n_iters):
            # E-step
            log_p = self._log_gauss(Xt, means, covs) + torch.log(weights + 1e-300)
            log_norm = torch.logsumexp(log_p, dim=1, keepdim=True)
            gamma = torch.exp(log_p - log_norm)
            ll = log_norm.sum().item()
            # M-step
            Nk = gamma.sum(0) + 1e-300
            weights = Nk / n
            means = (gamma.T @ Xt) / Nk[:, None]
            covs = torch.empty((self.K, d, d), dtype=Xt.dtype, device=self.device)
            for k in range(self.K):
                diff = Xt - means[k]
                covs[k] = (gamma[:, k, None] * diff).T @ diff / Nk[k]
                covs[k] += self.reg_covar * torch.eye(d, dtype=Xt.dtype, device=self.device)
            if abs(ll - prev) < self.tol and prev > -float("inf"):
                break
            prev = ll
        self.weights_ = weights.cpu().numpy()
        self.means_ = means.cpu().numpy()
        self.covariances_ = covs.cpu().numpy()
        self.labels_ = gamma.argmax(1).cpu().numpy()
        self.log_likelihood_ = ll
        return self


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    # anisotropic blobs: GMM (elliptical) should beat spherical k-means here.
    X, ytrue = make_blobs(n_samples=600, centers=3, cluster_std=1.0, random_state=SEED)
    transform = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = X @ transform

    for cov in ("full", "diag"):
        gmm = GMMNumPy(n_components=3, covariance_type=cov).fit(X)
        ari = adjusted_rand_score(ytrue, gmm.labels_)
        mono = all(b - a >= -1e-6 for a, b in zip(gmm.history_, gmm.history_[1:]))
        print(f"NumPy GMM ({cov:4s}): ARI={ari:.3f}  logL/n={gmm.score(X):.3f}  "
              f"BIC={gmm.bic(X):.1f}  monotone_LL={mono}")

    gt = GMMTorch(n_components=3).fit(X)
    print(f"Torch GMM (full): ARI={adjusted_rand_score(ytrue, gt.labels_):.3f}  "
          f"logL={gt.log_likelihood_:.1f}  device={gt.device}")

    print("\nModel selection by BIC (full covariance):")
    for k in range(2, 6):
        g = GMMNumPy(n_components=k, covariance_type="full").fit(X)
        print(f"  K={k}: BIC={g.bic(X):.1f}")


if __name__ == "__main__":
    demo()

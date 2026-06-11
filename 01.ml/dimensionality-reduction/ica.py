"""
Independent Component Analysis (FastICA)
========================================
Blind source separation: given mixtures X = A S of unknown independent sources
S (the "cocktail party problem"), recover the sources without knowing the
mixing matrix A. ICA finds an unmixing matrix W so that the rows of S ≈ W X are
statistically **independent** and maximally **non-Gaussian**.

Why non-Gaussianity? By the Central Limit Theorem, a sum (mixture) of
independent variables is *more Gaussian* than the originals; so maximizing
non-Gaussianity of the projections un-mixes them. We measure non-Gaussianity by
negentropy, approximated through nonlinear contrast functions (here: logcosh
and the exp/Gaussian and kurtosis variants).

Variants implemented here:
    - FastICA fixed-point iteration with deflation (one component at a time)
    - FastICA symmetric (all components in parallel, symmetric decorrelation)
    - Contrast functions: logcosh (G=log cosh), exp (G=-exp(-u²/2)), kurtosis
    - PCA whitening as the mandatory pre-processing step

Training techniques demonstrated:
    - Whitening (decorrelate + unit variance) via eigendecomposition of the
      covariance — turns ICA into a search over orthogonal matrices
    - Fixed-point (Newton-like) iteration; symmetric decorrelation by W (WᵀW)^-1/2

References:
    - Hyvärinen & Oja (2000), "Independent Component Analysis: Algorithms and
      Applications" (the FastICA paper)
    - Hyvärinen, Karhunen & Oja, "Independent Component Analysis" (book)
"""

from __future__ import annotations

import numpy as np

SEED = 0


# --- contrast functions G and their derivatives g, g' ----------------------
def _logcosh(u, alpha=1.0):
    # G(u) = (1/a) log cosh(a u);  g = tanh(a u);  g' = a (1 - tanh²)
    gu = np.tanh(alpha * u)
    g_prime = alpha * (1.0 - gu ** 2)
    return gu, g_prime


def _exp(u):
    # G(u) = -exp(-u²/2);  g = u exp(-u²/2);  g' = (1 - u²) exp(-u²/2)
    e = np.exp(-(u ** 2) / 2.0)
    return u * e, (1.0 - u ** 2) * e


def _cube(u):
    # kurtosis-based contrast G(u)=u⁴/4;  g = u³;  g' = 3u²
    return u ** 3, 3.0 * u ** 2


_CONTRASTS = {"logcosh": _logcosh, "exp": _exp, "cube": _cube}


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch)
# ---------------------------------------------------------------------------
class FastICANumPy:
    r"""
    FastICA.  Steps:

    1. **Center** X (zero mean per feature).
    2. **Whiten**: find K so that  X_w = K X  has identity covariance. Using the
       eigendecomposition of the covariance  C = E D Eᵀ,  K = D^{-1/2} Eᵀ.
       After whitening the unmixing reduces to finding an *orthogonal* W.
    3. **Fixed-point iteration** for each unit-norm row w:
           w⁺ = E[ x_w g(wᵀx_w) ] − E[ g'(wᵀx_w) ] w,
           w  = w⁺ / ‖w⁺‖.
       This is a Newton step maximizing the negentropy approximation
       J(wᵀx) ≈ [E{G(wᵀx)} − E{G(ν)}]²  (ν standard Gaussian).
    4. Deflation: orthogonalize each new w against the already-found ones
       (Gram–Schmidt) so components stay distinct.
    """

    def __init__(self, n_components=None, fun="logcosh", max_iter=200,
                 tol=1e-5, algorithm="deflation", seed=SEED):
        self.n_components = n_components
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.seed = seed

    def _whiten(self, X):
        # X is (features, samples). Center across samples.
        self.mean_ = X.mean(1, keepdims=True)
        Xc = X - self.mean_
        cov = (Xc @ Xc.T) / Xc.shape[1]            # (d, d) covariance
        d, E = np.linalg.eigh(cov)                  # eigvals ascending
        d = np.maximum(d, 1e-12)
        # whitening matrix K = D^{-1/2} Eᵀ so that cov(K Xc) = I
        K = np.diag(1.0 / np.sqrt(d)) @ E.T
        Xw = K @ Xc                                 # whitened: identity covariance
        return Xw, K

    def _g(self, u):
        return _CONTRASTS[self.fun](u)

    def fit(self, X):
        # Accept X as (samples, features); work internally as (features, samples).
        X = np.asarray(X, float).T
        n_features = X.shape[0]
        k = self.n_components or n_features
        rng = np.random.default_rng(self.seed)

        Xw, K = self._whiten(X)
        m = Xw.shape[1]                             # number of samples

        if self.algorithm == "deflation":
            W = np.zeros((k, Xw.shape[0]))
            for i in range(k):
                w = rng.standard_normal(Xw.shape[0])
                w /= np.linalg.norm(w)
                for _ in range(self.max_iter):
                    wx = w @ Xw                     # projection (1, m)
                    g, gp = self._g(wx)
                    # fixed-point update (the FastICA Newton step)
                    w_new = (Xw * g).mean(1) - gp.mean() * w
                    # Gram–Schmidt: remove components already found (deflation)
                    w_new -= W[:i].T @ (W[:i] @ w_new)
                    w_new /= np.linalg.norm(w_new) + 1e-12
                    if np.abs(np.abs(w_new @ w) - 1.0) < self.tol:
                        w = w_new
                        break
                    w = w_new
                W[i] = w
        else:  # symmetric: update all rows, then symmetric decorrelation
            W = rng.standard_normal((k, Xw.shape[0]))
            W = self._sym_decorrelate(W)
            for _ in range(self.max_iter):
                WX = W @ Xw                         # (k, m)
                g, gp = self._g(WX)
                W_new = (g @ Xw.T) / m - (gp.mean(1)[:, None] * W)
                W_new = self._sym_decorrelate(W_new)
                # convergence: largest change in alignment
                lim = np.max(np.abs(np.abs(np.einsum("ij,ij->i", W_new, W)) - 1.0))
                W = W_new
                if lim < self.tol:
                    break

        self.components_ = W                        # unmixing in whitened space
        self.whitening_ = K
        self.unmixing_ = W @ K                      # full unmixing: S = (W K)(X - mean)
        self.mixing_ = np.linalg.pinv(self.unmixing_)
        return self

    @staticmethod
    def _sym_decorrelate(W):
        # W <- (W Wᵀ)^{-1/2} W  (makes rows orthonormal without preferring order)
        s, U = np.linalg.eigh(W @ W.T)
        s = np.maximum(s, 1e-12)
        return (U * (1.0 / np.sqrt(s))) @ U.T @ W

    def transform(self, X):
        X = np.asarray(X, float).T
        S = self.unmixing_ @ (X - self.mean_)
        return S.T                                  # (samples, components)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# 2. PyTorch implementation
# ---------------------------------------------------------------------------
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fastica_torch(X, n_components=None, fun="logcosh", max_iter=200,
                  tol=1e-5, seed=SEED):
    r"""
    FastICA (symmetric) with torch tensors. Same whitening + fixed-point math;
    uses torch.linalg for the eigendecompositions. The fixed-point map is not a
    gradient step, so we run it explicitly (no autograd needed).
    """
    dev = get_device()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64, device=dev).T
    d, m = Xt.shape
    k = n_components or d

    mean = Xt.mean(1, keepdim=True)
    Xc = Xt - mean
    cov = (Xc @ Xc.T) / m
    evals, E = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=1e-12)
    K = torch.diag(1.0 / torch.sqrt(evals)) @ E.T
    Xw = K @ Xc

    def g_fn(u):
        if fun == "logcosh":
            gu = torch.tanh(u)
            return gu, (1.0 - gu ** 2)
        if fun == "exp":
            e = torch.exp(-(u ** 2) / 2)
            return u * e, (1.0 - u ** 2) * e
        return u ** 3, 3.0 * u ** 2                 # cube / kurtosis

    def sym_decorrelate(W):
        s, U = torch.linalg.eigh(W @ W.T)
        s = torch.clamp(s, min=1e-12)
        return (U * (1.0 / torch.sqrt(s))) @ U.T @ W

    gen = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(k, d, generator=gen).to(torch.float64).to(dev)
    W = sym_decorrelate(W)
    for _ in range(max_iter):
        WX = W @ Xw
        g, gp = g_fn(WX)
        W_new = (g @ Xw.T) / m - gp.mean(1)[:, None] * W
        W_new = sym_decorrelate(W_new)
        lim = (torch.abs(torch.einsum("ij,ij->i", W_new, W)) - 1.0).abs().max()
        W = W_new
        if lim < tol:
            break

    unmix = W @ K
    S = unmix @ Xc
    return S.T.cpu().numpy(), unmix.cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo — the classic blind source separation of mixed signals
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Build 3 independent, non-Gaussian source signals.
    n = 2000
    t = np.linspace(0, 8, n)
    s1 = np.sin(2 * t)                              # sinusoid
    s2 = np.sign(np.sin(3 * t))                     # square wave
    rng = np.random.default_rng(SEED)
    s3 = rng.uniform(-1, 1, n)                      # uniform noise (saw-like)
    S = np.c_[s1, s2, s3]
    S /= S.std(0)                                   # unit variance per source

    # Mix them with a random mixing matrix A: X = S Aᵀ.
    A = np.array([[1.0, 1.0, 1.0],
                  [0.5, 2.0, 1.0],
                  [1.5, 1.0, 2.0]])
    X = S @ A.T

    # Recover sources with FastICA (deflation).
    ica = FastICANumPy(n_components=3, fun="logcosh", algorithm="deflation").fit(X)
    S_hat = ica.transform(X)

    # ICA has sign + permutation ambiguity; match recovered to true sources by
    # absolute correlation and report the best alignment.
    def best_corr(S_true, S_est):
        C = np.abs(np.corrcoef(S_true.T, S_est.T)[:3, 3:])   # 3x3 |corr|
        # greedily match each true source to its best estimate
        used, total = set(), 0.0
        for _ in range(3):
            i, j = np.unravel_index(np.argmax(C), C.shape)
            total += C[i, j]
            C[i, :] = -1
            C[:, j] = -1
        return total / 3

    print(f"deflation  mean |corr| recovered vs true: {best_corr(S, S_hat):.3f}")

    # Symmetric algorithm and a different contrast function.
    S_sym = FastICANumPy(3, fun="exp", algorithm="symmetric").fit_transform(X)
    print(f"symmetric  mean |corr| recovered vs true: {best_corr(S, S_sym):.3f}")

    # PCA cannot separate them (uncorrelated ≠ independent).
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    S_pca = Xc @ Vt.T
    print(f"PCA        mean |corr| recovered vs true: {best_corr(S, S_pca):.3f}"
          f"  (worse: PCA only decorrelates)")

    # Torch FastICA agrees.
    S_t, _ = fastica_torch(X, n_components=3, fun="logcosh")
    print(f"torch ICA  mean |corr| recovered vs true: {best_corr(S, S_t):.3f}")

    # Sanity: recovered sources are nearly uncorrelated with each other.
    off = np.abs(np.corrcoef(S_hat.T) - np.eye(3)).max()
    print(f"max off-diagonal |corr| among recovered: {off:.3f}")


if __name__ == "__main__":
    demo()

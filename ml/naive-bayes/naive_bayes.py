"""
Naive Bayes
===========
A generative classifier built straight from Bayes' rule with one bold
simplifying assumption: given the class, the features are **conditionally
independent**. That assumption makes the class-conditional likelihood factorize
into a product of per-feature terms, so fitting is just counting / computing
simple statistics — there is no iterative optimization. Despite being "naive",
it is fast, needs little data, and is a strong baseline for text. We implement
the three classic flavours from scratch in NumPy — **Gaussian** (continuous
features), **Multinomial** (counts, e.g. bag-of-words), **Bernoulli** (binary
presence/absence) — plus a vectorized PyTorch version that computes the
log-probabilities on tensors.

Variants implemented here:
    - GaussianNB    — continuous features, per-class Gaussian per feature
    - MultinomialNB — count features (term frequencies), Laplace smoothing
    - BernoulliNB   — binary features, with the explicit "absent" term
    - PyTorch GaussianNB — same model, vectorized log-prob on tensors

Training techniques demonstrated:
    - Maximum-likelihood parameter estimation (closed form, no gradients)
    - Log-space computation for numerical stability (log-sum-exp)
    - Laplace / additive smoothing to avoid zero probabilities

References:
    - Manning, Raghavan, Schütze (2008), "Introduction to Information Retrieval", ch. 13
    - McCallum & Nigam (1998), "A comparison of event models for Naive Bayes text classification"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _logsumexp(A, axis=1):
    """Numerically stable log(sum(exp(A))) along `axis`."""
    m = A.max(axis=axis, keepdims=True)
    return (m.squeeze(axis) + np.log(np.exp(A - m).sum(axis=axis)))


# ---------------------------------------------------------------------------
# 1. NumPy implementations (from scratch)
# ---------------------------------------------------------------------------
# Bayes' rule:   P(y=c | x) ∝ P(y=c) * P(x | y=c).
# Naive assumption:  P(x | y=c) = Π_j P(x_j | y=c).
# We always work in LOG space:
#   log P(y=c | x) = log P(y=c) + Σ_j log P(x_j | y=c)  - log Z
# and pick argmax_c (the normalizer log Z is the same for all c).
class GaussianNB:
    r"""
    Continuous features. Per class c and feature j, model x_j ~ N(mu_cj, var_cj).

    MLE on the training points of class c:
        mu_cj  = mean_i x_ij,   var_cj = mean_i (x_ij - mu_cj)^2.
    Log-likelihood of a feature:
        log N(x; mu, var) = -1/2 [ log(2π var) + (x-mu)^2 / var ].
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing      # add to variances for stability

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        n = len(X)
        self.theta_ = []     # mean per class:  (n_classes, n_features)
        self.sigma_ = []     # variance per class
        self.log_prior_ = []
        eps = self.var_smoothing * X.var(0).max()
        for c in self.classes_:
            Xc = X[y == c]
            self.theta_.append(Xc.mean(0))
            self.sigma_.append(Xc.var(0) + eps)
            self.log_prior_.append(np.log(len(Xc) / n))     # log P(y=c)
        self.theta_ = np.array(self.theta_)
        self.sigma_ = np.array(self.sigma_)
        self.log_prior_ = np.array(self.log_prior_)
        return self

    def _joint_log_likelihood(self, X):
        # log P(y=c) + Σ_j log N(x_j; mu_cj, var_cj), for every class c
        jll = []
        for k in range(len(self.classes_)):
            mu, var = self.theta_[k], self.sigma_[k]
            ll = -0.5 * (np.log(2 * np.pi * var) + (X - mu) ** 2 / var).sum(1)
            jll.append(self.log_prior_[k] + ll)
        return np.array(jll).T          # (n_samples, n_classes)

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(np.asarray(X, float))
        return jll - _logsumexp(jll, axis=1)[:, None]      # normalize -> log P(y|x)

    def predict(self, X):
        return self.classes_[self._joint_log_likelihood(np.asarray(X, float)).argmax(1)]


class MultinomialNB:
    r"""
    Count features (e.g. bag-of-words term frequencies). Each class is a
    multinomial over the vocabulary.

    With Laplace (additive) smoothing alpha:
        P(j | c) = (count of feature j in class c + alpha)
                   / (total counts in class c + alpha * n_features).
    For a document x (a count vector):
        log P(x | c) = Σ_j x_j * log P(j | c)   (multinomial coeff drops in argmax).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha             # smoothing pseudo-count

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        n, d = X.shape
        self.feature_log_prob_ = []    # log P(j | c)
        self.log_prior_ = []
        for c in self.classes_:
            Xc = X[y == c]
            counts = Xc.sum(0) + self.alpha          # smoothed per-feature counts
            self.feature_log_prob_.append(np.log(counts) - np.log(counts.sum()))
            self.log_prior_.append(np.log(len(Xc) / n))
        self.feature_log_prob_ = np.array(self.feature_log_prob_)
        self.log_prior_ = np.array(self.log_prior_)
        return self

    def _joint_log_likelihood(self, X):
        # log P(y=c) + Σ_j x_j log P(j|c)  ==  prior + X @ feature_log_prob^T
        return X @ self.feature_log_prob_.T + self.log_prior_

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(np.asarray(X, float))
        return jll - _logsumexp(jll, axis=1)[:, None]

    def predict(self, X):
        return self.classes_[self._joint_log_likelihood(np.asarray(X, float)).argmax(1)]


class BernoulliNB:
    r"""
    Binary features (presence/absence). Each feature is a Bernoulli per class.

    Smoothed parameter:
        p_cj = (count of docs in class c with feature j + alpha)
               / (n docs in class c + 2 alpha).
    Likelihood (note the explicit ABSENT term — what makes it differ from
    Multinomial):
        log P(x | c) = Σ_j [ x_j log p_cj + (1 - x_j) log(1 - p_cj) ].
    """

    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize       # threshold to binarize inputs

    def _binarize(self, X):
        X = np.asarray(X, float)
        return (X > self.binarize).astype(float) if self.binarize is not None else X

    def fit(self, X, y):
        X = self._binarize(X)
        self.classes_ = np.unique(y)
        n = len(X)
        self.feature_log_prob_ = []        # log p_cj
        self.neg_log_prob_ = []            # log(1 - p_cj)
        self.log_prior_ = []
        for c in self.classes_:
            Xc = X[y == c]
            p = (Xc.sum(0) + self.alpha) / (len(Xc) + 2 * self.alpha)
            self.feature_log_prob_.append(np.log(p))
            self.neg_log_prob_.append(np.log(1 - p))
            self.log_prior_.append(np.log(len(Xc) / n))
        self.feature_log_prob_ = np.array(self.feature_log_prob_)
        self.neg_log_prob_ = np.array(self.neg_log_prob_)
        self.log_prior_ = np.array(self.log_prior_)
        return self

    def _joint_log_likelihood(self, X):
        X = self._binarize(X)
        # Σ_j x_j log p + (1-x_j) log(1-p) = X@logp^T + (1-X)@log(1-p)^T
        present = X @ self.feature_log_prob_.T
        absent = (1 - X) @ self.neg_log_prob_.T
        return present + absent + self.log_prior_

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        return jll - _logsumexp(jll, axis=1)[:, None]

    def predict(self, X):
        return self.classes_[self._joint_log_likelihood(X).argmax(1)]


# ---------------------------------------------------------------------------
# 2. PyTorch implementation — vectorized log-prob on tensors (Gaussian NB)
# ---------------------------------------------------------------------------
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GaussianNBTorch:
    r"""
    Gaussian Naive Bayes with the same closed-form MLE, but the parameter fit and
    the per-class log-likelihood are computed as **vectorized tensor ops** and the
    posterior is normalized with `torch.logsumexp`. No autograd / gradients —
    Naive Bayes has a closed-form solution; we just lean on tensor broadcasting.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        dev = get_device(); self.device = dev
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        y = torch.as_tensor(np.asarray(y), device=dev)
        self.classes_ = torch.unique(y)
        eps = self.var_smoothing * X.var(0).max()
        means, vars, log_priors = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            means.append(Xc.mean(0))
            vars.append(Xc.var(0, unbiased=False) + eps)
            log_priors.append(torch.log(torch.tensor(len(Xc) / len(X), device=dev)))
        self.theta_ = torch.stack(means)            # (C, d)
        self.sigma_ = torch.stack(vars)             # (C, d)
        self.log_prior_ = torch.stack(log_priors)   # (C,)
        return self

    def _jll(self, X):
        # Broadcast over classes: X (n,1,d) vs params (1,C,d) -> (n, C)
        Xe = X[:, None, :]
        mu = self.theta_[None, :, :]
        var = self.sigma_[None, :, :]
        ll = -0.5 * (torch.log(2 * np.pi * var) + (Xe - mu) ** 2 / var).sum(-1)
        return ll + self.log_prior_[None, :]

    @torch.no_grad()
    def predict_log_proba(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        jll = self._jll(X)
        return (jll - torch.logsumexp(jll, dim=1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def predict(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        idx = self._jll(X).argmax(1)
        return self.classes_[idx].cpu().numpy()


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def _toy_text():
    """Tiny bag-of-words corpus: 2 classes (sports vs tech), 6-word vocabulary."""
    vocab = ["ball", "game", "score", "cpu", "code", "data"]
    docs = [
        ("ball game score score", 0), ("game ball game", 0),
        ("score ball game ball", 0), ("game game score", 0),
        ("cpu code data data", 1), ("code cpu code", 1),
        ("data data code cpu", 1), ("cpu cpu code", 1),
    ]
    idx = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    y = np.zeros(len(docs), int)
    for r, (text, label) in enumerate(docs):
        for w in text.split():
            X[r, idx[w]] += 1
        y[r] = label
    return X, y, vocab


def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)

    # --- Gaussian NB on iris (continuous features) ---
    from sklearn.datasets import load_iris
    iris = load_iris()
    Xi, yi = iris.data, iris.target
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Xi))
    Xi, yi = Xi[perm], yi[perm]
    Xtr, ytr, Xte, yte = Xi[:120], yi[:120], Xi[120:], yi[120:]

    print("=== Gaussian NB on iris (continuous) ===")
    g = GaussianNB().fit(Xtr, ytr)
    print(f"  NumPy GaussianNB  test acc = {np.mean(g.predict(Xte) == yte):.3f}")
    gt = GaussianNBTorch().fit(Xtr, ytr)
    print(f"  Torch GaussianNB  test acc = {np.mean(gt.predict(Xte) == yte):.3f}")
    # the two should agree on log-probabilities
    diff = np.abs(g.predict_log_proba(Xte) - gt.predict_log_proba(Xte)).max()
    print(f"  max |logP_numpy - logP_torch| = {diff:.2e}  (same model)")

    # --- Multinomial & Bernoulli NB on a tiny bag-of-words corpus ---
    X, y, vocab = _toy_text()
    print("\n=== Text NB on tiny bag-of-words (sports=0 vs tech=1) ===")
    mnb = MultinomialNB(alpha=1.0).fit(X, y)
    bnb = BernoulliNB(alpha=1.0).fit(X, y)
    print(f"  MultinomialNB train acc = {np.mean(mnb.predict(X) == y):.3f}")
    print(f"  BernoulliNB   train acc = {np.mean(bnb.predict(X) == y):.3f}")

    tests = ["game ball score", "cpu code data", "game code"]
    idx = {w: i for i, w in enumerate(vocab)}
    Xt = np.zeros((len(tests), len(vocab)))
    for r, t in enumerate(tests):
        for w in t.split():
            Xt[r, idx[w]] += 1
    names = {0: "sports", 1: "tech"}
    for t, p, lp in zip(tests, mnb.predict(Xt), mnb.predict_log_proba(Xt)):
        print(f"    '{t:18s}' -> {names[p]:6s}  P={np.exp(lp.max()):.3f}")

    # --- show Laplace smoothing prevents zero probabilities ---
    print("\n  Laplace smoothing: P(word|class) for class 'tech' (alpha=1):")
    for w, lp in zip(vocab, mnb.feature_log_prob_[1]):
        print(f"    {w:6s}: {np.exp(lp):.3f}")
    print("  -> 'ball'/'game' never appear in tech docs yet get nonzero prob.")


if __name__ == "__main__":
    demo()

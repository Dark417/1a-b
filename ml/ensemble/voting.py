"""
Voting Ensembles (Hard & Soft Voting)
======================================
Combine several *already-good*, *diverse* classifiers by letting them vote.
**Hard voting** takes the majority predicted label; **soft voting** averages the
predicted class probabilities and takes the argmax (optionally weighted). Soft
voting usually wins because it uses confidence, not just the top-1 label — a
confident correct model can outweigh several unsure wrong ones.

Variants implemented here:
    - Hard voting (plurality of predicted labels)
    - Soft voting (mean of predicted probabilities)
    - Weighted soft voting (per-model weights)
    - Heterogeneous base learners (logistic regression, shallow tree, k-NN, GNB)

Training techniques demonstrated:
    - Ensembling diverse learners
    - Hard vs soft aggregation and why confidence helps

References:
    - Dietterich (2000), "Ensemble Methods in Machine Learning"
    - scikit-learn VotingClassifier documentation
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Local self-contained base classifiers (each exposes predict / predict_proba).
# ---------------------------------------------------------------------------
class _LogReg:
    """Softmax (multinomial) logistic regression via gradient descent."""

    def __init__(self, lr=0.3, n_iter=400, l2=1e-3):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(int)
        n, d = X.shape
        self.classes_ = np.unique(y); K = len(self.classes_)
        Y = np.eye(K)[np.searchsorted(self.classes_, y)]
        self.W = np.zeros((d, K)); self.b = np.zeros(K)
        for _ in range(self.n_iter):
            P = self._softmax(X @ self.W + self.b)
            self.W -= self.lr * (X.T @ (P - Y) / n + self.l2 * self.W)
            self.b -= self.lr * (P - Y).mean(0)
        return self

    def predict_proba(self, X):
        return self._softmax(np.asarray(X, float) @ self.W + self.b)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class _Tree:
    """Shallow CART classifier; leaves hold class proportions (probabilities)."""

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth, self.min_samples_split = max_depth, min_samples_split

    def _best_split(self, X, y):
        n, d = X.shape; best = (np.inf, None, None)
        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs = X[order, f]; valid = xs[:-1] != xs[1:]
            if not valid.any():
                continue
            cl_n = np.arange(1, n); cr_n = n - cl_n
            oh = np.eye(self._K)[y[order].astype(int)]
            cum = np.cumsum(oh, axis=0); tot = cum[-1]
            cl, cr = cum[:-1], tot - cum[:-1]
            gl = 1 - ((cl / cl_n[:, None]) ** 2).sum(1)
            gr = 1 - ((cr / cr_n[:, None]) ** 2).sum(1)
            s = np.where(valid, (cl_n * gl + cr_n * gr) / n, np.inf)
            j = int(np.argmin(s))
            if s[j] < best[0]:
                best = (s[j], f, (xs[j] + xs[j + 1]) / 2)
        return best

    def _build(self, X, y, depth):
        if (len(y) < self.min_samples_split or depth >= self.max_depth or
                len(np.unique(y)) == 1):
            return ("leaf", np.bincount(y.astype(int), minlength=self._K) / len(y))
        _, f, t = self._best_split(X, y)
        if f is None:
            return ("leaf", np.bincount(y.astype(int), minlength=self._K) / len(y))
        m = X[:, f] <= t
        return ("node", f, t, self._build(X[m], y[m], depth + 1),
                self._build(X[~m], y[~m], depth + 1))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y); self._K = int(y.max()) + 1
        self.root = self._build(X, y, 0)
        return self

    def _one(self, x, node):
        if node[0] == "leaf":
            return node[1]
        _, f, t, l, r = node
        return self._one(x, l if x[f] <= t else r)

    def predict_proba(self, X):
        return np.array([self._one(x, self.root) for x in np.asarray(X, float)])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class _GaussianNB:
    """Gaussian Naive Bayes — independent per-feature Gaussians per class."""

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        self.theta, self.var, self.prior = [], [], []
        for c in self.classes_:
            Xc = X[y == c]
            self.theta.append(Xc.mean(0))
            self.var.append(Xc.var(0) + 1e-9)        # variance floor
            self.prior.append(len(Xc) / len(X))
        self.theta = np.array(self.theta); self.var = np.array(self.var)
        self.prior = np.array(self.prior)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        # log P(c|x) ∝ log prior - 1/2 sum[log(2πσ²) + (x-μ)²/σ²]
        logp = []
        for k in range(len(self.classes_)):
            ll = -0.5 * (np.log(2 * np.pi * self.var[k])
                         + (X - self.theta[k]) ** 2 / self.var[k]).sum(1)
            logp.append(np.log(self.prior[k]) + ll)
        logp = np.array(logp).T
        logp -= logp.max(1, keepdims=True)
        P = np.exp(logp)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ---------------------------------------------------------------------------
# 1. NumPy implementation: the Voting ensemble
# ---------------------------------------------------------------------------
class VotingNumPy:
    """Hard or soft voting over a list of fitted-on-fit base classifiers.

    estimators : list of (name, fresh_unfitted_model) tuples.
    voting     : "hard" (majority label) or "soft" (mean probabilities).
    weights    : optional per-estimator weights (soft voting).
    """

    def __init__(self, estimators, voting="soft", weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = [(name, m.fit(X, y)) for name, m in self.estimators]
        return self

    def _weights(self):
        if self.weights is None:
            return np.ones(len(self.models_))
        return np.asarray(self.weights, float)

    def predict_proba(self, X):
        w = self._weights()
        P = np.zeros((len(np.asarray(X, float)), len(self.classes_)))
        for (_, m), wi in zip(self.models_, w):
            P += wi * m.predict_proba(X)         # weighted average of probabilities
        return P / w.sum()

    def predict(self, X):
        if self.voting == "soft":
            return self.classes_[self.predict_proba(X).argmax(1)]
        # hard voting: weighted plurality of predicted labels
        w = self._weights()
        n = len(np.asarray(X, float))
        tally = np.zeros((n, len(self.classes_)))
        idx = {c: i for i, c in enumerate(self.classes_)}
        for (_, m), wi in zip(self.models_, w):
            pred = m.predict(X)
            for i in range(n):
                tally[i, idx[pred[i]]] += wi
        return self.classes_[tally.argmax(1)]


# ---------------------------------------------------------------------------
# 2. "PyTorch" note + scikit-learn cross-check
# ---------------------------------------------------------------------------
# Voting is a thin aggregation layer over arbitrary, already-trained classifiers
# (here logistic regression, a tree, and Gaussian NB). The aggregation (argmax of
# averaged probabilities, or plurality of labels) has nothing to differentiate,
# so an idiomatic PyTorch model is not the natural tool. We cross-check against
# scikit-learn's VotingClassifier.
def sklearn_reference(X, y, voting="soft"):
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    estimators = [("lr", LogisticRegression(max_iter=500)),
                  ("dt", DecisionTreeClassifier(max_depth=5, random_state=SEED)),
                  ("gnb", GaussianNB())]
    return VotingClassifier(estimators, voting=voting).fit(X, y)


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=600, n_features=12, n_informative=7,
                               n_redundant=2, n_classes=3, n_clusters_per_class=1,
                               random_state=SEED)
    Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]

    def fresh():
        return [("lr", _LogReg()), ("dt", _Tree(max_depth=5)), ("gnb", _GaussianNB())]

    # individual base accuracies
    for name, m in fresh():
        m.fit(Xtr, ytr)
        print(f"[base] {name:4s} acc={np.mean(m.predict(Xte) == yte):.3f}")

    hard = VotingNumPy(fresh(), voting="hard").fit(Xtr, ytr)
    soft = VotingNumPy(fresh(), voting="soft").fit(Xtr, ytr)
    print(f"[vote] hard acc={np.mean(hard.predict(Xte) == yte):.3f}")
    print(f"[vote] soft acc={np.mean(soft.predict(Xte) == yte):.3f}")

    # weighted soft voting: trust logistic regression more
    wsoft = VotingNumPy(fresh(), voting="soft", weights=[2, 1, 1]).fit(Xtr, ytr)
    print(f"[vote] soft weighted[2,1,1] acc={np.mean(wsoft.predict(Xte) == yte):.3f}")

    skh = sklearn_reference(Xtr, ytr, voting="hard")
    sks = sklearn_reference(Xtr, ytr, voting="soft")
    print(f"[sk  ] hard acc={np.mean(skh.predict(Xte) == yte):.3f}  "
          f"soft acc={np.mean(sks.predict(Xte) == yte):.3f}")


if __name__ == "__main__":
    demo()

"""
Stacking (Stacked Generalization)
==================================
Combine *heterogeneous* base learners by training a **meta-learner** on their
predictions. The crucial trick: feed the meta-learner **out-of-fold (OOF)**
predictions, so the meta-features for each training row come from base models
that did *not* see that row. This prevents the leakage that would occur if base
models predicted their own training data, which would make the meta-learner
trust overfit base models.

Variants implemented here:
    - Classification (meta-features = base class probabilities) and regression
    - K-fold out-of-fold generation of meta-features (Wolpert's scheme)
    - Optional `passthrough` (concatenate original features to meta-features)
    - Base learners refit on the full training set for test-time prediction

Training techniques demonstrated:
    - Out-of-fold cross-validated stacking (leakage prevention)
    - Meta-learning over heterogeneous models

References:
    - Wolpert (1992), "Stacked Generalization"
    - Breiman (1996), "Stacked Regressions"
"""

from __future__ import annotations

import copy

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Local self-contained base learners (no sibling imports).
# ---------------------------------------------------------------------------
class _LogReg:
    """Multinomial logistic regression (softmax) via full-batch gradient descent."""

    def __init__(self, lr=0.1, n_iter=400, l2=1e-3):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y).astype(int)
        n, d = X.shape
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        Y = np.eye(K)[np.searchsorted(self.classes_, y)]   # one-hot
        self.W = np.zeros((d, K)); self.b = np.zeros(K)
        for _ in range(self.n_iter):
            P = self._softmax(X @ self.W + self.b)
            gW = X.T @ (P - Y) / n + self.l2 * self.W
            gb = (P - Y).mean(0)
            self.W -= self.lr * gW; self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        return self._softmax(np.asarray(X, float) @ self.W + self.b)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class _Tree:
    """Shallow CART (depth-limited) base learner — diverse from logistic reg.

    Leaves store class proportions, so it exposes `predict_proba` for stacking.
    Vectorized prefix-sum split search keeps it fast."""

    def __init__(self, task="classification", max_depth=4, min_samples_split=2):
        self.task, self.max_depth, self.min_samples_split = task, max_depth, min_samples_split

    def _best_split(self, X, y):
        n, d = X.shape
        best = (np.inf, None, None)
        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs = X[order, f]
            valid = xs[:-1] != xs[1:]
            if not valid.any():
                continue
            cl_n = np.arange(1, n); cr_n = n - cl_n
            if self.task == "classification":
                oh = np.eye(self._K)[y[order].astype(int)]
                cum = np.cumsum(oh, axis=0); tot = cum[-1]
                cl, cr = cum[:-1], tot - cum[:-1]
                gl = 1 - ((cl / cl_n[:, None]) ** 2).sum(1)
                gr = 1 - ((cr / cr_n[:, None]) ** 2).sum(1)
                s = (cl_n * gl + cr_n * gr) / n
            else:
                ys = y[order].astype(float)
                cs = np.cumsum(ys)[:-1]; cs2 = np.cumsum(ys ** 2)[:-1]
                tot, tot2 = cs[-1] + ys[-1], cs2[-1] + ys[-1] ** 2
                vl = cs2 / cl_n - (cs / cl_n) ** 2
                vr = (tot2 - cs2) / cr_n - ((tot - cs) / cr_n) ** 2
                s = (cl_n * vl + cr_n * vr) / n
            s = np.where(valid, s, np.inf)
            j = int(np.argmin(s))
            if s[j] < best[0]:
                best = (s[j], f, (xs[j] + xs[j + 1]) / 2)
        return best

    def _leaf(self, y):
        if self.task == "classification":
            return np.bincount(y.astype(int), minlength=self._K) / len(y)
        return float(y.mean())

    def _build(self, X, y, depth):
        if (len(y) < self.min_samples_split or depth >= self.max_depth or
                len(np.unique(y)) == 1):
            return ("leaf", self._leaf(y))
        _, f, t = self._best_split(X, y)
        if f is None:
            return ("leaf", self._leaf(y))
        m = X[:, f] <= t
        return ("node", f, t, self._build(X[m], y[m], depth + 1),
                self._build(X[~m], y[~m], depth + 1))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            self._K = int(y.max()) + 1
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
        if self.task == "classification":
            return self.classes_[self.predict_proba(X).argmax(1)]
        return np.array([self._one(x, self.root) for x in np.asarray(X, float)])


# ---------------------------------------------------------------------------
# 1. NumPy implementation: Stacking with out-of-fold meta-features
# ---------------------------------------------------------------------------
def _kfold_indices(n, k, rng):
    idx = rng.permutation(n)
    return np.array_split(idx, k)


class StackingNumPy:
    """Stacked generalization.

    base_factories : list of 0-arg callables -> fresh base learners.
    meta_factory   : 0-arg callable -> the meta learner.
    """

    def __init__(self, base_factories, meta_factory, task="classification",
                 n_folds=5, passthrough=False, seed=SEED):
        self.base_factories = base_factories
        self.meta_factory = meta_factory
        self.task = task
        self.n_folds = n_folds
        self.passthrough = passthrough
        self.seed = seed

    def _base_out(self, model, X):
        """Per-base output used as meta-features: probabilities (clf, drop last
        column to avoid collinearity) or scalar prediction (reg)."""
        if self.task == "classification":
            P = model.predict_proba(X)
            return P[:, :-1]                # K-1 columns suffice
        return model.predict(X).reshape(-1, 1)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        n = len(X)
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y) if self.task == "classification" else None

        folds = _kfold_indices(n, self.n_folds, rng)
        # ---- generate OUT-OF-FOLD meta-features (no leakage) ----
        meta_cols = []
        for factory in self.base_factories:
            col = np.zeros((n, (len(self.classes_) - 1) if self.task == "classification" else 1))
            for val in folds:
                train = np.setdiff1d(np.arange(n), val)
                m = copy.deepcopy(factory())
                m.fit(X[train], y[train])           # fit on K-1 folds
                col[val] = self._base_out(m, X[val])  # predict the held-out fold
            meta_cols.append(col)
        Z = np.hstack(meta_cols)
        if self.passthrough:
            Z = np.hstack([Z, X])

        # ---- meta-learner trained on OOF features ----
        self.meta_ = copy.deepcopy(self.meta_factory())
        self.meta_.fit(Z, y)

        # ---- refit each base model on the FULL training set for test time ----
        self.bases_ = [copy.deepcopy(f()).fit(X, y) for f in self.base_factories]
        return self

    def _meta_features(self, X):
        cols = [self._base_out(m, X) for m in self.bases_]
        Z = np.hstack(cols)
        if self.passthrough:
            Z = np.hstack([Z, np.asarray(X, float)])
        return Z

    def predict(self, X):
        return self.meta_.predict(self._meta_features(X))

    def predict_proba(self, X):
        assert self.task == "classification"
        return self.meta_.predict_proba(self._meta_features(X))


# ---------------------------------------------------------------------------
# 2. "PyTorch" note + scikit-learn cross-check
# ---------------------------------------------------------------------------
# Stacking is a meta-procedure: it orchestrates arbitrary base learners (here
# discrete trees + logistic regression) and a meta learner via cross-validated
# OOF predictions. The orchestration itself has nothing to differentiate, so an
# idiomatic PyTorch model is not the natural tool. We cross-check against
# scikit-learn's StackingClassifier / StackingRegressor.
def sklearn_reference(X, y, task="classification", **kw):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    if task == "classification":
        from sklearn.ensemble import StackingClassifier
        estimators = [("stump", DecisionTreeClassifier(max_depth=1)),
                      ("tree", DecisionTreeClassifier(max_depth=4))]
        return StackingClassifier(estimators,
                                  final_estimator=LogisticRegression(max_iter=500),
                                  cv=5, **kw).fit(X, y)
    from sklearn.ensemble import StackingRegressor
    estimators = [("stump", DecisionTreeRegressor(max_depth=1)),
                  ("tree", DecisionTreeRegressor(max_depth=4))]
    return StackingRegressor(estimators, final_estimator=Ridge(), cv=5, **kw).fit(X, y)


# ---------------------------------------------------------------------------
# Local Ridge meta-learner for regression stacking (self-contained).
# ---------------------------------------------------------------------------
class _Ridge:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xb = np.hstack([np.ones((len(X), 1)), X])
        A = Xb.T @ Xb + self.alpha * np.eye(Xb.shape[1])
        A[0, 0] -= self.alpha                        # don't penalize the bias
        self.w = np.linalg.solve(A, Xb.T @ y)
        return self

    def predict(self, X):
        Xb = np.hstack([np.ones((len(np.asarray(X, float)), 1)), np.asarray(X, float)])
        return Xb @ self.w


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    from sklearn.datasets import make_classification, make_friedman1

    # ---------- classification ----------
    X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                               n_redundant=2, n_classes=3, n_clusters_per_class=1,
                               random_state=SEED)
    Xtr, ytr, Xte, yte = X[:380], y[:380], X[380:], y[380:]

    bases = [lambda: _Tree(task="classification", max_depth=4),
             lambda: _LogReg(lr=0.3, n_iter=400)]
    for f, name in [(bases[0], "tree"), (bases[1], "logreg")]:
        m = f().fit(Xtr, ytr)
        print(f"[clf] base {name:7s} acc={np.mean(m.predict(Xte) == yte):.3f}")

    stack = StackingNumPy(bases, lambda: _LogReg(lr=0.3, n_iter=400),
                          task="classification", n_folds=5).fit(Xtr, ytr)
    print(f"[clf] STACK (OOF) acc={np.mean(stack.predict(Xte) == yte):.3f}")
    stack_pt = StackingNumPy(bases, lambda: _LogReg(lr=0.3, n_iter=400),
                             task="classification", passthrough=True).fit(Xtr, ytr)
    print(f"[clf] STACK +passthrough acc={np.mean(stack_pt.predict(Xte) == yte):.3f}")
    sk = sklearn_reference(Xtr, ytr)
    print(f"[clf] sklearn Stacking acc={np.mean(sk.predict(Xte) == yte):.3f}")

    # ---------- regression ----------
    Xr, yr = make_friedman1(n_samples=400, noise=1.0, random_state=SEED)
    Xrtr, yrtr, Xrte, yrte = Xr[:300], yr[:300], Xr[300:], yr[300:]
    rbases = [lambda: _Tree(task="regression", max_depth=4),
              lambda: _Ridge(alpha=1.0)]
    stackr = StackingNumPy(rbases, lambda: _Ridge(alpha=1.0),
                           task="regression", n_folds=5).fit(Xrtr, yrtr)
    print(f"[reg] STACK MSE={np.mean((stackr.predict(Xrte) - yrte) ** 2):.3f}")
    skr = sklearn_reference(Xrtr, yrtr, task="regression")
    print(f"[reg] sklearn Stacking MSE={np.mean((skr.predict(Xrte) - yrte) ** 2):.3f}")


if __name__ == "__main__":
    demo()

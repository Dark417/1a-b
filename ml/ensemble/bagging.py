"""
Bagging (Bootstrap Aggregation)
===============================
Train many copies of the *same* base learner on different **bootstrap
resamples** of the data and aggregate their predictions (vote for
classification, average for regression). Bagging leaves bias roughly unchanged
but slashes **variance**, so it helps most for high-variance, low-bias learners
such as fully grown decision trees.

Variants implemented here:
    - Classification (majority vote) and regression (mean)
    - Generic over any base learner exposing fit/predict (clone via factory)
    - Out-of-bag (OOB) error estimate
    - An explicit variance-reduction experiment

Training techniques demonstrated:
    - Bootstrap resampling
    - Averaging to reduce variance (the 1/M and correlation rho story)
    - OOB estimation as free validation

References:
    - Breiman (1996), "Bagging Predictors"
    - Hastie, Tibshirani, Friedman, "ESL" ch. 8.7
"""

from __future__ import annotations

import copy

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Local self-contained base learners (no sibling imports).
# ---------------------------------------------------------------------------
class _Stump:
    """Depth-1 CART (decision stump) — a deliberately weak, biased learner."""

    def __init__(self, task="classification"):
        self.task = task

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        best = (np.inf, None, None, None, None)
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            for t in (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else []:
                m = X[:, f] <= t
                if m.sum() == 0 or (~m).sum() == 0:
                    continue
                if self.task == "classification":
                    score = self._gini(y[m]) * m.sum() + self._gini(y[~m]) * (~m).sum()
                    lv = self._mode(y[m]); rv = self._mode(y[~m])
                else:
                    score = np.var(y[m]) * m.sum() + np.var(y[~m]) * (~m).sum()
                    lv = y[m].mean(); rv = y[~m].mean()
                if score < best[0]:
                    best = (score, f, t, lv, rv)
        _, self.f, self.t, self.lv, self.rv = best
        if self.f is None:
            self.f, self.t = 0, 0.0
            self.lv = self.rv = (self._mode(y) if self.task == "classification" else y.mean())
        return self

    @staticmethod
    def _gini(y):
        _, c = np.unique(y, return_counts=True); p = c / c.sum()
        return 1 - (p ** 2).sum()

    @staticmethod
    def _mode(y):
        v, c = np.unique(y, return_counts=True); return v[c.argmax()]

    def predict(self, X):
        X = np.asarray(X, float)
        return np.where(X[:, self.f] <= self.t, self.lv, self.rv)


class _Tree:
    """Unconstrained CART (deep) — a low-bias, HIGH-variance learner. Bagging
    shines here. Recursive, self-contained."""

    def __init__(self, task="classification", max_depth=None, min_samples_split=2):
        self.task, self.max_depth, self.min_samples_split = task, max_depth, min_samples_split

    def _imp(self, y):
        if self.task == "classification":
            _, c = np.unique(y, return_counts=True); p = c / c.sum()
            return 1 - (p ** 2).sum()
        return np.var(y) if len(y) else 0.0

    def _leaf(self, y):
        if self.task == "classification":
            v, c = np.unique(y, return_counts=True); return v[c.argmax()]
        return float(y.mean())

    def _build(self, X, y, depth):
        if (len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(np.unique(y)) == 1):
            return ("leaf", self._leaf(y))
        best = (np.inf, None, None)
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            for t in (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else []:
                m = X[:, f] <= t
                if m.sum() == 0 or (~m).sum() == 0:
                    continue
                s = (self._imp(y[m]) * m.sum() + self._imp(y[~m]) * (~m).sum()) / len(y)
                if s < best[0]:
                    best = (s, f, t)
        _, f, t = best
        if f is None:
            return ("leaf", self._leaf(y))
        m = X[:, f] <= t
        return ("node", f, t, self._build(X[m], y[m], depth + 1),
                self._build(X[~m], y[~m], depth + 1))

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y), 0)
        return self

    def _one(self, x, node):
        if node[0] == "leaf":
            return node[1]
        _, f, t, l, r = node
        return self._one(x, l if x[f] <= t else r)

    def predict(self, X):
        return np.array([self._one(x, self.root) for x in np.asarray(X, float)])


# ---------------------------------------------------------------------------
# 1. NumPy implementation: generic Bagging over any (fit/predict) base learner
# ---------------------------------------------------------------------------
class BaggingNumPy:
    """Bootstrap-aggregate an arbitrary base learner.

    `base_factory` is a 0-arg callable returning a *fresh* unfitted learner with
    `.fit(X, y)` / `.predict(X)`. We deep-copy it per bag for safety.
    """

    def __init__(self, base_factory, n_estimators=50, task="classification",
                 max_samples=1.0, seed=SEED):
        self.base_factory = base_factory
        self.n_estimators = n_estimators
        self.task = task
        self.max_samples = max_samples
        self.seed = seed
        self.models_ = []
        self.oob_indices_ = []
        self.oob_score_ = None

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        n = len(X)
        m = max(1, int(self.max_samples * n))
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y) if self.task == "classification" else None
        self.models_, self.oob_indices_ = [], []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, m)                       # bootstrap sample
            oob = np.setdiff1d(np.arange(n), np.unique(idx))  # left-out rows
            model = copy.deepcopy(self.base_factory())
            model.fit(X[idx], y[idx])
            self.models_.append(model)
            self.oob_indices_.append(oob)
        self._oob(X, y)
        return self

    def _aggregate(self, preds):
        """preds: (M, n). Vote (clf) or mean (reg) over the M models."""
        if self.task == "classification":
            out = np.empty(preds.shape[1], dtype=self.classes_.dtype)
            for i in range(preds.shape[1]):
                v, c = np.unique(preds[:, i], return_counts=True)
                out[i] = v[c.argmax()]
            return out
        return preds.mean(0)

    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models_])
        return self._aggregate(preds)

    def _oob(self, X, y):
        n = len(X)
        # collect, for each sample, predictions only from models that didn't see it
        per_sample = [[] for _ in range(n)]
        for model, oob in zip(self.models_, self.oob_indices_):
            if len(oob) == 0:
                continue
            p = model.predict(X[oob])
            for i, idx in enumerate(oob):
                per_sample[idx].append(p[i])
        have = [i for i in range(n) if per_sample[i]]
        if not have:
            self.oob_score_ = None
            return
        if self.task == "classification":
            agg = []
            for i in have:
                v, c = np.unique(per_sample[i], return_counts=True)
                agg.append(v[c.argmax()])
            self.oob_score_ = float(np.mean(np.array(agg) == y[have]))   # OOB acc
        else:
            agg = np.array([np.mean(per_sample[i]) for i in have])
            self.oob_score_ = float(np.mean((agg - y[have]) ** 2))       # OOB MSE


# ---------------------------------------------------------------------------
# 2. "PyTorch" note + scikit-learn cross-check
# ---------------------------------------------------------------------------
# Bagging is a meta-procedure around any base learner (here: discrete trees /
# stumps). There is nothing to differentiate at the ensemble level, so an
# idiomatic PyTorch model is not the natural tool. We cross-check against
# scikit-learn's BaggingClassifier / BaggingRegressor.
def sklearn_reference(X, y, task="classification", **kw):
    from sklearn.ensemble import BaggingClassifier, BaggingRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    if task == "classification":
        return BaggingClassifier(DecisionTreeClassifier(), oob_score=True,
                                 random_state=SEED, **kw).fit(X, y)
    return BaggingRegressor(DecisionTreeRegressor(), oob_score=True,
                            random_state=SEED, **kw).fit(X, y)


# ---------------------------------------------------------------------------
# Variance-reduction experiment: estimate Var of a single tree vs a bagged
# ensemble across independent training sets drawn from the same generator.
# ---------------------------------------------------------------------------
def variance_experiment(n_trials=20, n_train=200, seed=SEED):
    from sklearn.datasets import make_friedman1
    rng = np.random.default_rng(seed)
    Xte, _ = make_friedman1(n_samples=100, noise=0.0, random_state=999)
    single_preds, bag_preds = [], []
    for t in range(n_trials):
        Xtr, ytr = make_friedman1(n_samples=n_train, noise=1.0,
                                  random_state=int(rng.integers(1 << 30)))
        single = _Tree(task="regression", max_depth=None).fit(Xtr, ytr)
        single_preds.append(single.predict(Xte))
        bag = BaggingNumPy(lambda: _Tree(task="regression"), n_estimators=25,
                           task="regression", seed=t).fit(Xtr, ytr)
        bag_preds.append(bag.predict(Xte))
    # variance of the prediction at each test point, averaged over points
    var_single = np.var(np.array(single_preds), axis=0).mean()
    var_bag = np.var(np.array(bag_preds), axis=0).mean()
    return var_single, var_bag


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    from sklearn.datasets import make_classification, make_friedman1

    # ---------- classification: bag deep trees ----------
    X, y = make_classification(n_samples=500, n_features=10, n_informative=6,
                               n_redundant=2, random_state=SEED)
    Xtr, ytr, Xte, yte = X[:380], y[:380], X[380:], y[380:]
    single = _Tree(task="classification").fit(Xtr, ytr)
    print(f"[clf] single deep tree acc={np.mean(single.predict(Xte) == yte):.3f}")
    bag = BaggingNumPy(lambda: _Tree(task="classification"),
                       n_estimators=50, task="classification").fit(Xtr, ytr)
    print(f"[clf] bagged trees acc={np.mean(bag.predict(Xte) == yte):.3f}  "
          f"OOB acc={bag.oob_score_:.3f}")
    sk = sklearn_reference(Xtr, ytr, n_estimators=50)
    print(f"[clf] sklearn Bagging acc={np.mean(sk.predict(Xte) == yte):.3f}  "
          f"OOB acc={sk.oob_score_:.3f}")

    # ---------- regression ----------
    Xr, yr = make_friedman1(n_samples=400, noise=1.0, random_state=SEED)
    Xrtr, yrtr, Xrte, yrte = Xr[:300], yr[:300], Xr[300:], yr[300:]
    bagr = BaggingNumPy(lambda: _Tree(task="regression"), n_estimators=50,
                        task="regression").fit(Xrtr, yrtr)
    single_r = _Tree(task="regression").fit(Xrtr, yrtr)
    print(f"[reg] single tree MSE={np.mean((single_r.predict(Xrte) - yrte) ** 2):.3f}")
    print(f"[reg] bagged MSE={np.mean((bagr.predict(Xrte) - yrte) ** 2):.3f}  "
          f"OOB MSE={bagr.oob_score_:.3f}")

    # ---------- variance reduction made explicit ----------
    vs, vb = variance_experiment(n_trials=12)
    print(f"[var] mean prediction variance: single tree={vs:.3f}  bagged={vb:.3f}  "
          f"(ratio {vb / vs:.2f}x)")


if __name__ == "__main__":
    demo()

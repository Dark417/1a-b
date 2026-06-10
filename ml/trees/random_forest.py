"""
Random Forest
=============
An ensemble of decision trees, each grown on a bootstrap resample of the data
*and* restricted to a random subset of features at every split. Averaging many
such de-correlated, high-variance trees dramatically reduces variance while
keeping bias low — a textbook bias-variance win. The out-of-bag (OOB) samples
give a free, cross-validation-like error estimate.

Variants implemented here:
    - Classification (majority vote across trees)
    - Regression (mean prediction across trees)
    - Feature subsampling per split (`max_features`: "sqrt", "log2", int, float)
    - Out-of-bag (OOB) error estimate
    - Feature importances (impurity-decrease, Breiman-style)

Training techniques demonstrated:
    - Bagging (bootstrap aggregation) for variance reduction
    - Random feature subspaces to de-correlate trees
    - OOB estimation as built-in validation

References:
    - Breiman (2001), "Random Forests"
    - Breiman (1996), "Bagging Predictors"
    - Hastie, Tibshirani, Friedman, "ESL" ch. 15
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Local shallow CART tree (self-contained: we do NOT import decision_tree.py).
# Supports per-split random feature subsampling, which is the heart of an RF.
# ---------------------------------------------------------------------------
def _gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return 1.0 - (p ** 2).sum()


def _mse(y):
    return float(np.var(y)) if len(y) else 0.0


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = self.threshold = self.left = self.right = self.value = None


class _Tree:
    """Minimal CART with random feature subsets at each split."""

    def __init__(self, task="classification", max_depth=None, min_samples_split=2,
                 max_features=None, n_classes=None, rng=None):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features        # how many features each split sees
        self.n_classes = n_classes
        self.rng = rng or np.random.default_rng(SEED)
        self._imp = _gini if task == "classification" else _mse
        # accumulate weighted impurity decrease per feature (importances)
        self.importances_ = None

    def _n_feat(self, d):
        m = self.max_features
        if m is None:
            return d
        if m == "sqrt":
            return max(1, int(np.sqrt(d)))
        if m == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(m, float):
            return max(1, int(m * d))
        return min(int(m), d)

    def _best_split(self, X, y):
        # Vectorized split search via prefix sums (O(n log n) per feature).
        # The random feature subspace is what de-correlates RF trees.
        n, d = X.shape
        feats = self.rng.choice(d, self._n_feat(d), replace=False)
        best = (np.inf, None, None)     # minimize weighted child impurity
        for f in feats:
            order = np.argsort(X[:, f], kind="mergesort")
            xs = X[order, f]
            valid = xs[:-1] != xs[1:]   # only split where the value changes
            if not valid.any():
                continue
            cnt_l = np.arange(1, n)
            cnt_r = n - cnt_l
            if self.task == "classification":
                ys = y[order].astype(int)
                # one-hot prefix counts per class -> Gini of each child
                oh = np.eye(self.n_classes)[ys]
                cum = np.cumsum(oh, axis=0)             # (n, C)
                tot = cum[-1]
                cl = cum[:-1]                           # left counts
                cr = tot - cl                           # right counts
                gini_l = 1.0 - ((cl / cnt_l[:, None]) ** 2).sum(1)
                gini_r = 1.0 - ((cr / cnt_r[:, None]) ** 2).sum(1)
                score = (cnt_l * gini_l + cnt_r * gini_r) / n
            else:
                ys = y[order].astype(float)
                cs = np.cumsum(ys)[:-1]
                cs2 = np.cumsum(ys ** 2)[:-1]
                tot, tot2 = cs[-1] + ys[-1], cs2[-1] + ys[-1] ** 2
                # child MSE = E[y^2] - E[y]^2  (variance), weighted by count
                var_l = cs2 / cnt_l - (cs / cnt_l) ** 2
                var_r = (tot2 - cs2) / cnt_r - ((tot - cs) / cnt_r) ** 2
                score = (cnt_l * var_l + cnt_r * var_r) / n
            score = np.where(valid, score, np.inf)
            j = int(np.argmin(score))
            if score[j] < best[0]:
                best = (score[j], f, (xs[j] + xs[j + 1]) / 2)
        return best

    def _leaf_value(self, y):
        if self.task == "classification":
            # store full class-count vector → enables soft (probability) voting
            counts = np.bincount(y.astype(int), minlength=self.n_classes)
            return counts / counts.sum()
        return float(np.mean(y))

    def _build(self, X, y, depth):
        node = _Node()
        stop = (len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                (self.task == "classification" and len(np.unique(y)) == 1))
        if stop:
            node.value = self._leaf_value(y)
            return node
        score, f, t = self._best_split(X, y)
        if f is None or score == np.inf:
            node.value = self._leaf_value(y)
            return node
        # impurity decrease for this node, weighted by samples reaching it
        self.importances_[f] += len(y) * (self._imp(y) - score)
        mask = X[:, f] <= t
        node.feature, node.threshold = f, t
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.importances_ = np.zeros(X.shape[1])
        self.root = self._build(X, np.asarray(y), 0)
        self.importances_ /= max(self.importances_.sum(), 1e-12)
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def predict_raw(self, X):
        """Per-sample leaf value (prob-vector for clf, scalar for reg)."""
        return [self._predict_one(x, self.root) for x in np.asarray(X, float)]


# ---------------------------------------------------------------------------
# 1. NumPy implementation: the Random Forest
# ---------------------------------------------------------------------------
class RandomForestNumPy:
    """Bagged trees + random feature subspaces + OOB error.

    For classification a tree's leaf stores class *probabilities*; the forest
    can vote hard (argmax of summed votes) or soft (average of probabilities).
    """

    def __init__(self, n_estimators=100, task="classification", max_depth=None,
                 min_samples_split=2, max_features="sqrt", bootstrap=True,
                 voting="soft", seed=SEED):
        self.n_estimators = n_estimators
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.voting = voting
        self.seed = seed
        self.trees_ = []
        self.oob_indices_ = []       # per tree: indices NOT in its bootstrap
        self.oob_score_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        n = len(X)
        rng = np.random.default_rng(self.seed)
        self.n_classes = int(y.max()) + 1 if self.task == "classification" else None
        self.trees_, self.oob_indices_ = [], []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                # bootstrap = sample n indices WITH replacement
                idx = rng.integers(0, n, n)
                oob = np.setdiff1d(np.arange(n), np.unique(idx))
            else:
                idx = np.arange(n)
                oob = np.array([], dtype=int)
            tree = _Tree(task=self.task, max_depth=self.max_depth,
                         min_samples_split=self.min_samples_split,
                         max_features=self.max_features,
                         n_classes=self.n_classes,
                         rng=np.random.default_rng(rng.integers(1 << 30)))
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            self.oob_indices_.append(oob)

        # average impurity-decrease importances over the forest
        self.feature_importances_ = np.mean(
            [t.importances_ for t in self.trees_], axis=0)
        if self.bootstrap:
            self._compute_oob(X, y)
        return self

    # ---- prediction ----
    def predict_proba(self, X):
        assert self.task == "classification"
        # average leaf probability vectors across all trees (soft voting)
        P = np.zeros((len(np.asarray(X, float)), self.n_classes))
        for t in self.trees_:
            P += np.array(t.predict_raw(X))
        return P / len(self.trees_)

    def predict(self, X):
        if self.task == "classification":
            if self.voting == "soft":
                return self.predict_proba(X).argmax(1)
            # hard voting: each tree casts argmax vote, take the mode
            votes = np.array([np.array(t.predict_raw(X)).argmax(1)
                              for t in self.trees_])  # (T, n)
            return np.array([np.bincount(votes[:, i], minlength=self.n_classes).argmax()
                             for i in range(votes.shape[1])])
        # regression: mean of tree predictions
        preds = np.array([t.predict_raw(X) for t in self.trees_])
        return preds.mean(0)

    # ---- out-of-bag estimate ----
    def _compute_oob(self, X, y):
        """For each sample, aggregate ONLY trees that did not train on it."""
        n = len(X)
        if self.task == "classification":
            agg = np.zeros((n, self.n_classes))
        else:
            agg = np.zeros(n)
        count = np.zeros(n)
        for t, oob in zip(self.trees_, self.oob_indices_):
            if len(oob) == 0:
                continue
            raw = np.array(t.predict_raw(X[oob]))
            agg[oob] += raw
            count[oob] += 1
        mask = count > 0       # samples that were OOB for at least one tree
        if self.task == "classification":
            pred = agg[mask].argmax(1)
            self.oob_score_ = float(np.mean(pred == y[mask]))   # OOB accuracy
        else:
            pred = agg[mask] / count[mask]
            self.oob_score_ = float(np.mean((pred - y[mask]) ** 2))  # OOB MSE


# ---------------------------------------------------------------------------
# 2. "PyTorch" note + scikit-learn cross-check
# ---------------------------------------------------------------------------
# A random forest is a bag of discrete, greedily-grown trees — there is no loss
# to differentiate and no parameters to optimize by gradient descent, so an
# idiomatic PyTorch model is not the natural tool here (PyTorch shines for
# differentiable models). We instead cross-check the from-scratch forest against
# scikit-learn's optimized C implementation.
def sklearn_reference(X, y, task="classification", **kw):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    Model = RandomForestClassifier if task == "classification" else RandomForestRegressor
    model = Model(random_state=SEED, oob_score=True, bootstrap=True, **kw)
    return model.fit(X, y)


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    from sklearn.datasets import make_classification, make_friedman1

    # ---------- classification ----------
    X, y = make_classification(n_samples=500, n_features=12, n_informative=6,
                               n_redundant=2, n_classes=3, n_clusters_per_class=1,
                               random_state=SEED)
    Xtr, ytr, Xte, yte = X[:380], y[:380], X[380:], y[380:]

    # single tree vs forest → variance reduction in action
    single = _Tree(task="classification", max_depth=None, n_classes=3,
                   rng=np.random.default_rng(SEED)).fit(Xtr, ytr)
    s_pred = np.array(single.predict_raw(Xte)).argmax(1)
    print(f"[clf] single tree   test acc={np.mean(s_pred == yte):.3f}")

    rf = RandomForestNumPy(n_estimators=80, task="classification",
                           max_features="sqrt").fit(Xtr, ytr)
    print(f"[clf] forest (soft) test acc={np.mean(rf.predict(Xte) == yte):.3f}  "
          f"OOB acc={rf.oob_score_:.3f}")
    rf_hard = RandomForestNumPy(n_estimators=80, voting="hard").fit(Xtr, ytr)
    print(f"[clf] forest (hard) test acc={np.mean(rf_hard.predict(Xte) == yte):.3f}")

    sk = sklearn_reference(Xtr, ytr, n_estimators=80, max_features="sqrt")
    print(f"[clf] sklearn RF    test acc={np.mean(sk.predict(Xte) == yte):.3f}  "
          f"OOB acc={sk.oob_score_:.3f}")
    top = np.argsort(rf.feature_importances_)[::-1][:3]
    print(f"[clf] top-3 important features (ours): {top.tolist()}")

    # ---------- regression ----------
    Xr, yr = make_friedman1(n_samples=400, noise=1.0, random_state=SEED)
    Xrtr, yrtr, Xrte, yrte = Xr[:320], yr[:320], Xr[320:], yr[320:]
    rfr = RandomForestNumPy(n_estimators=80, task="regression",
                            max_features=0.5, max_depth=8).fit(Xrtr, yrtr)
    mse = np.mean((rfr.predict(Xrte) - yrte) ** 2)
    print(f"[reg] forest test MSE={mse:.3f}  OOB MSE={rfr.oob_score_:.3f}")
    skr = sklearn_reference(Xrtr, yrtr, task="regression", n_estimators=80, max_depth=8)
    print(f"[reg] sklearn RF MSE={np.mean((skr.predict(Xrte) - yrte) ** 2):.3f}")


if __name__ == "__main__":
    demo()

"""
Decision Tree (CART)
====================
Recursively split the feature space on the question that best separates the
data, building a tree of if-then rules. Interpretable, nonlinear, and the base
learner for random forests and gradient boosting.

Variants implemented here:
    - Classification (Gini / entropy impurity)
    - Regression (variance / MSE reduction)
    - Pre-pruning (max_depth, min_samples_split) — fights overfitting

Training techniques demonstrated:
    - Greedy impurity-reduction splitting
    - The depth ↔ bias/variance trade-off (pruning)

References:
    - Breiman et al. (1984), "Classification and Regression Trees"
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _gini(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum()
    return 1.0 - (p ** 2).sum()


def _entropy(y):
    _, c = np.unique(y, return_counts=True); p = c / c.sum()
    return -(p * np.log2(p + 1e-12)).sum()


def _mse(y):
    return float(np.var(y)) if len(y) else 0.0


# ---------------------------------------------------------------------------
# 1. NumPy implementation
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = self.threshold = self.left = self.right = self.value = None


class DecisionTreeNumPy:
    def __init__(self, task="classification", criterion=None,
                 max_depth=None, min_samples_split=2):
        self.task = task
        self.criterion = criterion or ("gini" if task == "classification" else "mse")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._imp = {"gini": _gini, "entropy": _entropy, "mse": _mse}[self.criterion]

    # weighted impurity after a split (the thing we minimize)
    def _split_score(self, y, mask):
        l, r = y[mask], y[~mask]
        if len(l) == 0 or len(r) == 0:
            return np.inf
        n = len(y)
        return (len(l) * self._imp(l) + len(r) * self._imp(r)) / n

    def _best_split(self, X, y):
        best = (np.inf, None, None)
        for f in range(X.shape[1]):
            thresholds = np.unique(X[:, f])
            # midpoints between consecutive unique values
            for t in (thresholds[:-1] + thresholds[1:]) / 2 if len(thresholds) > 1 else []:
                mask = X[:, f] <= t
                score = self._split_score(y, mask)
                if score < best[0]:
                    best = (score, f, t)
        return best  # (score, feature, threshold)

    def _leaf_value(self, y):
        if self.task == "classification":
            vals, c = np.unique(y, return_counts=True)
            return vals[c.argmax()]
        return float(np.mean(y))

    def _build(self, X, y, depth):
        node = _Node()
        # stopping rules (pre-pruning)
        if (len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(np.unique(y)) == 1):
            node.value = self._leaf_value(y); return node
        score, f, t = self._best_split(X, y)
        if f is None or score == np.inf:
            node.value = self._leaf_value(y); return node
        mask = X[:, f] <= t
        node.feature, node.threshold = f, t
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in np.asarray(X, float)])

    def depth(self, node=None):
        node = node or self.root
        if node.value is not None:
            return 0
        return 1 + max(self.depth(node.left), self.depth(node.right))


# ---------------------------------------------------------------------------
# 2. "PyTorch" note + sklearn cross-check
# ---------------------------------------------------------------------------
# Decision trees are discrete, non-differentiable greedy structures — there is
# no gradient to backprop, so an idiomatic "PyTorch" tree is not the natural
# tool (PyTorch shines for differentiable models). For completeness we
# cross-check our from-scratch tree against scikit-learn's optimized CART.
def sklearn_reference(X, y, task="classification", **kw):
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    Tree = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    return Tree(random_state=SEED, **kw).fit(X, y)


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    from sklearn.datasets import load_iris, make_friedman1

    # classification
    X, y = load_iris(return_X_y=True)
    idx = np.random.permutation(len(X)); X, y = X[idx], y[idx]
    Xtr, ytr, Xte, yte = X[:120], y[:120], X[120:], y[120:]
    for depth in (1, 3, None):
        t = DecisionTreeNumPy(max_depth=depth).fit(Xtr, ytr)
        print(f"[clf] max_depth={str(depth):>4}  depth={t.depth()}  "
              f"test acc={np.mean(t.predict(Xte) == yte):.3f}")

    sk = sklearn_reference(Xtr, ytr, max_depth=3)
    print(f"[clf] sklearn (depth 3) acc={np.mean(sk.predict(Xte) == yte):.3f}")

    # regression
    Xr, yr = make_friedman1(n_samples=300, noise=1.0, random_state=SEED)
    tr = DecisionTreeNumPy(task="regression", max_depth=4).fit(Xr[:240], yr[:240])
    mse = np.mean((tr.predict(Xr[240:]) - yr[240:]) ** 2)
    print(f"[reg] test MSE (depth 4): {mse:.3f}")


if __name__ == "__main__":
    demo()

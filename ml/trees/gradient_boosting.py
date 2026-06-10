"""
Gradient Boosting Machine (GBM)
===============================
Build an additive model stagewise: at each round fit a *new* shallow regression
tree to the **negative gradient** of the loss (the "pseudo-residuals") evaluated
at the current predictions, then add a shrunken version of that tree to the
ensemble. This is functional gradient descent — gradient descent in the space of
functions, where each step is a tree.

Variants implemented here:
    - Regression (squared-error loss → residual fitting)
    - Binary classification (logistic / log-loss, deviance)
    - First-order GBM (Friedman) and a second-order, regularized
      Newton / XGBoost-style variant (gradients + Hessians + leaf shrinkage)
    - Shrinkage (learning rate) and stochastic row subsampling

Training techniques demonstrated:
    - Functional gradient descent (boosting)
    - Shrinkage / learning rate as regularization
    - Stochastic gradient boosting (row subsampling)
    - Newton boosting with a regularized leaf-weight solution (XGBoost)

References:
    - Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine"
    - Friedman (2002), "Stochastic Gradient Boosting"
    - Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Local regression-tree base learner (self-contained). Two split criteria:
#   - "variance"  : ordinary squared-error reduction (fits pseudo-residuals)
#   - "newton"    : XGBoost-style structure score using gradients g & Hessians h
# ---------------------------------------------------------------------------
class _RegNode:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self):
        self.feature = self.threshold = self.left = self.right = self.value = None


class _RegTree:
    """Shallow CART regressor used as the weak learner inside boosting."""

    def __init__(self, max_depth=3, min_samples_split=2, lam=0.0, gamma=0.0,
                 mode="variance"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lam = lam            # L2 leaf regularization (Newton mode)
        self.gamma = gamma        # min split gain (Newton mode)
        self.mode = mode

    # ----- variance (squared-error) splitting -----
    # Vectorized via prefix sums: for a feature sorted ascending, every split is
    # a prefix/suffix. SSE = sum(y^2) - (sum y)^2 / count, so total child SSE for
    # all thresholds at once costs O(n) after the O(n log n) sort.
    def _best_split_var(self, X, y):
        n, d = X.shape
        best = (np.inf, None, None)   # minimize total child SSE
        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys = X[order, f], y[order]
            cs = np.cumsum(ys)          # prefix sum of y
            cs2 = np.cumsum(ys ** 2)    # prefix sum of y^2
            tot, tot2 = cs[-1], cs2[-1]
            cnt = np.arange(1, n)       # left sizes 1..n-1
            sl, sl2 = cs[:-1], cs2[:-1]
            sr, sr2 = tot - sl, tot2 - sl2
            sse_l = sl2 - sl ** 2 / cnt
            sse_r = sr2 - sr ** 2 / (n - cnt)
            total = sse_l + sse_r
            # only valid where the feature value actually changes (real split)
            valid = xs[:-1] != xs[1:]
            total = np.where(valid, total, np.inf)
            j = int(np.argmin(total))
            if total[j] < best[0]:
                best = (total[j], f, (xs[j] + xs[j + 1]) / 2)
        return best

    # ----- Newton (gradient/Hessian) splitting -----
    @staticmethod
    def _leaf_weight(g, h, lam):
        # closed-form minimizer of  G*w + 1/2 (H+lam) w^2  ->  w = -G/(H+lam)
        return -g.sum() / (h.sum() + lam)

    @staticmethod
    def _struct_gain(g, h, lam):
        # XGBoost structure score for a node:  1/2 * G^2 / (H + lam)
        return 0.5 * g.sum() ** 2 / (h.sum() + lam)

    def _best_split_newton(self, X, g, h):
        # Vectorized XGBoost gain over every threshold via prefix sums of g, h.
        n, d = X.shape
        lam = self.lam
        parent = self._struct_gain(g, h, lam)
        best = (-np.inf, None, None)  # maximize gain
        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, gs, hs = X[order, f], g[order], h[order]
            Gl, Hl = np.cumsum(gs)[:-1], np.cumsum(hs)[:-1]
            Gr, Hr = g.sum() - Gl, h.sum() - Hl
            gain = 0.5 * (Gl ** 2 / (Hl + lam) + Gr ** 2 / (Hr + lam)) - parent - self.gamma
            valid = xs[:-1] != xs[1:]
            gain = np.where(valid, gain, -np.inf)
            j = int(np.argmax(gain))
            if gain[j] > best[0]:
                best = (gain[j], f, (xs[j] + xs[j + 1]) / 2)
        return best

    def _build(self, X, target, depth, g=None, h=None):
        node = _RegNode()
        n = len(X)
        if self.mode == "newton":
            if n < self.min_samples_split or depth >= self.max_depth:
                node.value = self._leaf_weight(g, h, self.lam)
                return node
            gain, f, t = self._best_split_newton(X, g, h)
            if f is None or gain <= 0:        # gain<=0 means split not worth it
                node.value = self._leaf_weight(g, h, self.lam)
                return node
            m = X[:, f] <= t
            node.feature, node.threshold = f, t
            node.left = self._build(X[m], None, depth + 1, g[m], h[m])
            node.right = self._build(X[~m], None, depth + 1, g[~m], h[~m])
            return node
        # variance mode (fits `target` = pseudo-residuals)
        if n < self.min_samples_split or depth >= self.max_depth or np.var(target) == 0:
            node.value = float(np.mean(target))
            return node
        _, f, t = self._best_split_var(X, target)
        if f is None:
            node.value = float(np.mean(target))
            return node
        m = X[:, f] <= t
        node.feature, node.threshold = f, t
        node.left = self._build(X[m], target[m], depth + 1)
        node.right = self._build(X[~m], target[~m], depth + 1)
        return node

    def fit(self, X, target=None, g=None, h=None):
        X = np.asarray(X, float)
        self.root = self._build(X, target, 0, g, h)
        return self

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root)
                         for x in np.asarray(X, float)])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------------------
# 1. NumPy implementation: Gradient Boosting (regression + classification)
# ---------------------------------------------------------------------------
class GradientBoostingNumPy:
    """Stagewise additive boosting of shallow regression trees.

    loss="squared"  -> regression (pseudo-residuals = y - F)
    loss="logistic" -> binary classification (deviance / log-loss)

    method="gradient" : first-order (Friedman) — fit tree to negative gradient.
    method="newton"   : second-order (XGBoost) — fit tree using g & h with a
                        regularized closed-form leaf weight w* = -G/(H+lambda).
    """

    def __init__(self, loss="squared", method="gradient", n_estimators=100,
                 learning_rate=0.1, max_depth=3, subsample=1.0,
                 lam=1.0, gamma=0.0, seed=SEED):
        self.loss = loss
        self.method = method
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample        # stochastic GB: fraction of rows / round
        self.lam = lam
        self.gamma = gamma
        self.seed = seed
        self.trees_ = []
        self.train_loss_ = []

    # ---- loss-specific quantities ----
    def _init_raw(self, y):
        # F_0 = argmin_c sum loss(y, c)
        if self.loss == "squared":
            return float(np.mean(y))                  # mean minimizes MSE
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1 - p)))             # log-odds minimizes log-loss

    def _neg_gradient(self, y, F):
        # -dL/dF  (the pseudo-residual)
        if self.loss == "squared":
            return y - F                              # residual
        return y - _sigmoid(F)                        # logistic: y - p

    def _hessian(self, y, F):
        # d^2 L / dF^2
        if self.loss == "squared":
            return np.ones_like(F)
        p = _sigmoid(F)
        return np.clip(p * (1 - p), 1e-6, None)

    def _loss_value(self, y, F):
        if self.loss == "squared":
            return float(np.mean((y - F) ** 2))
        p = _sigmoid(F)
        return float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(X)
        rng = np.random.default_rng(self.seed)
        self.F0_ = self._init_raw(y)
        F = np.full(n, self.F0_)
        self.trees_, self.train_loss_ = [], []

        for _ in range(self.n_estimators):
            # stochastic subsampling of rows (Friedman 2002)
            if self.subsample < 1.0:
                idx = rng.choice(n, max(1, int(self.subsample * n)), replace=False)
            else:
                idx = np.arange(n)

            if self.method == "newton":
                # second order: gradient g = dL/dF, hessian h = d^2L/dF^2
                g = -self._neg_gradient(y[idx], F[idx])   # note: g = dL/dF = -(neg grad)
                h = self._hessian(y[idx], F[idx])
                tree = _RegTree(max_depth=self.max_depth, lam=self.lam,
                                gamma=self.gamma, mode="newton").fit(X[idx], g=g, h=h)
            else:
                # first order: fit tree to the negative gradient (pseudo-residual)
                resid = self._neg_gradient(y[idx], F[idx])
                tree = _RegTree(max_depth=self.max_depth,
                                mode="variance").fit(X[idx], target=resid)

            # shrinkage: F <- F + nu * tree(X)  (learning rate regularizes)
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.train_loss_.append(self._loss_value(y, F))
        return self

    def decision_function(self, X):
        """Raw additive score F(x) = F0 + nu * sum_m tree_m(x)."""
        F = np.full(len(np.asarray(X, float)), self.F0_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        assert self.loss == "logistic"
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        F = self.decision_function(X)
        if self.loss == "logistic":
            return (_sigmoid(F) >= 0.5).astype(int)
        return F


# ---------------------------------------------------------------------------
# 2. "PyTorch" note + scikit-learn cross-check
# ---------------------------------------------------------------------------
# Gradient boosting fits *discrete* regression trees stagewise; the trees are
# grown greedily and are not differentiable, so an idiomatic PyTorch model is not
# the natural tool (the "gradient" here is the functional gradient of the loss
# w.r.t. the model's predictions, computed in closed form — not autograd). We
# cross-check against scikit-learn's GradientBoosting estimators.
def sklearn_reference(X, y, task="classification", **kw):
    from sklearn.ensemble import (GradientBoostingClassifier,
                                   GradientBoostingRegressor)
    Model = (GradientBoostingClassifier if task == "classification"
             else GradientBoostingRegressor)
    return Model(random_state=SEED, **kw).fit(X, y)


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    from sklearn.datasets import make_friedman1, make_classification

    # ---------- regression (squared loss) ----------
    Xr, yr = make_friedman1(n_samples=400, noise=1.0, random_state=SEED)
    Xtr, ytr, Xte, yte = Xr[:320], yr[:320], Xr[320:], yr[320:]
    gbr = GradientBoostingNumPy(loss="squared", n_estimators=200,
                                learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
    print(f"[reg] GBM MSE={np.mean((gbr.predict(Xte) - yte) ** 2):.3f}  "
          f"(final train loss {gbr.train_loss_[-1]:.3f})")
    skr = sklearn_reference(Xtr, ytr, task="regression",
                            n_estimators=200, learning_rate=0.1, max_depth=3)
    print(f"[reg] sklearn GBM MSE={np.mean((skr.predict(Xte) - yte) ** 2):.3f}")

    # ---------- binary classification (logistic loss) ----------
    Xc, yc = make_classification(n_samples=500, n_features=12, n_informative=6,
                                 n_redundant=2, random_state=SEED)
    Xctr, yctr, Xcte, ycte = Xc[:380], yc[:380], Xc[380:], yc[380:]
    gbc = GradientBoostingNumPy(loss="logistic", method="gradient",
                                n_estimators=150, learning_rate=0.1,
                                max_depth=3).fit(Xctr, yctr)
    print(f"[clf] GBM (gradient) acc={np.mean(gbc.predict(Xcte) == ycte):.3f}")

    # ---------- second-order / regularized (XGBoost-style) ----------
    gbn = GradientBoostingNumPy(loss="logistic", method="newton",
                                n_estimators=150, learning_rate=0.1,
                                max_depth=3, lam=1.0, gamma=0.0).fit(Xctr, yctr)
    print(f"[clf] GBM (Newton/XGB) acc={np.mean(gbn.predict(Xcte) == ycte):.3f}")

    # stochastic gradient boosting
    gbs = GradientBoostingNumPy(loss="logistic", n_estimators=150,
                                learning_rate=0.1, subsample=0.6).fit(Xctr, yctr)
    print(f"[clf] GBM (subsample=0.6) acc={np.mean(gbs.predict(Xcte) == ycte):.3f}")

    skc = sklearn_reference(Xctr, yctr, task="classification",
                            n_estimators=150, learning_rate=0.1, max_depth=3)
    print(f"[clf] sklearn GBM acc={np.mean(skc.predict(Xcte) == ycte):.3f}")


if __name__ == "__main__":
    demo()

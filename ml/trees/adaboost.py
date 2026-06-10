"""
AdaBoost (Adaptive Boosting)
============================
Boost a sequence of *weak* learners (here: decision stumps — depth-1 trees) into
a strong classifier. After each round, misclassified points get **up-weighted**
so the next learner concentrates on the hard cases; each learner gets a vote
weight $\\alpha_m$ tied to its (weighted) accuracy. The final prediction is a
weighted vote. AdaBoost is exactly forward stagewise additive modeling under the
**exponential loss**.

Variants implemented here:
    - Binary AdaBoost (Freund & Schapire, discrete AdaBoost.M1)
    - SAMME multiclass boosting (Zhu et al.) — the K-class generalization
    - Decision-stump base learner (weighted Gini), implemented locally

Training techniques demonstrated:
    - Sample reweighting (boosting hard examples)
    - The alpha (learner-weight) update derived from exponential loss
    - Stagewise additive modeling

References:
    - Freund & Schapire (1997), "A Decision-Theoretic Generalization of On-Line
      Learning and an Application to Boosting"
    - Zhu, Zou, Rosset, Hastie (2009), "Multi-class AdaBoost" (SAMME)
    - Friedman, Hastie, Tibshirani (2000), "Additive Logistic Regression"
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# Local weighted decision stump (depth-1 CART) — the canonical weak learner.
# Splits to minimize *weighted* Gini; predicts the weighted-majority class in
# each side. Self-contained (no import of decision_tree.py).
# ---------------------------------------------------------------------------
class _Stump:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.feature = self.threshold = None
        self.left_class = self.right_class = None

    @staticmethod
    def _weighted_gini(w_per_class, w_total):
        if w_total <= 0:
            return 0.0
        p = w_per_class / w_total
        return 1.0 - (p ** 2).sum()

    def fit(self, X, y, w):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        n, d = X.shape
        C = self.n_classes
        best = (np.inf, None, None, None, None)   # (impurity, f, t, lc, rc)
        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys, ws = X[order, f], y[order], w[order]
            # per-class weighted prefix sums -> evaluate every threshold at once
            oh = np.eye(C)[ys] * ws[:, None]          # (n, C) weighted one-hot
            cum = np.cumsum(oh, axis=0)               # left class-weight prefix
            tot = cum[-1]
            valid = xs[:-1] != xs[1:]
            for j in np.where(valid)[0]:
                wl = cum[j]                            # left class weights
                wr = tot - wl                          # right class weights
                gl, gr = wl.sum(), wr.sum()
                imp = (gl * self._weighted_gini(wl, gl)
                       + gr * self._weighted_gini(wr, gr)) / (gl + gr + 1e-12)
                if imp < best[0]:
                    best = (imp, f, (xs[j] + xs[j + 1]) / 2,
                            int(wl.argmax()), int(wr.argmax()))
        _, self.feature, self.threshold, self.left_class, self.right_class = best
        # degenerate fallback: no valid split found
        if self.feature is None:
            self.feature, self.threshold = 0, 0.0
            maj = int((np.eye(C)[y] * w[:, None]).sum(0).argmax())
            self.left_class = self.right_class = maj
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return np.where(X[:, self.feature] <= self.threshold,
                        self.left_class, self.right_class)


# ---------------------------------------------------------------------------
# 1. NumPy implementation: SAMME AdaBoost (binary is the K=2 special case)
# ---------------------------------------------------------------------------
class AdaBoostNumPy:
    """Discrete SAMME AdaBoost with decision-stump weak learners.

    For K=2 this reduces to classic Freund-Schapire AdaBoost.M1. The SAMME
    learner weight adds a $\\log(K-1)$ term so a learner only needs to beat
    *random* (accuracy > 1/K), not 1/2, to contribute positively.
    """

    def __init__(self, n_estimators=50, seed=SEED):
        self.n_estimators = n_estimators
        self.seed = seed
        self.stumps_ = []
        self.alphas_ = []
        self.errors_ = []

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        n = len(X)
        self.n_classes = int(y.max()) + 1
        K = self.n_classes
        w = np.full(n, 1.0 / n)            # uniform initial sample weights
        self.stumps_, self.alphas_, self.errors_ = [], [], []

        for _ in range(self.n_estimators):
            stump = _Stump(K).fit(X, y, w)
            pred = stump.predict(X)
            miss = (pred != y).astype(float)
            err = float(np.dot(w, miss) / w.sum())          # weighted error
            err = min(max(err, 1e-10), 1 - 1e-10)           # clip for stability

            # SAMME learner weight (derivation in the notebook):
            #   alpha = log((1-err)/err) + log(K-1)
            alpha = np.log((1 - err) / err) + np.log(K - 1)
            if alpha <= 0 and self.stumps_:
                # learner worse than random and we already have learners -> stop
                break

            # reweight: up-weight misclassified, down-weight correct, renormalize
            w = w * np.exp(alpha * miss)
            w = w / w.sum()

            self.stumps_.append(stump)
            self.alphas_.append(alpha)
            self.errors_.append(err)
        self.alphas_ = np.array(self.alphas_)
        return self

    def decision_function(self, X):
        """Aggregate weighted votes per class: (n, K) score matrix."""
        n = len(np.asarray(X, float))
        scores = np.zeros((n, self.n_classes))
        for stump, alpha in zip(self.stumps_, self.alphas_):
            pred = stump.predict(X)
            scores[np.arange(n), pred] += alpha          # each stump votes alpha
        return scores

    def predict(self, X):
        return self.decision_function(X).argmax(1)

    def staged_predict(self, X):
        """Yield the ensemble prediction after each added stump (for curves)."""
        n = len(np.asarray(X, float))
        scores = np.zeros((n, self.n_classes))
        for stump, alpha in zip(self.stumps_, self.alphas_):
            scores[np.arange(n), stump.predict(X)] += alpha
            yield scores.argmax(1)


# ---------------------------------------------------------------------------
# 2. "PyTorch" note + scikit-learn cross-check
# ---------------------------------------------------------------------------
# AdaBoost stacks discrete, non-differentiable decision stumps with closed-form
# reweighting — there is no gradient to backpropagate, so an idiomatic PyTorch
# model is not the natural tool here. We cross-check the from-scratch SAMME
# implementation against scikit-learn's AdaBoostClassifier (also SAMME).
def sklearn_reference(X, y, **kw):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    kw.setdefault("n_estimators", 50)
    stump = DecisionTreeClassifier(max_depth=1)
    # sklearn >=1.6 dropped the `algorithm` arg (SAMME is now the only option);
    # older versions need algorithm="SAMME" to match our discrete boosting.
    try:
        return AdaBoostClassifier(estimator=stump, algorithm="SAMME",
                                  random_state=SEED, **kw).fit(X, y)
    except TypeError:
        return AdaBoostClassifier(estimator=stump,
                                  random_state=SEED, **kw).fit(X, y)


# ---------------------------------------------------------------------------
# 3. Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED)
    from sklearn.datasets import make_classification

    # ---------- binary (K=2) ----------
    X, y = make_classification(n_samples=400, n_features=10, n_informative=5,
                               n_redundant=2, random_state=SEED)
    Xtr, ytr, Xte, yte = X[:300], y[:300], X[300:], y[300:]
    ada = AdaBoostNumPy(n_estimators=50).fit(Xtr, ytr)
    print(f"[bin] AdaBoost ({len(ada.stumps_)} stumps) acc="
          f"{np.mean(ada.predict(Xte) == yte):.3f}  "
          f"first/last err={ada.errors_[0]:.3f}/{ada.errors_[-1]:.3f}")
    print(f"[bin] single stump acc="
          f"{np.mean(_Stump(2).fit(Xtr, ytr, np.full(len(Xtr), 1/len(Xtr))).predict(Xte) == yte):.3f}")
    sk = sklearn_reference(Xtr, ytr, n_estimators=50)
    print(f"[bin] sklearn SAMME acc={np.mean(sk.predict(Xte) == yte):.3f}")

    # ---------- multiclass (K=3) via SAMME ----------
    Xm, ym = make_classification(n_samples=600, n_features=12, n_informative=8,
                                 n_classes=3, n_clusters_per_class=1, random_state=SEED)
    Xmtr, ymtr, Xmte, ymte = Xm[:450], ym[:450], Xm[450:], ym[450:]
    adam = AdaBoostNumPy(n_estimators=80).fit(Xmtr, ymtr)
    print(f"[mc ] SAMME ({len(adam.stumps_)} stumps) acc={np.mean(adam.predict(Xmte) == ymte):.3f}")
    skm = sklearn_reference(Xmtr, ymtr, n_estimators=80)
    print(f"[mc ] sklearn SAMME acc={np.mean(skm.predict(Xmte) == ymte):.3f}")

    # boosting curve: test accuracy improves as stumps accumulate
    accs = [np.mean(p == ymte) for p in adam.staged_predict(Xmte)]
    print(f"[mc ] staged acc @ 1/10/40/last: "
          f"{accs[0]:.3f}/{accs[9]:.3f}/{accs[39]:.3f}/{accs[-1]:.3f}")


if __name__ == "__main__":
    demo()

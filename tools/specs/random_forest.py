from tools.nbreg import register, md, code, show, run_demo

MOD = "random_forest"


@register("random_forest", "ml/trees/random_forest.ipynb")
def build():
    return [
        md(r"""
# Random Forest — averaging de-correlated trees

> Tutorial pair for [`random_forest.py`](random_forest.py).

## 1. Intuition
A single deep decision tree has *low bias but high variance*: perturb the data a
little and it grows a very different tree. A **random forest** grows many such
trees, each on a different bootstrap resample and each split restricted to a
random subset of features, then **averages** their predictions. Averaging many
noisy-but-unbiased predictors cancels the noise — variance drops, bias stays
low. The samples left out of each bootstrap (the *out-of-bag* set) give a free
validation estimate.
"""),
        md(r"""
## 2. Concept (the slide)
- **Bagging:** each tree trains on a bootstrap sample (draw $n$ rows *with*
  replacement) → trees see slightly different data.
- **Random feature subspaces:** at every split consider only $m\ll d$ random
  features (`max_features`, e.g. $\sqrt d$). This **de-correlates** the trees,
  which is what makes averaging effective.
- **Aggregate:** classification = vote (hard) or average probabilities (soft);
  regression = mean.
- **OOB error:** for each sample, aggregate only the trees that did *not* train
  on it → an unbiased generalization estimate without a hold-out set.
- **Importances:** sum the impurity decrease attributed to each feature.
"""),
        md(r"""
## 3. Math derivation

**Why averaging reduces variance.** Let each tree be an unbiased predictor with
variance $\sigma^2$ and pairwise correlation $\rho$ between trees. The variance
of the average of $B$ such predictors is

$$\operatorname{Var}\!\left(\frac1B\sum_{b=1}^{B} T_b(x)\right)
 = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .$$

As $B\to\infty$ the second term vanishes, leaving $\rho\,\sigma^2$. So the
**limiting variance is controlled by the correlation $\rho$**, not by $B$. Bagging
alone lowers the $\tfrac{1-\rho}{B}\sigma^2$ part; the *random feature subspace*
attacks $\rho$ itself by forcing trees to split on different features —
this is the key idea beyond plain bagging.

Bias is essentially unchanged: $\mathbb E[\frac1B\sum_b T_b]=\mathbb E[T]$, so a
forest of low-bias trees stays low-bias. Net effect: **same bias, much lower
variance** ⇒ lower expected test error (bias–variance decomposition
$\text{Err}=\text{bias}^2+\text{variance}+\sigma_\varepsilon^2$).

**Bootstrap & OOB.** Drawing $n$ samples with replacement, the probability a
given row is *never* picked is
$\left(1-\tfrac1n\right)^n \xrightarrow{n\to\infty} e^{-1}\approx 0.368.$
So each tree leaves out $\approx 37\%$ of the data — those are its OOB samples,
and aggregating predictions over the trees for which a row is OOB yields a
cross-validation-like error estimate essentially for free.

**Feature subsampling sizes.** Common defaults: $m=\sqrt d$ (classification),
$m=d/3$ (regression). Smaller $m$ → more de-correlation (lower $\rho$) but each
tree is individually weaker (higher $\sigma^2$); the optimum trades these off.
"""),
        md("## 4. NumPy implementation (bagged CART + random subspaces + OOB)"),
        show(MOD, "RandomForestNumPy"),
        md(r"""## 5. Reference / cross-check — why not PyTorch?

A random forest is a *bag of discrete, greedily-grown trees*: there is no loss to
differentiate and no parameters to update by gradient descent, so an idiomatic
PyTorch model is not the natural tool (PyTorch shines for differentiable models).
We validate the from-scratch forest against scikit-learn's optimized
`RandomForest*`."""),
        show(MOD, "sklearn_reference"),
        md("## 6. Train — single tree vs forest, OOB, importances, and the cross-check"),
        run_demo(MOD),
        md("## 7. Visualization — variance reduction and decision regions"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import random_forest as M

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 200),
                     np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, n_est in zip(axes, (1, 10, 100)):
    rf = M.RandomForestNumPy(n_estimators=n_est, max_features="sqrt").fit(X, y)
    zz = rf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.3, cmap="coolwarm")
    ax.scatter(X[:,0], X[:,1], c=y, s=12, edgecolor="k", cmap="coolwarm")
    ax.set_title(f"{n_est} tree(s)")   # boundary smooths as trees accumulate
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- More trees never hurts accuracy (variance keeps dropping toward $\rho\sigma^2$)
  — it only costs compute. Tune **`max_features`**, depth, and `min_samples`.
- **OOB error** is a cheap, honest validation signal — use it instead of a split
  when data is scarce.
- Impurity-based importances are biased toward high-cardinality / continuous
  features; prefer permutation importance when it matters.
- Forests **smooth** but stay axis-aligned; they cannot extrapolate beyond the
  training range (a regression-tree limitation). For bias reduction by sequential
  error-fitting, use **gradient boosting** (next file).
"""),
    ]

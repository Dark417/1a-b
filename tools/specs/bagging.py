from tools.nbreg import register, md, code, show, run_demo

MOD = "bagging"


@register("bagging", "ml/ensemble/bagging.ipynb")
def build():
    return [
        md(r"""
# Bagging — bootstrap aggregation for variance reduction

> Tutorial pair for [`bagging.py`](bagging.py).

## 1. Intuition
Take one unstable learner — say a fully grown decision tree, which changes a lot
if you jiggle the data — and train **many copies** of it on different bootstrap
resamples of your training set. Average their predictions (vote for
classification). The individual trees are still wiggly, but their *average* is
smooth: the random errors cancel out. Bias barely changes; **variance drops**.
That is bagging in one sentence.
"""),
        md(r"""
## 2. Concept (the slide)
- **Bootstrap:** draw $n$ rows *with replacement* → each learner sees a slightly
  different dataset.
- **Aggregate:** mean (regression) or majority vote (classification).
- **Best for high-variance, low-bias learners** (deep trees). For stable
  learners (linear models) it barely helps.
- **OOB error:** the $\approx 37\%$ of rows left out of each bootstrap form a
  built-in validation set.
- **Random forest = bagging + per-split feature subsampling** (the extra step
  that de-correlates the trees).
"""),
        md(r"""
## 3. Math derivation

**Bootstrap.** A bootstrap sample draws $n$ indices uniformly with replacement.
The chance a particular row is omitted is
$\left(1-\frac1n\right)^n\to e^{-1}\approx0.368$, so each bag uses $\approx63\%$
distinct rows and leaves $\approx37\%$ out-of-bag.

**Variance of an average.** Let $\{T_b\}_{b=1}^B$ be predictors, each with
variance $\sigma^2$ and pairwise correlation $\rho$. Then for the bagged
predictor $\bar T=\frac1B\sum_b T_b$,

$$\operatorname{Var}(\bar T)=\frac1{B^2}\Big(\sum_b\operatorname{Var}(T_b)
  +\sum_{b\ne b'}\operatorname{Cov}(T_b,T_{b'})\Big)
  =\frac{\sigma^2}{B}+\frac{B-1}{B}\rho\sigma^2
  \;\xrightarrow{B\to\infty}\;\rho\,\sigma^2 .$$

- If the predictors were **independent** ($\rho=0$): variance shrinks like
  $\sigma^2/B$ — perfect averaging.
- In practice bootstrap samples overlap, so $\rho>0$ and the variance floors at
  $\rho\sigma^2$. This is *why* random forests add feature subsampling: to push
  $\rho$ down and lower that floor.

**Bias is unchanged.** $\mathbb E[\bar T]=\mathbb E[T]$, so bagging does not fix a
biased learner — it only removes variance. In the bias–variance decomposition
$\mathbb E[(y-\hat f)^2]=\text{bias}^2+\text{variance}+\sigma_\varepsilon^2$,
bagging targets the middle term.

**OOB estimate.** For each sample $i$, average the predictions of only those bags
that did *not* contain $i$; comparing to $y_i$ over all $i$ gives an
(approximately) unbiased test-error estimate without a hold-out split.
"""),
        md("## 4. NumPy implementation (generic bagging + OOB + variance experiment)"),
        show(MOD, "BaggingNumPy"),
        md(r"""## 5. Reference / cross-check — why not PyTorch?

Bagging is a *meta-procedure* wrapped around an arbitrary base learner (here,
discrete decision trees / stumps). There is nothing to differentiate at the
ensemble level, so an idiomatic PyTorch model is not the natural tool. We
cross-check against scikit-learn's `Bagging*` estimators."""),
        show(MOD, "sklearn_reference"),
        md("## 6. Train — single tree vs bag, OOB, and an explicit variance experiment"),
        run_demo(MOD),
        md("## 7. Visualization — averaging smooths a wiggly regressor"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import bagging as M

# 1-D noisy sine: a deep tree wiggles, the bag of trees smooths it out
rng = np.random.default_rng(0)
Xtr = np.sort(rng.uniform(-3, 3, 80)).reshape(-1, 1)
ytr = np.sin(Xtr).ravel() + 0.3 * rng.standard_normal(80)
Xte = np.linspace(-3, 3, 400).reshape(-1, 1)

single = M._Tree(task="regression").fit(Xtr, ytr)
bag = M.BaggingNumPy(lambda: M._Tree(task="regression"), n_estimators=50,
                     task="regression").fit(Xtr, ytr)

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for a, (model, title) in zip(ax, [(single, "single deep tree (high variance)"),
                                   (bag, "bagged 50 trees (variance reduced)")]):
    a.scatter(Xtr, ytr, s=12, c="k", alpha=.5)
    a.plot(Xte, np.sin(Xte), "g--", lw=1, label="true")
    a.plot(Xte, model.predict(Xte), "r", lw=1.5, label="prediction")
    a.set_title(title); a.legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Bagging **reduces variance, not bias** — use it on *overfit-prone* learners
  (deep trees), not on already-stable ones.
- Variance floors at $\rho\sigma^2$; to go lower, **de-correlate** the learners →
  random forest (feature subsampling).
- **OOB error** is a free validation estimate; more bags only help (and cost
  compute), they don't overfit.
- For **bias** reduction, you need sequential error-correction instead →
  boosting (AdaBoost / gradient boosting).
"""),
    ]

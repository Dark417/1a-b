from tools.nbreg import register, md, code, show, run_demo

MOD = "decision_tree"


@register("decision_tree", "ml/trees/decision_tree.ipynb")
def build():
    return [
        md(r"""
# Decision Trees (CART) — learning a flowchart

> Tutorial pair for [`decision_tree.py`](decision_tree.py).

## 1. Intuition
A decision tree is a flowchart of yes/no questions on the features. Each question
splits the data to make the resulting groups **purer** (more single-class). Keep
splitting until groups are pure enough, then predict the majority class (or mean
value) in each leaf. Interpretable, nonlinear, scale-invariant.
"""),
        md(r"""
## 2. Concept (the slide)
- **Greedy growth:** at each node pick the (feature, threshold) that most reduces
  impurity.
- **Impurity:** Gini or entropy (classification), variance/MSE (regression).
- **Pre-pruning:** cap `max_depth` / require `min_samples_split` to avoid
  memorizing noise (a fully grown tree overfits).
- Base learner for **random forests** (bagging) and **gradient boosting**.
"""),
        md(r"""
## 3. Math derivation

**Impurity measures** for a node with class proportions $p_c$:

$$\text{Gini}=1-\sum_c p_c^2,\qquad
  \text{Entropy}=-\sum_c p_c\log_2 p_c,\qquad
  \text{(regression) } \text{MSE}=\tfrac1{|S|}\sum_{i\in S}(y_i-\bar y)^2 .$$

**Best split.** Splitting set $S$ into $S_L,S_R$ has weighted child impurity

$$I_{\text{split}}=\frac{|S_L|}{|S|}\,I(S_L)+\frac{|S_R|}{|S|}\,I(S_R),$$

and the **information gain** is $\Delta=I(S)-I_{\text{split}}$. CART scans every
feature and every candidate threshold (midpoints between sorted unique values)
and picks the split with the largest $\Delta$ (= smallest $I_{\text{split}}$).
This is repeated recursively — a greedy, locally optimal procedure.

**Why Gini vs entropy?** Both peak at a uniform class mix and vanish at purity;
Gini avoids a logarithm (slightly cheaper) and usually gives near-identical
trees. Entropy/information gain is the ID3/C4.5 lineage.

**Overfitting & pruning.** An unconstrained tree can place every point in its own
leaf (zero training error, terrible generalization — pure **variance**).
Pre-pruning (`max_depth`, `min_samples_split`, `min_impurity_decrease`) or
post-pruning (cost-complexity $R_\alpha(T)=R(T)+\alpha|T|$) trades a little bias
for much less variance. You'll see the test accuracy peak at moderate depth.
"""),
        md("## 4. NumPy implementation (CART: classification + regression + pruning)"),
        show(MOD, "DecisionTreeNumPy"),
        md(r"""## 5. "PyTorch" note

Trees are discrete, non-differentiable greedy structures — there is no gradient to
backprop, so PyTorch isn't the natural tool (it shines for differentiable models;
*soft/differentiable* trees exist but are a research variant). We instead
cross-check against scikit-learn's optimized CART."""),
        show(MOD, "sklearn_reference"),
        md("## 6. Train — depth vs accuracy, and the sklearn cross-check"),
        run_demo(MOD),
        md("## 7. Visualization — decision regions get more complex with depth"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import decision_tree as M

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 200),
                     np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, depth in zip(axes, (1, 3, 8)):
    t = M.DecisionTreeNumPy(max_depth=depth).fit(X, y)
    zz = t.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.3, cmap="coolwarm")
    ax.scatter(X[:,0], X[:,1], c=y, s=12, edgecolor="k", cmap="coolwarm")
    ax.set_title(f"max_depth = {depth}")   # axis-aligned, staircase boundaries
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Boundaries are **axis-aligned staircases**; deep trees overfit (high variance).
- No feature scaling needed; handles mixed feature types and is interpretable.
- A single tree is weak/unstable → **ensembles**: random forest (bag many trees
  on bootstrap samples + feature subsets) and gradient boosting (fit trees to
  residuals). Those are the next files in `ml/trees/` and `ml/ensemble/`.
"""),
    ]

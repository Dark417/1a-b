from tools.nbreg import register, md, code, show, run_demo

MOD = "stacking"


@register("stacking", "ml/ensemble/stacking.ipynb")
def build():
    return [
        md(r"""
# Stacking — a meta-learner over base models

> Tutorial pair for [`stacking.py`](stacking.py).

## 1. Intuition
Different models make different mistakes. Instead of *averaging* them (voting) or
resampling *one* model (bagging), **stacking trains a second model to learn how
to combine the first ones**. The base models output predictions; a *meta-learner*
takes those predictions as its input features and learns the best blend — e.g.
"trust the tree near the boundary, the logistic model elsewhere." The one catch:
the meta-features must be generated **out-of-fold** so the meta-learner never
sees a base model predicting its own training data.
"""),
        md(r"""
## 2. Concept (the slide)
- **Level-0 (base) models:** heterogeneous learners (tree, logistic regression,
  k-NN, …) trained on the data.
- **Level-1 (meta) model:** trained on the base models' predictions as features.
- **Out-of-fold (OOF) trick:** generate each row's meta-features from base models
  fit on the *other* folds → no leakage. (If you used in-sample base predictions,
  the meta-learner would over-trust overfit base models.)
- **Passthrough:** optionally also feed the original features to the meta-learner.
- At test time, base models are refit on the full training set.
"""),
        md(r"""
## 3. Math derivation

**The leakage problem, formally.** Suppose base model $g$ overfits: on its own
training data $\hat g(x_i)\approx y_i$ even though it generalizes poorly. If the
meta-learner $f$ is trained on features $z_i=\hat g(x_i)$ computed *in-sample*,
those $z_i$ look almost perfect, so $f$ learns to put all its weight on $g$ —
and then fails at test time, where $g$ is no longer near-perfect. The meta-model
must see base predictions whose error distribution **matches test time**.

**Out-of-fold construction (Wolpert).** Partition the $n$ rows into $K$ folds
$\{V_1,\dots,V_K\}$. For each base model $g^{(b)}$ and each fold $k$:

$$g^{(b)}_{-k}=\text{fit on } \{(x_i,y_i): i\notin V_k\},\qquad
  z_i^{(b)}=g^{(b)}_{-k}(x_i)\ \text{ for } i\in V_k .$$

Every meta-feature $z_i^{(b)}$ thus comes from a model that **did not train on
row $i$**, so it carries an honest, test-like error. Stack the columns into
$Z\in\mathbb R^{n\times(B\cdot c)}$ ($c=1$ for regression, $K{-}1$ probability
columns per base for $K$-class classification — dropping one column avoids the
sum-to-one collinearity). Train the meta-learner

$$f^\star=\arg\min_f \sum_i \ell\big(y_i,\, f(z_i)\big).$$

**Test time.** Refit each base model on **all** $n$ rows (more data ⇒ better base
predictions), form $z(x)=\big(g^{(1)}(x),\dots,g^{(B)}(x)\big)$, and predict
$f^\star(z(x))$.

**Why it can beat any single base model.** Stacking can implement a *convex
combination* of base predictions (if $f$ is e.g. linear/logistic with positive
weights), so in the worst case it recovers the best single model; with diverse,
decorrelated bases it does strictly better — it is doing supervised model
selection *per region of feature space*. Using a **simple** meta-learner (linear,
ridge, logistic) is standard: the bases already did the heavy lifting, and a
simple blender resists overfitting the small meta-feature set.
"""),
        md("## 4. NumPy implementation (OOF meta-features + meta-learner)"),
        show(MOD, "StackingNumPy"),
        md(r"""## 5. Reference / cross-check — why not PyTorch?

Stacking is an *orchestration* layer: it cross-validates arbitrary base learners
(here a shallow tree, logistic regression, ridge) to build OOF meta-features and
fits a meta-learner. The orchestration has nothing to differentiate end-to-end,
so an idiomatic PyTorch model is not the natural tool. We cross-check against
scikit-learn's `Stacking*` estimators."""),
        show(MOD, "sklearn_reference"),
        md("## 6. Train — base models vs the stacked ensemble, and the cross-check"),
        run_demo(MOD),
        md("## 7. Visualization — base vs stacked decision regions"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import stacking as M

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 200),
                     np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

bases = [lambda: M._Tree(task="classification", max_depth=3),
         lambda: M._LogReg(lr=0.5, n_iter=500)]
models = {
    "tree (base)": bases[0]().fit(X, y),
    "logreg (base)": bases[1]().fit(X, y),
    "stacked": M.StackingNumPy(bases, lambda: M._LogReg(lr=0.5, n_iter=500),
                               task="classification", passthrough=True).fit(X, y),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.3, cmap="coolwarm")
    ax.scatter(X[:,0], X[:,1], c=y, s=12, edgecolor="k", cmap="coolwarm")
    ax.set_title(name)
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Always** build meta-features out-of-fold; in-sample base predictions leak and
  ruin the meta-learner.
- Keep the **meta-learner simple** (linear / ridge / logistic) to avoid
  overfitting the small set of meta-features; let the bases be diverse and strong.
- Diversity matters more than individual accuracy — decorrelated bases give the
  blender something to exploit.
- More expensive than voting (it refits bases $K{+}1$ times). When the bases are
  near-identical, plain **soft voting** is simpler and nearly as good.
"""),
    ]

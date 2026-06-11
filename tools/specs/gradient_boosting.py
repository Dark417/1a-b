from tools.nbreg import register, md, code, show, run_demo

MOD = "gradient_boosting"


@register("gradient_boosting", "01.ml/trees/gradient_boosting.ipynb")
def build():
    return [
        md(r"""
# Gradient Boosting — functional gradient descent with trees

> Tutorial pair for [`gradient_boosting.py`](gradient_boosting.py).

## 1. Intuition
Boosting builds the model **one tree at a time**. Start with a constant guess.
Look at where you're wrong — the *residuals* — and fit a small tree to those
errors. Add a shrunken copy of that tree to your running prediction and repeat.
Each tree nudges the prediction down the loss surface, so the ensemble keeps
*reducing bias* (the opposite emphasis from a random forest, which reduces
variance). This "fit the residual" recipe is exactly gradient descent — but in
the space of *functions*.
"""),
        md(r"""
## 2. Concept (the slide)
- **Additive model:** $F_M(x)=F_0(x)+\nu\sum_{m=1}^{M} h_m(x)$, each $h_m$ a
  shallow regression tree.
- **Stagewise:** freeze previous trees; the new tree fits the **negative
  gradient** of the loss at the current predictions (the *pseudo-residuals*).
- **Losses:** squared error → residuals $y-F$ (regression); logistic/deviance →
  $y-\sigma(F)$ (classification).
- **Shrinkage** $\nu$ (learning rate): take small steps → better generalization
  (needs more trees).
- **Stochastic GB:** subsample rows per round for speed + extra regularization.
- **Newton / XGBoost:** use the **second** derivative (Hessian) and an L2 penalty
  on leaf weights for a regularized, faster-converging step.
"""),
        md(r"""
## 3. Math derivation

**Functional gradient descent.** We want $F$ minimizing
$\mathcal L(F)=\sum_{i=1}^n \ell(y_i, F(x_i))$. Treat the vector of predictions
$\big(F(x_1),\dots,F(x_n)\big)$ as the parameters. Gradient descent would update
$F(x_i)\leftarrow F(x_i)-\nu\,g_i$ with the **functional gradient**

$$g_i=\frac{\partial \ell(y_i,F(x_i))}{\partial F(x_i)} .$$

But that only updates the values at training points. To **generalize**, we fit a
regression tree $h_m$ to the negative gradient (the *pseudo-residuals*
$r_i=-g_i$) and step in that direction:

$$F_m = F_{m-1} + \nu\, h_m,\qquad h_m \approx \arg\min_h \sum_i \big(r_i-h(x_i)\big)^2 .$$

**Squared error** $\ell=\tfrac12(y-F)^2$: $-g_i = y_i-F_{m-1}(x_i)$ — the
ordinary **residual**. So least-squares boosting literally fits each tree to the
residuals, and $F_0=\bar y$ (the minimizer of constant loss).

**Logistic loss** for $y\in\{0,1\}$ with $\ell=-[y\log p+(1-y)\log(1-p)]$,
$p=\sigma(F)$: one finds $\partial\ell/\partial F = \sigma(F)-y$, hence
$-g_i = y_i-\sigma(F_{m-1}(x_i))$, and $F_0=\log\frac{\bar y}{1-\bar y}$ (the
log-odds). Predictions come from $\sigma(F_M)$.

**Shrinkage.** Replacing $F_m=F_{m-1}+h_m$ with $F_m=F_{m-1}+\nu h_m$, $\nu\in(0,1]$,
is a learning rate. Small $\nu$ (e.g. $0.1$) regularizes — it prevents any single
tree from dominating and empirically lowers test error, at the cost of needing
more trees ($M\propto 1/\nu$).

**Newton boosting (XGBoost).** Second-order Taylor expansion of the loss around
$F_{m-1}$ with per-sample gradient $g_i$ and Hessian $h_i=\partial^2\ell/\partial F^2$:

$$\mathcal L(F_{m-1}+f)\approx \text{const}+\sum_i\Big(g_i f(x_i)+\tfrac12 h_i f(x_i)^2\Big)
  +\tfrac12\lambda\sum_j w_j^2 .$$

For a tree with leaves $j$ (region $I_j$), $f$ is constant $w_j$ on each leaf. Let
$G_j=\sum_{i\in I_j} g_i,\ H_j=\sum_{i\in I_j} h_i$. Minimizing the quadratic in
$w_j$ gives the **optimal leaf weight and value**

$$\boxed{\,w_j^\star=-\frac{G_j}{H_j+\lambda}\,},\qquad
  \mathcal L^\star=-\tfrac12\sum_j\frac{G_j^2}{H_j+\lambda}.$$

The **split gain** for partitioning a node into $L,R$ is therefore

$$\text{Gain}=\tfrac12\!\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
   -\frac{(G_L{+}G_R)^2}{H_L{+}H_R+\lambda}\right]-\gamma,$$

with $\gamma$ a minimum-gain (complexity) penalty. For squared loss $h_i\equiv1$
and Newton reduces to ordinary residual fitting; for logistic loss
$h_i=p_i(1-p_i)$ gives the curvature-aware step.
"""),
        md("## 4. NumPy implementation (GBM regression + classification + Newton/XGBoost)"),
        show(MOD, "GradientBoostingNumPy"),
        md(r"""## 5. Reference / cross-check — why not PyTorch?

The "gradient" in gradient boosting is the **functional** gradient of the loss
w.r.t. the model's predictions — computed in closed form, not via autograd. The
weak learners are *discrete, greedily-grown* regression trees, which are
non-differentiable, so an idiomatic PyTorch model is not the natural tool. We
cross-check against scikit-learn's `GradientBoosting*`."""),
        show(MOD, "sklearn_reference"),
        md("## 6. Train — regression, logistic classification, Newton, and stochastic GB"),
        run_demo(MOD),
        md("## 7. Visualization — loss decreasing per stage; shrinkage effect"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
import gradient_boosting as M

Xr, yr = make_friedman1(n_samples=400, noise=1.0, random_state=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for lr in (0.03, 0.1, 0.3, 1.0):
    gb = M.GradientBoostingNumPy(loss="squared", n_estimators=150,
                                 learning_rate=lr, max_depth=3).fit(Xr, yr)
    ax[0].plot(gb.train_loss_, label=f"lr={lr}")
ax[0].set_xlabel("boosting round"); ax[0].set_ylabel("train MSE")
ax[0].set_title("Shrinkage: smaller lr = slower, smoother descent"); ax[0].legend()

# first-order vs Newton on a classification problem
from sklearn.datasets import make_classification
Xc, yc = make_classification(n_samples=400, n_features=10, n_informative=6,
                             random_state=0)
g1 = M.GradientBoostingNumPy(loss="logistic", method="gradient",
                             n_estimators=120, learning_rate=0.1).fit(Xc, yc)
g2 = M.GradientBoostingNumPy(loss="logistic", method="newton",
                             n_estimators=120, learning_rate=0.1, lam=1.0).fit(Xc, yc)
ax[1].plot(g1.train_loss_, label="first-order (gradient)")
ax[1].plot(g2.train_loss_, label="second-order (Newton/XGB)")
ax[1].set_xlabel("boosting round"); ax[1].set_ylabel("train log-loss")
ax[1].set_title("Newton step converges faster"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Boosting **reduces bias** by sequentially correcting errors — complementary to
  bagging/forests (which reduce variance). It can overfit if you boost too long.
- The trio **(learning rate, n_estimators, max_depth)** is the core trade-off:
  small $\nu$ + many shallow trees + early stopping generalizes best.
- Sequential ⇒ **not embarrassingly parallel** like a forest; sensitive to
  noisy labels (it chases them).
- **Newton/XGBoost** uses curvature and an L2 leaf penalty $\lambda$ for a more
  regularized, faster-converging step; add $\gamma$ to prune low-gain splits.
- Always tune on a validation curve; the training loss alone keeps dropping.
"""),
    ]

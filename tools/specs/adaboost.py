from tools.nbreg import register, md, code, show, run_demo

MOD = "adaboost"


@register("adaboost", "ml/trees/adaboost.ipynb")
def build():
    return [
        md(r"""
# AdaBoost — boosting by reweighting hard examples

> Tutorial pair for [`adaboost.py`](adaboost.py).

## 1. Intuition
Train a sequence of *weak* learners (decision stumps). After each one, **increase
the weight** of the examples it got wrong, so the next learner focuses on the
hard cases. Give each learner a vote $\alpha_m$ proportional to how well it did.
The committee of reweighted stumps becomes a strong classifier. AdaBoost is, in
disguise, stagewise additive modeling under the **exponential loss**.
"""),
        md(r"""
## 2. Concept (the slide)
- **Weak learner:** a decision stump (depth-1 tree) — barely better than chance.
- **Sample weights** $w_i$: start uniform; after round $m$, up-weight
  misclassified points by $e^{\alpha_m}$ and renormalize.
- **Learner weight** $\alpha_m$: large when the weighted error $\varepsilon_m$ is
  small; can go negative if worse than random (we stop then).
- **Final vote:** $H(x)=\arg\max_k \sum_m \alpha_m\,\mathbb 1[h_m(x)=k]$.
- **SAMME** generalizes the two-class rule to $K$ classes with a $\log(K-1)$ term.
"""),
        md(r"""
## 3. Math derivation

**Exponential loss & forward stagewise additive modeling.** Consider binary
labels $y\in\{-1,+1\}$ and an additive score $F_M(x)=\sum_{m} \alpha_m h_m(x)$
with $h_m(x)\in\{-1,+1\}$. AdaBoost greedily minimizes the **exponential loss**

$$\mathcal L=\sum_{i=1}^n \exp\!\big(-y_i F_M(x_i)\big).$$

At round $m$, with $F_{m-1}$ fixed, define weights
$w_i^{(m)}=\exp(-y_i F_{m-1}(x_i))$. We choose $(\alpha,h)$ to minimize

$$\sum_i w_i^{(m)} e^{-\alpha y_i h(x_i)}
 = e^{-\alpha}\!\!\sum_{y_i=h(x_i)}\!\! w_i^{(m)}
 + e^{\alpha}\!\!\sum_{y_i\ne h(x_i)}\!\! w_i^{(m)} .$$

Let the **weighted error** be
$\varepsilon_m=\dfrac{\sum_i w_i^{(m)}\,\mathbb 1[h(x_i)\ne y_i]}{\sum_i w_i^{(m)}}$.
The bracket is $(e^{\alpha}-e^{-\alpha})\varepsilon_m + e^{-\alpha}$. For fixed
$\alpha>0$ this is minimized by the $h$ with the **smallest** $\varepsilon_m$
(so we just fit a weighted-error stump). Setting $\partial/\partial\alpha=0$:

$$-e^{-\alpha}(1-\varepsilon_m)+e^{\alpha}\varepsilon_m=0
 \;\Rightarrow\;
 \boxed{\;\alpha_m=\tfrac12\log\frac{1-\varepsilon_m}{\varepsilon_m}\;}$$

(the factor $\tfrac12$ or $1$ is a convention absorbed into the vote scale).
Substituting back, the weight update is
$w_i^{(m+1)}=w_i^{(m)}e^{-\alpha_m y_i h_m(x_i)}$, i.e. **multiply misclassified
weights by $e^{\alpha_m}$, correct ones by $e^{-\alpha_m}$**, then renormalize —
exactly the AdaBoost reweighting. So each step is one coordinate of greedy
descent on the exponential loss.

**SAMME (multiclass).** For $K$ classes with stumps predicting a label directly,
Zhu et al. show the analogous learner weight is

$$\alpha_m=\log\frac{1-\varepsilon_m}{\varepsilon_m}+\log(K-1).$$

The extra $\log(K-1)$ makes $\alpha_m>0$ whenever the learner beats *random*
guessing ($\varepsilon_m<1-\tfrac1K$), not merely $<\tfrac12$. For $K=2$ it
reduces to classic AdaBoost. The final classifier is the weighted plurality vote.

**Why it works / margins.** Minimizing exponential loss drives up the *margin*
$y_i F(x_i)$; AdaBoost keeps improving test error even after training error hits
zero because it keeps increasing margins (Schapire et al.). The flip side: the
exponential loss is **not robust to label noise** — outliers get exponentially
large weights.
"""),
        md("## 4. NumPy implementation (SAMME with weighted-Gini decision stumps)"),
        show(MOD, "AdaBoostNumPy"),
        md(r"""## 5. Reference / cross-check — why not PyTorch?

AdaBoost stacks *discrete, non-differentiable* decision stumps and reweights
samples with a closed-form rule — there is no gradient to backpropagate, so an
idiomatic PyTorch model is not the natural tool. We cross-check the from-scratch
SAMME implementation against scikit-learn's `AdaBoostClassifier` (also SAMME)."""),
        show(MOD, "sklearn_reference"),
        md("## 6. Train — binary & multiclass AdaBoost vs the sklearn cross-check"),
        run_demo(MOD),
        md("## 7. Visualization — test accuracy vs number of stumps; weight evolution"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import adaboost as M

X, y = make_classification(n_samples=600, n_features=12, n_informative=8,
                           n_classes=3, n_clusters_per_class=1, random_state=0)
Xtr, ytr, Xte, yte = X[:450], y[:450], X[450:], y[450:]
ada = M.AdaBoostNumPy(n_estimators=120).fit(Xtr, ytr)

acc_curve = [np.mean(p == yte) for p in ada.staged_predict(Xte)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(1, len(acc_curve) + 1), acc_curve)
ax[0].set_xlabel("number of stumps"); ax[0].set_ylabel("test accuracy")
ax[0].set_title("Boosting curve: weak learners -> strong ensemble")

ax[1].plot(ada.errors_, "o-", ms=3)
ax[1].axhline(1 - 1/3, ls="--", c="r", label="random (1 - 1/K)")
ax[1].set_xlabel("round"); ax[1].set_ylabel("weighted error of stump")
ax[1].set_title("Per-round weighted error stays < random"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- AdaBoost = forward stagewise fitting of the **exponential loss**; $\alpha_m$ and
  the reweighting both fall out of one line of algebra.
- **SAMME**'s $\log(K-1)$ term is what lets stumps contribute in the multiclass
  setting (beat $1/K$, not $1/2$).
- **Not robust to noise/outliers:** mislabeled points get exponentially
  up-weighted; consider gradient boosting with a robust loss (e.g. logistic /
  Huber) when labels are noisy.
- Weak base learners (stumps) are deliberate — strong base learners overfit fast
  under boosting. Stop when a learner can no longer beat random ($\alpha\le 0$).
"""),
    ]

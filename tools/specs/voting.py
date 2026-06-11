from tools.nbreg import register, md, code, show, run_demo

MOD = "voting"


@register("voting", "01.ml/ensemble/voting.ipynb")
def build():
    return [
        md(r"""
# Voting — let diverse classifiers vote

> Tutorial pair for [`voting.py`](voting.py).

## 1. Intuition
The simplest ensemble: train a few good, *different* classifiers and let them
vote. **Hard voting** counts predicted labels (majority wins). **Soft voting**
averages the predicted *probabilities* and takes the argmax — so a model that is
very confident and right can override several models that are unsure and wrong.
No meta-learner, no resampling: just aggregate. It works because diverse models
make *independent* mistakes that tend to cancel.
"""),
        md(r"""
## 2. Concept (the slide)
- **Hard voting:** $\hat y=\text{mode}\{h_b(x)\}$ — plurality of predicted labels.
- **Soft voting:** $\hat y=\arg\max_k \frac1B\sum_b p_b(k\mid x)$ — average
  probabilities (uses confidence, usually better).
- **Weighted voting:** trust some models more via weights $w_b$.
- **Requirement:** the base models should be **diverse** (different inductive
  biases) and individually *better than random* — else voting can hurt.
- Contrast: voting fixes the combiner (mean/mode); **stacking learns** it.
"""),
        md(r"""
## 3. Math derivation

**Why voting helps — the independent-errors argument.** Suppose $B$ classifiers
each have accuracy $p>\tfrac12$ on a binary problem and make **independent**
errors. Majority vote is correct when more than half are correct; the number
correct is $\text{Binomial}(B,p)$, so

$$\Pr[\text{majority correct}]=\sum_{k=\lceil B/2\rceil}^{B}\binom{B}{k}p^k(1-p)^{B-k}
  \xrightarrow{B\to\infty} 1 .$$

(Condorcet's jury theorem.) E.g. $B=15$ voters at $p=0.7$ give $\approx0.95$
ensemble accuracy. The catch is the word **independent**: if the classifiers are
identical their votes are perfectly correlated and the ensemble equals one model.
So **diversity is the whole game**.

**Soft vs hard.** Hard voting discards confidence. Soft voting averages the class
posteriors,

$$\bar p(k\mid x)=\frac{1}{\sum_b w_b}\sum_{b=1}^{B} w_b\, p_b(k\mid x),\qquad
  \hat y=\arg\max_k \bar p(k\mid x),$$

which is the Bayes-optimal combination if the $p_b$ are calibrated estimates of
the same posterior. Concretely, if two models say class A with probability $0.55$
and one says class B with probability $0.95$, hard voting picks A (2 vs 1) but
soft voting picks B ($\bar p_B=0.95/3\approx0.32$ vs $\bar p_A=1.10/3\approx0.37$
— A still wins here, but tilt the confidences and soft voting flips, correctly
following the confident model). In practice soft voting $\ge$ hard voting when
the base probabilities are reasonably calibrated.

**Bias–variance view.** Like bagging, averaging *uncorrelated* predictors reduces
variance ($\operatorname{Var}(\bar p)\to\rho\sigma^2$ as $B\to\infty$); unlike
bagging, the diversity here comes from using **different model families** rather
than resampling one family. Weighted voting lets you down-weight a weak/redundant
member.
"""),
        md("## 4. NumPy implementation (hard / soft / weighted voting over diverse bases)"),
        show(MOD, "VotingNumPy"),
        md(r"""## 5. Reference / cross-check — why not PyTorch?

Voting is a thin aggregation layer over already-trained, heterogeneous
classifiers (logistic regression, a tree, Gaussian NB). The combiner is a fixed
`argmax` of averaged probabilities or a plurality of labels — nothing to
differentiate — so an idiomatic PyTorch model is not the natural tool. We
cross-check against scikit-learn's `VotingClassifier`."""),
        show(MOD, "sklearn_reference"),
        md("## 6. Train — base accuracies vs hard / soft / weighted voting"),
        run_demo(MOD),
        md("## 7. Visualization — base vs voted decision regions"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import voting as M

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 200),
                     np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

def fresh():
    return [("lr", M._LogReg()), ("dt", M._Tree(max_depth=5)), ("gnb", M._GaussianNB())]

models = {
    "logreg": M._LogReg().fit(X, y),
    "tree": M._Tree(max_depth=5).fit(X, y),
    "soft vote": M.VotingNumPy(fresh(), voting="soft").fit(X, y),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    zz = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=.3, cmap="coolwarm")
    ax.scatter(X[:,0], X[:,1], c=y, s=12, edgecolor="k", cmap="coolwarm")
    ax.set_title(name)   # voted boundary blends linear + axis-aligned biases
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Soft voting usually beats hard voting** — but only if the base probabilities
  are roughly *calibrated*; uncalibrated models can mislead the average.
- **Diversity is essential** (Condorcet): correlated/identical models give no
  gain. Mix model families and feature views.
- A base model worse than random *drags the ensemble down* — drop it or
  down-weight it.
- Voting has no learned combiner; when you want the blend itself optimized, use
  **stacking**. When you want variance reduction from *one* family, use bagging.
"""),
    ]

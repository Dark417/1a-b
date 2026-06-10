from tools.nbreg import register, md, code, show, run_demo

MOD = "logistic_regression"


@register("logistic_regression", "ml/linear-models/logistic_regression.ipynb")
def build():
    return [
        md(r"""
# Logistic Regression — sigmoid, cross-entropy, softmax

> Tutorial pair for [`logistic_regression.py`](logistic_regression.py).

## 1. Intuition
Linear regression outputs any real number — useless for "is this spam (0/1)?".
We squash the linear score through a **sigmoid** so the output is a probability
in $(0,1)$, then train it to assign high probability to the correct class. It is
*the* single-neuron classifier and the building block of every neural net.
"""),
        md(r"""
## 2. Concept (the slide)
- **Model (binary):** $p=\sigma(z),\ z=\mathbf w^\top\mathbf x+b,\ \sigma(z)=\frac{1}{1+e^{-z}}$.
- **Decision:** predict class 1 if $p\ge 0.5$ (i.e. $z\ge 0$) — a *linear* boundary.
- **Loss:** binary cross-entropy (negative log-likelihood of a Bernoulli).
- **Multiclass:** replace sigmoid with **softmax** over $K$ logits; loss is
  categorical cross-entropy.
"""),
        md(r"""
## 3. Math derivation

**Likelihood.** Each label is Bernoulli: $P(y\mid\mathbf x)=p^{y}(1-p)^{1-y}$.
The negative log-likelihood over the data is

$$\mathcal{L}=-\frac1n\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big].$$

**The clean gradient.** Use $\sigma'(z)=\sigma(z)(1-\sigma(z))$. For one example,

$$\frac{\partial \ell}{\partial z}
 =-\Big(\frac{y}{p}-\frac{1-y}{1-p}\Big)\,p(1-p)
 = p-y.$$

So, exactly like linear regression but with $p$ in place of $\hat y$,

$$\boxed{\;\nabla_{\mathbf w}\mathcal{L}=\frac1n X^\top(\mathbf p-\mathbf y),\qquad
 \frac{\partial\mathcal L}{\partial b}=\frac1n\sum_i(p_i-y_i).\;}$$

There is **no closed form** (the equations are transcendental) → we use gradient
descent / Newton's method (IRLS).

**Softmax generalization.** With logits $\mathbf z=W^\top\mathbf x+\mathbf b$ and
$p_k=\dfrac{e^{z_k}}{\sum_j e^{z_j}}$, cross-entropy $-\log p_{y}$ has the same
elegant gradient $\nabla_{\!W}\mathcal L=\frac1nX^\top(P-Y)$ where $Y$ is one-hot.
This "predicted minus target" form is why these losses pair so naturally with
their output nonlinearities.
"""),
        md("## 4. NumPy implementation"),
        show(MOD, "LogisticRegressionNumPy"),
        md("## 5. PyTorch implementation\n`BCEWithLogitsLoss` / `CrossEntropyLoss` fuse the sigmoid/softmax with the log for numerical stability."),
        show(MOD, "LogisticRegressionTorch"),
        md("## 6. Train — binary and softmax, NumPy vs PyTorch"),
        run_demo(MOD),
        md("## 7. Visualization — decision boundary & loss curve"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import logistic_regression as M

X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=0)
X = (X - X.mean(0)) / X.std(0)
m = M.LogisticRegressionNumPy(lr=0.5, n_iters=2000).fit(X, y)

xx, yy = np.meshgrid(np.linspace(*[X[:,0].min()-1, X[:,0].max()+1], 200),
                     np.linspace(*[X[:,1].min()-1, X[:,1].max()+1], 200))
grid = np.c_[xx.ravel(), yy.ravel()]
proba = m.predict_proba(grid).reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].contourf(xx, yy, proba, levels=20, cmap="RdBu", alpha=.7)
ax[0].scatter(X[:,0], X[:,1], c=y, edgecolor="k", s=15, cmap="RdBu")
ax[0].set_title("P(y=1) and the linear boundary")
ax[1].plot(m.history); ax[1].set_xlabel("iter"); ax[1].set_ylabel("cross-entropy")
ax[1].set_title("Training loss")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways
- Sigmoid+BCE and softmax+CE both give the gradient $\frac1nX^\top(\hat y-y)$ —
  remember this shape.
- The boundary is **linear**; for nonlinear data, add features or hidden layers.
- Always use the *logits* loss (`BCEWithLogitsLoss`) for stability.

**Next:** stack these neurons with nonlinearities → [the MLP](../../dl/mlp/mlp.ipynb).
"""),
    ]

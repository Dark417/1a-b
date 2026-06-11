from tools.nbreg import register, md, code, show, run_demo

MOD = "perceptron"


@register("perceptron", "01.ml/perceptron/perceptron.ipynb")
def build():
    return [
        md(r"""
# The Perceptron — the first trainable neuron (1958)

> Tutorial pair for [`perceptron.py`](perceptron.py).

## 1. Intuition
A perceptron draws a straight line and says "this side = +1, that side = −1." It
**learns from its mistakes**: every time it misclassifies a point, it nudges the
line toward getting that point right. This mistake-driven rule is the historical
ancestor of all neural networks.
"""),
        md(r"""
## 2. Concept (the slide)
- **Model:** $\hat y=\operatorname{sign}(\mathbf w^\top\mathbf x+b)$, labels in $\{-1,+1\}$.
- **Learning rule:** only update on a mistake.
- **Guarantee:** if the data is **linearly separable**, it converges in a finite
  number of updates (Novikoff). If not, it never settles → use the *pocket*
  (keep the best-so-far) or *averaged* variant.
- **Limit:** a single perceptron cannot represent XOR → you need a hidden layer.
"""),
        md(r"""
## 3. Math derivation

**Perceptron loss.** Define $\ell(\mathbf w)=\max(0,\,-y(\mathbf w^\top\mathbf x+b))$
— zero when correctly classified, otherwise the (negative) margin. Its
(sub)gradient on a misclassified point is $-y\mathbf x$. One step of SGD gives the
classic rule:

$$\boxed{\;\mathbf w\leftarrow\mathbf w+\eta\,y\,\mathbf x,\qquad b\leftarrow b+\eta\,y\;}\quad\text{(only when }y(\mathbf w^\top\mathbf x+b)\le 0).$$

**Why the update helps.** After the update the new margin on that point is
$y(\mathbf w+\eta y\mathbf x)^\top\mathbf x = y\mathbf w^\top\mathbf x+\eta\lVert\mathbf x\rVert^2$,
which is **larger** by $\eta\lVert\mathbf x\rVert^2$ — pushing toward correctness.

**Convergence (Novikoff).** If there is a unit $\mathbf w^\star$ with margin
$\gamma=\min_i y_i\mathbf w^{\star\top}\mathbf x_i>0$ and $\lVert\mathbf x_i\rVert\le R$,
the number of updates is bounded by $(R/\gamma)^2$ — independent of dimension and
dataset size. (Proof tracks $\mathbf w_t^\top\mathbf w^\star$ growing linearly while
$\lVert\mathbf w_t\rVert$ grows at most like $\sqrt t$.)

**XOR.** No single hyperplane separates $\{(0,0),(1,1)\}$ from $\{(0,1),(1,0)\}$;
the best any line does is 3/4. This concrete failure is exactly what a hidden
layer (the MLP) fixes.
"""),
        md("## 4. NumPy implementation (vanilla / pocket / averaged / multiclass)"),
        show(MOD, "PerceptronNumPy", "MulticlassPerceptron"),
        md("## 5. PyTorch implementation (single linear unit, perceptron loss)"),
        show(MOD, "PerceptronTorch"),
        md("## 6. Train — separable data, and the XOR failure"),
        run_demo(MOD),
        md("## 7. Visualization — the boundary and mistakes per epoch"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import perceptron as M

X, y = make_blobs(n_samples=200, centers=2, cluster_std=0.8, random_state=0)
p = M.PerceptronNumPy(n_epochs=15).fit(X, y)
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
zz = p.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].contourf(xx, yy, zz, alpha=.3, cmap="bwr")
ax[0].scatter(X[:,0], X[:,1], c=y, edgecolor="k", s=15, cmap="bwr")
ax[0].set_title("Learned linear boundary")
ax[1].plot(p.errors_, "o-"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mistakes")
ax[1].set_title("Mistakes per epoch")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways
- The perceptron = SGD on the perceptron loss; updates only on mistakes.
- Guaranteed convergence **iff** linearly separable; otherwise use pocket/averaged.
- It outputs a hard label, no probability → logistic regression softens it; SVM
  maximizes the margin instead of just *a* separating line.
- XOR ⇒ we need depth → **[MLP](../../02.dl/mlp/mlp.ipynb)**.
"""),
    ]

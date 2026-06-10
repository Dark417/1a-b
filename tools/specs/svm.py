from tools.nbreg import register, md, code, show, run_demo

MOD = "svm"


@register("svm", "ml/svm/svm.ipynb")
def build():
    return [
        md(r"""
# Support Vector Machines — the maximum-margin classifier

> Tutorial pair for [`svm.py`](svm.py).

## 1. Intuition
Many hyperplanes separate two classes; the SVM picks the one with the widest
"street" between them — the **maximum margin**. Only the points on the edge of
the street (the **support vectors**) matter; everything else could be deleted
without changing the boundary. When classes are not linearly separable, the
**kernel trick** measures similarity $K(x,x')$ in a richer feature space, bending
the boundary into curves while never forming those features explicitly.
"""),
        md(r"""
## 2. Concept (the slide)
- **Margin = $1/\lVert w\rVert$**: scale $w,b$ so the closest points satisfy
  $y_i(w^\top x_i+b)=1$. Maximizing the margin $\equiv$ minimizing $\lVert w\rVert^2$.
- **Soft margin**: allow slack $\xi_i\ge 0$ for noisy/overlapping data, penalized
  by $C$. Large $C$ = hard margin (few violations); small $C$ = wide, forgiving margin.
- **Dual**: rewriting with Lagrange multipliers $\alpha_i$ makes the data appear
  *only* as inner products $x_i^\top x_j$ — replace them with $K(x_i,x_j)$ for
  nonlinearity (linear, RBF, polynomial).
- **Support vectors**: the only points with $\alpha_i>0$.
"""),
        md(r"""
## 3. Math derivation — primal, dual, KKT, kernel trick

### Primal (soft margin)
$$\min_{w,b,\xi}\;\tfrac12\lVert w\rVert^2 + C\sum_i \xi_i
\quad\text{s.t.}\quad y_i(w^\top\phi(x_i)+b)\ge 1-\xi_i,\;\xi_i\ge 0.$$

### Lagrangian
Introduce multipliers $\alpha_i\ge0$ (margin) and $\mu_i\ge0$ (slack):
$$\mathcal L=\tfrac12\lVert w\rVert^2+C\sum_i\xi_i
-\sum_i\alpha_i\big(y_i(w^\top\phi(x_i)+b)-1+\xi_i\big)-\sum_i\mu_i\xi_i.$$
Stationarity:
$$\frac{\partial\mathcal L}{\partial w}=0\Rightarrow w=\sum_i\alpha_i y_i\phi(x_i),\quad
\frac{\partial\mathcal L}{\partial b}=0\Rightarrow \sum_i\alpha_i y_i=0,\quad
\frac{\partial\mathcal L}{\partial\xi_i}=0\Rightarrow \alpha_i=C-\mu_i.$$
Since $\mu_i\ge0$, the last gives $0\le\alpha_i\le C$.

### Dual
Substitute $w=\sum_i\alpha_i y_i\phi(x_i)$ back in. The $\phi$'s only ever appear
as inner products $\phi(x_i)^\top\phi(x_j)=K(x_i,x_j)$ — the **kernel trick**:
$$\boxed{\;\max_{\alpha}\;\sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j)
\;\;\text{s.t.}\;\;0\le\alpha_i\le C,\;\sum_i\alpha_i y_i=0.\;}$$

### KKT conditions (who is a support vector)
Complementary slackness $\alpha_i\big(y_i f(x_i)-1+\xi_i\big)=0$ and $\mu_i\xi_i=0$ give:
$$\alpha_i=0\Rightarrow y_i f(x_i)\ge 1\ (\text{outside}),\quad
0<\alpha_i<C\Rightarrow y_i f(x_i)=1\ (\text{on margin}),\quad
\alpha_i=C\Rightarrow y_i f(x_i)\le 1\ (\text{inside/violated}).$$
The decision function depends only on support vectors:
$$f(x)=\sum_{i:\alpha_i>0}\alpha_i y_i K(x_i,x)+b.$$

### Kernels
- **Linear** $K=x^\top z$, **Polynomial** $K=(\gamma\,x^\top z+c)^d$,
  **RBF** $K=\exp(-\gamma\lVert x-z\rVert^2)$ (infinite-dimensional $\phi$).

### Solving the dual: SMO
SMO does **coordinate ascent on two multipliers at a time** so the equality
constraint $\sum_i\alpha_i y_i=0$ stays satisfied: fix all but $\alpha_i,\alpha_j$,
solve that 1-D quadratic in closed form (optimum
$\alpha_j\!\leftarrow\!\alpha_j-\tfrac{y_j(E_i-E_j)}{\eta}$ with curvature
$\eta=2K_{ij}-K_{ii}-K_{jj}$), clip to the box $[L,H]$, repeat until KKT holds.

### Primal (PyTorch) view
Eliminating the constraints turns the primal into the **hinge-loss** objective
$$\min_{w,b}\;\tfrac12\lVert w\rVert^2+C\sum_i\max\!\big(0,\,1-y_i(w^\top x_i+b)\big),$$
which we minimize directly by (sub-)gradient descent via autograd.
"""),
        md("## 4. NumPy implementation — dual soft-margin SVM via simplified SMO"),
        show(MOD, "SVMNumPy"),
        md("## 5. PyTorch implementation — primal hinge loss (+ RBF random features)"),
        show(MOD, "SVMTorch"),
        md("## 6. Train — linear vs kernel SVM, dual vs primal"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — decision boundaries: linear vs RBF kernel

On the two-moons data a linear boundary cannot separate the classes, while the
RBF kernel bends the boundary around them. Support vectors are circled.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import svm as M

X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
y = np.where(y == 0, -1, 1).astype(float)
X = (X - X.mean(0)) / X.std(0)

xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 200),
                     np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (name, kw) in zip(ax, [("linear", dict(kernel="linear")),
                              ("RBF", dict(kernel="rbf", gamma=1.0))]):
    m = M.SVMNumPy(C=1.0, n_iters=100, **kw).fit(X, y)
    Z = m.decision_function(grid).reshape(xx.shape)
    a.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#ffd5d5", "#d5d5ff"], alpha=.6)
    a.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=1)
    a.scatter(X[:,0], X[:,1], c=y, cmap="bwr", s=14, edgecolors="k", linewidths=.3)
    a.scatter(m.sv_X[:,0], m.sv_X[:,1], s=120, facecolors="none", edgecolors="lime", linewidths=1.4)
    a.set_title(f"{name} SVM  (#SV={len(m.alpha)})")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- The model is defined entirely by its **support vectors**; the rest of the data
  is irrelevant once trained — that is the SVM's sparsity and robustness.
- **$C$** trades margin width against training violations; **$\gamma$** (RBF) sets
  the reach of each point — too large $\gamma$ overfits (islands around points).
- Always **standardize features**: RBF/poly kernels are distance/scale sensitive.
- The dual is $O(n^2)$ in memory (Gram matrix) — fine for thousands of points,
  not millions. For huge data prefer the linear primal (hinge SGD) or random
  features (as in the PyTorch class) to approximate the kernel.
- Hinge loss gives no calibrated probabilities; use Platt scaling if you need them.
"""),
    ]

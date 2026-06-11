from tools.nbreg import register, md, code, show, run_demo

MOD = "svr"


@register("svr", "01.ml/svm/svr.ipynb")
def build():
    return [
        md(r"""
# Support Vector Regression — fitting a tube, not a line

> Tutorial pair for [`svr.py`](svr.py).

## 1. Intuition
Ordinary least squares punishes *every* deviation, however tiny. SVR instead
draws a tube of half-width $\varepsilon$ around the function and says: **if a
point lands inside the tube, its error is zero — don't move for it.** Only points
that poke *outside* the tube exert force on the model. The result is a sparse,
robust regressor that ignores small noise and is steered only by the points that
matter (the support vectors). Nonlinearity comes from a kernel / feature map,
exactly as in classification SVMs.
"""),
        md(r"""
## 2. Concept (the slide)
- **$\varepsilon$-insensitive tube:** errors $\le\varepsilon$ cost nothing;
  beyond that the cost grows linearly (an L1-like, robust penalty).
- **Objective:** $\tfrac12\lVert w\rVert^2 + C\sum_i L_\varepsilon(y_i-f(x_i))$.
  $\lVert w\rVert^2$ keeps the function flat/simple; $C$ weights fit vs flatness.
- **Support vectors:** only points on or outside the tube ($|y_i-f(x_i)|\ge\varepsilon$).
- **Kernels:** replace $x$ by features $\phi(x)$; here we use **random Fourier
  features** to approximate the RBF kernel cheaply and explicitly.
"""),
        md(r"""
## 3. Math derivation — the $\varepsilon$-insensitive loss & its sub-gradient

### The loss
$$L_\varepsilon(r)=\max\big(0,\;|r|-\varepsilon\big),\qquad r=y-f(x).$$
A flat-bottomed valley: zero on $[-\varepsilon,\varepsilon]$, then slope $\pm1$.

### Primal objective
$$\min_{w,b}\;J(w,b)=\tfrac12\lVert w\rVert^2
 + C\sum_{i=1}^n \max\!\big(0,\;|y_i-(w^\top\phi(x_i)+b)|-\varepsilon\big).$$

### Sub-gradient
$L_\varepsilon$ is non-differentiable at the kinks, so we use a **sub-gradient**.
With residual $r_i=y_i-f(x_i)$,
$$\frac{\partial L_\varepsilon}{\partial r_i}=
\begin{cases}
0 & |r_i|\le\varepsilon \quad(\text{inside the tube})\\
-\operatorname{sign}(r_i) & |r_i|>\varepsilon \quad(\text{outside})
\end{cases}$$
Chaining through $f=w^\top\phi(x)+b$ (note $\partial r_i/\partial w=-\phi(x_i)$),
let $s_i=-\operatorname{sign}(r_i)\,\mathbf 1[|r_i|>\varepsilon]$. Then
$$\boxed{\;\nabla_w J = w + C\sum_i s_i\,\phi(x_i),\qquad
 \nabla_b J = C\sum_i s_i.\;}$$
Gradient descent on this is what `SVRNumPy` runs; PyTorch reproduces the same
sub-gradient via autograd on `clamp(|r|-eps, min=0)`.

### Dual / kernel view (for context)
Introducing multipliers $\alpha_i,\alpha_i^*$ for the two sides of the tube gives
the dual
$$\max -\tfrac12\sum_{i,j}(\alpha_i-\alpha_i^*)(\alpha_j-\alpha_j^*)K(x_i,x_j)
 -\varepsilon\sum_i(\alpha_i+\alpha_i^*)+\sum_i y_i(\alpha_i-\alpha_i^*)$$
subject to $\sum_i(\alpha_i-\alpha_i^*)=0$, $0\le\alpha_i,\alpha_i^*\le C$, with
$f(x)=\sum_i(\alpha_i-\alpha_i^*)K(x_i,x)+b$. KKT forces $\alpha_i=\alpha_i^*=0$
for points strictly inside the tube — that is the sparsity. We instead optimize
the **primal** directly and get RBF nonlinearity from random Fourier features:
$$\phi(x)=\sqrt{\tfrac{2}{D}}\cos(Wx+b_{\text{rf}}),\quad W\sim\mathcal N(0,2\gamma I)
 \;\Rightarrow\; \langle\phi(x),\phi(x')\rangle\approx e^{-\gamma\lVert x-x'\rVert^2}.$$
"""),
        md("## 4. NumPy implementation — primal SVR by sub-gradient descent"),
        show(MOD, "SVRNumPy"),
        md("## 5. PyTorch implementation — same objective via autograd"),
        show(MOD, "SVRTorch"),
        md("## 6. Train — linear vs RBF SVR, NumPy vs Torch"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — the $\varepsilon$-tube and the RBF fit

Left: a linear SVR with its $\pm\varepsilon$ tube (points inside cost nothing).
Right: on a nonlinear target the linear fit underfits while the RBF feature map
tracks the curve.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import svr as M

rng = np.random.default_rng(0)

# Left: linear SVR + epsilon tube
Xl = rng.uniform(-3, 3, size=(150, 1))
yl = 1.7 * Xl[:, 0] - 0.5 + rng.normal(scale=0.3, size=150)
lin = M.SVRNumPy(C=1.0, epsilon=0.4, kernel="linear", lr=0.05, n_iters=600).fit(Xl, yl)
xs = np.linspace(-3, 3, 200)[:, None]
f = lin.predict(xs)

# Right: nonlinear target, linear vs RBF
Xn = rng.uniform(-3, 3, size=(200, 1))
yn = np.sin(Xn[:, 0]) + 0.3 * Xn[:, 0] + rng.normal(scale=0.1, size=200)
Xn = (Xn - Xn.mean(0)) / Xn.std(0)
lin_n = M.SVRNumPy(C=1.0, epsilon=0.05, kernel="linear", lr=0.05, n_iters=600).fit(Xn, yn)
rbf_n = M.SVRNumPy(C=2.0, epsilon=0.05, kernel="rbf", n_rff=200, gamma=1.0, lr=0.1, n_iters=1000).fit(Xn, yn)
order = np.argsort(Xn[:, 0]); xo = Xn[order]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(Xl[:, 0], yl, s=12, alpha=.5)
ax[0].plot(xs[:, 0], f, "b-", label="SVR fit")
ax[0].plot(xs[:, 0], f + lin.epsilon, "b--", alpha=.6, label="$\\pm\\varepsilon$ tube")
ax[0].plot(xs[:, 0], f - lin.epsilon, "b--", alpha=.6)
ax[0].set_title("Linear SVR with $\\varepsilon$-tube"); ax[0].legend()

ax[1].scatter(Xn[:, 0], yn, s=12, alpha=.4)
ax[1].plot(xo[:, 0], lin_n.predict(xo), "g-", label="linear (underfits)")
ax[1].plot(xo[:, 0], rbf_n.predict(xo), "r-", label="RBF features")
ax[1].set_title("Nonlinear target: linear vs RBF SVR"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **$\varepsilon$** sets how much error you call "noise" (a dead-zone); larger
  $\varepsilon$ → sparser model, smoother, more bias. **$C$** trades flatness
  against fitting the out-of-tube points.
- SVR's L1-style penalty outside the tube makes it **robust to outliers**
  compared with squared-error regression.
- For nonlinear targets you need a kernel / feature map; the **random Fourier
  features** here approximate RBF in $O(nD)$ instead of forming an $n\times n$
  Gram matrix, so it scales.
- Standardize inputs (and often the target): the tube width $\varepsilon$ and RBF
  $\gamma$ are both scale-dependent.
- The sub-gradient is zero inside the tube — if $\varepsilon$ is too large the
  model gets *no* signal and stalls flat.
"""),
    ]

from tools.nbreg import register, md, code, show, run_demo

MOD = "linear_regression"


@register("linear_regression", "01.ml/linear-models/linear_regression.ipynb")
def build():
    return [
        md(r"""
# Linear Regression — from the math to PyTorch

> Tutorial pair for [`linear_regression.py`](linear_regression.py). Concept →
> derivation → NumPy → PyTorch → variants. Runs top-to-bottom on CPU.

## 1. Intuition

We have points $(x_i, y_i)$ and we believe $y$ depends *linearly* on $x$. Draw
the straight line (hyperplane in higher dimensions) that passes **as close as
possible** to all points, where "close" means *small squared vertical
distance*. That line is a model we can use to predict $y$ for a new $x$.

Everything downstream — logistic regression, neural nets — is "this, but with a
nonlinearity and more layers." So we derive it carefully once.
"""),
        md(r"""
## 2. Concept (the slide)

- **Model:** $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$.
- **Loss:** mean squared error
  $\;\mathcal{L}(\mathbf{w},b)=\dfrac{1}{2n}\sum_{i=1}^n(\hat{y}_i-y_i)^2.$
- **Fit:** choose $\mathbf{w},b$ to minimize $\mathcal{L}$ — either by a
  **closed-form** solution (set the gradient to zero) or by **gradient
  descent** (step downhill).
- **Why squared error?** It is the negative log-likelihood under Gaussian noise
  $y = \mathbf{w}^\top\mathbf{x}+b+\varepsilon,\ \varepsilon\sim\mathcal{N}(0,\sigma^2)$
  — so least squares = maximum likelihood.
"""),
        md(r"""
## 3. Math derivation

Stack the data: rows of $X\in\mathbb{R}^{n\times d}$ are samples, fold the bias
into $\mathbf{w}$ by appending a column of ones so $X\!\leftarrow\![X\,|\,\mathbf 1]$.
Then $\hat{\mathbf y}=X\mathbf{w}$ and

$$\mathcal{L}(\mathbf w)=\tfrac{1}{2n}\,\lVert X\mathbf w-\mathbf y\rVert_2^2 .$$

**Gradient.** Using $\nabla_{\mathbf w}\tfrac12\lVert X\mathbf w-\mathbf y\rVert^2 = X^\top(X\mathbf w-\mathbf y)$:

$$\nabla_{\mathbf w}\mathcal{L}=\frac{1}{n}X^\top(X\mathbf w-\mathbf y).$$

**Closed form (normal equations).** Set the gradient to zero:

$$X^\top X\,\mathbf w = X^\top\mathbf y \;\Longrightarrow\; \boxed{\;\mathbf w^\star=(X^\top X)^{-1}X^\top\mathbf y\;}$$

valid when $X^\top X$ is invertible. **Gradient descent** instead iterates
$\mathbf w \leftarrow \mathbf w-\eta\,\nabla_{\mathbf w}\mathcal{L}$, which is
what we need once $d$ is large or $X^\top X$ is singular.

### Regularization
Penalize large weights to fight overfitting / ill-conditioning:

$$\underbrace{(\lambda/2)\lVert\mathbf w\rVert_2^2}_{\text{Ridge (L2)}},\qquad
  \underbrace{\lambda\lVert\mathbf w\rVert_1}_{\text{Lasso (L1)}},\qquad
  \underbrace{\lambda\big[\alpha\lVert\mathbf w\rVert_1+\tfrac{1-\alpha}{2}\lVert\mathbf w\rVert_2^2\big]}_{\text{ElasticNet}}.$$

Ridge has the closed form $\mathbf w^\star=(X^\top X+\lambda I)^{-1}X^\top\mathbf y$
(don't penalize the bias). Lasso is non-differentiable at $0$ → we use the
subgradient $\lambda\,\mathrm{sign}(\mathbf w)$, and its $L_1$ corner is exactly
what drives weights to **zero** (feature selection).
"""),
        md("## 4. NumPy implementation\n\nThe real source from the module (so the notebook never drifts):"),
        show(MOD, "LinearRegressionNumPy"),
        md("## 5. PyTorch implementation\n\nSame model; autograd computes the gradient we derived above."),
        show(MOD, "LinearRegressionTorch"),
        md("## 6. Train & compare variants\n\nRun the packaged demo: closed-form vs gradient descent, OLS vs Ridge vs Lasso, NumPy vs PyTorch."),
        run_demo(MOD),
        md(r"""
## 7. Visualization — the fit and the loss curve
"""),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
import linear_regression as M

X, y, _ = M._toy(n=120, d=1)
mu, sd = X.mean(0), X.std(0) + 1e-12
Xz = (X - mu) / sd

model = M.LinearRegressionNumPy(lr=0.2, n_iters=2000).fit(Xz, y)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(X[:, 0], y, s=12, alpha=.6, label="data")
xs = np.linspace(X.min(), X.max(), 100)[:, None]
ax[0].plot(xs, model.predict((xs - mu) / sd), "r", lw=2, label="OLS fit")
ax[0].set_title("Fit"); ax[0].legend()
ax[1].plot(model.history); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("MSE")
ax[1].set_title("Gradient-descent loss curve")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls

- **Closed form vs gradient descent:** identical answer for OLS; GD is what
  scales and what every neural net uses.
- **Standardize features** before regularizing — otherwise $\lambda$ penalizes
  large-scale features unfairly.
- **Ridge** shrinks all weights smoothly; **Lasso** zeroes some (sparsity).
- The Gaussian-noise view explains *why* squared error — change the noise model
  and you get a different loss (e.g. Laplace → $L_1$ regression).

**Next:** swap the identity output for a sigmoid and the squared error for
cross-entropy → [logistic regression](../linear-models/logistic_regression.ipynb).
"""),
    ]

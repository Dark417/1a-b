from tools.nbreg import register, md, code, show, run_demo

MOD = "generalized_linear_models"


@register("generalized_linear_models", "01.ml/linear-models/generalized_linear_models.ipynb")
def build():
    return [
        md(r"""
# Generalized Linear Models — one framework, many regressions

> Tutorial pair for [`generalized_linear_models.py`](generalized_linear_models.py).

## 1. Intuition
Ordinary least squares quietly assumes the response is Gaussian with constant
variance. That is wrong for **counts** (non-negative integers, variance grows
with the mean) and for **positive skewed amounts** (waiting times, insurance
costs). GLMs keep the familiar linear predictor $\eta = Xw$ but feed it through a
**link function** so the model's mean lives in the right space, and they model
the noise with the matching **exponential-family** distribution. Pick the
family + link and you recover OLS, logistic regression, **Poisson** regression
(counts), **Gamma** regression (positive skew), and more — all fit by the *same*
algorithm.
"""),
        md(r"""
## 2. Concept (the slide)
- **Random component:** $y_i$ drawn from an exponential-family distribution with
  mean $\mu_i$ (Poisson, Gamma, Gaussian, Bernoulli, ...).
- **Systematic component:** a linear predictor $\eta_i = x_i^\top w$.
- **Link:** an invertible $g$ with $g(\mu_i) = \eta_i$, so $\mu_i = g^{-1}(\eta_i)$.
  We use the **log link** $g(\mu)=\log\mu \Rightarrow \mu=e^{\eta}$, which keeps
  $\mu>0$ — exactly what counts and amounts need.
- **Fit:** maximize the log-likelihood. Two routes: **IRLS / Fisher scoring**
  (Newton's method, a few iterations) and **gradient descent** on the NLL.
"""),
        md(r"""
## 3. Math derivation — exponential family, score, and IRLS

### Exponential family
Write the density in canonical form
$$p(y\mid\theta,\phi)=\exp\!\Big(\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi)\Big).$$
Standard identities give the mean and variance from $b$:
$$\mu=\mathbb E[y]=b'(\theta),\qquad \operatorname{Var}(y)=a(\phi)\,b''(\theta)=a(\phi)\,V(\mu),$$
where $V(\mu)=b''(\theta)$ is the **variance function**. The model couples $\mu$
to the linear predictor via the link $g(\mu)=\eta=x^\top w$.

### Score (gradient of the log-likelihood)
For one observation, by the chain rule
$$\frac{\partial \ell}{\partial w}
 =\underbrace{\frac{y-\mu}{a(\phi)\,V(\mu)}}_{\partial\ell/\partial\mu\;\cdot\;?}\,
   \frac{d\mu}{d\eta}\,x .$$
Stacking all $n$ rows, the **score** is
$$\nabla_w \ell = X^\top D\,(y-\mu),\qquad
  D=\operatorname{diag}\!\Big(\tfrac{1}{V(\mu_i)}\tfrac{d\mu_i}{d\eta_i}\Big)
  \;(\text{absorb }a(\phi)).$$

### Two families with the log link ($\mu=e^\eta\Rightarrow d\mu/d\eta=\mu$)

**Poisson.** $V(\mu)=\mu$, so $\tfrac{1}{V}\tfrac{d\mu}{d\eta}=\tfrac{1}{\mu}\mu=1$:
$$\ell=\sum_i\big(y_i\eta_i-e^{\eta_i}\big)+\text{const},\qquad
  \nabla_w(-\ell)=-X^\top(y-\mu).$$

**Gamma (unit shape).** $V(\mu)=\mu^2$, so $\tfrac{1}{V}\tfrac{d\mu}{d\eta}=\tfrac{1}{\mu^2}\mu=\tfrac1\mu$:
$$-\ell\;\propto\;\sum_i\Big(\frac{y_i}{\mu_i}+\eta_i\Big),\qquad
  \nabla_w(-\ell)=-X^\top\frac{y-\mu}{\mu}.$$

### IRLS = Fisher scoring = Newton with the expected Hessian
Newton's update is $w\leftarrow w-H^{-1}\nabla(-\ell)$. Replacing the Hessian by
its expectation (the **Fisher information** $\mathcal I = X^\top W X$ with working
weights $W_i=\big(\tfrac{d\mu_i}{d\eta_i}\big)^2/V(\mu_i)$) and rearranging turns
each step into a **weighted least squares** on a *working response*:
$$z_i=\eta_i+(y_i-\mu_i)\frac{d\eta_i}{d\mu_i},\qquad
  \boxed{\,w^{+}=\big(X^\top W X\big)^{-1}X^\top W z\,}.$$
For the log link, $d\eta/d\mu=1/\mu$, so $z=\eta+(y-\mu)/\mu$; the weights are
$W=\mu$ (Poisson) and $W=\mathbf 1$ (Gamma). Each iteration solves one weighted
normal-equations system — Newton-fast, typically a handful of steps.
"""),
        md("## 4. NumPy implementation — IRLS *and* gradient descent, by hand"),
        show(MOD, "GLMNumPy"),
        md("## 5. PyTorch implementation — minimize the same NLL with autograd"),
        show(MOD, "GLMTorch"),
        md("## 6. Train — IRLS vs GD vs Torch on synthetic Poisson and Gamma data"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — IRLS converges in a few Newton steps; GD crawls
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import generalized_linear_models as M

rng = np.random.default_rng(0)
Xp, yp, _ = M._make_poisson(rng)

irls = M.GLMNumPy(family="poisson", fit_method="irls", n_iters=50).fit(Xp, yp)
gd   = M.GLMNumPy(family="poisson", fit_method="gd", lr=0.2, n_iters=400).fit(Xp, yp)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(irls.history, "o-", label="IRLS (Newton)")
ax[0].plot(gd.history, "-", label="gradient descent")
ax[0].set_xlabel("iteration"); ax[0].set_ylabel("negative log-likelihood")
ax[0].set_title("Poisson NLL vs iteration"); ax[0].legend(); ax[0].grid(True, alpha=.3)

mu_hat = irls.predict(Xp)
order = np.argsort(mu_hat)
ax[1].scatter(mu_hat, yp, s=10, alpha=.4, label="observed count")
ax[1].plot(mu_hat[order], mu_hat[order], "r-", label="predicted mean $\\mu$")
ax[1].set_xlabel("predicted mean $\\mu=e^{Xw}$"); ax[1].set_ylabel("y")
ax[1].set_title("Poisson fit: y vs predicted mean"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Pick the family for the data**: counts → Poisson; positive skewed amounts →
  Gamma; binary → Bernoulli (logistic); real-valued → Gaussian (OLS). All are the
  *same* GLM machinery with a different variance function and link.
- The **log link** guarantees $\mu>0$ and makes coefficients multiplicative:
  $e^{w_j}$ is the factor by which the mean changes per unit of $x_j$.
- **IRLS is Fisher scoring**: each step is one weighted least squares; it usually
  converges in a few iterations, while plain GD needs hundreds and a tuned LR —
  but GD scales to huge data / streaming where forming $X^\top W X$ is costly.
- Poisson regression assumes mean = variance; real counts are often
  **overdispersed** (variance > mean) — then use Negative Binomial or a
  quasi-Poisson dispersion estimate.
"""),
    ]

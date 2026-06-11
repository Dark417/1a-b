from tools.nbreg import register, md, code, show, run_demo

MOD = "optimizers"


@register("optimizers", "02.dl/optimizers/optimizers.ipynb")
def build():
    return [
        md(r"""
# Gradient-Descent Optimizers — momentum, adaptivity, and Adam

> Tutorial pair for [`optimizers.py`](optimizers.py).

## 1. Intuition
All these optimizers do the same thing — step downhill — but they disagree on
*how big* and *in which direction* the step should be. **Momentum** remembers
where it was going and damps zig-zags. **Adaptive** methods (AdaGrad/RMSProp/Adam)
give each parameter its own learning rate based on the history of its gradients.
**Adam** combines both and adds **bias correction** to fix the cold-start.
"""),
        md(r"""
## 2. Concept (the slide)
On an **ill-conditioned** loss (long, narrow valley) plain SGD bounces across the
steep walls while crawling along the floor. The cures:
- **Momentum / Nesterov:** average gradients into a velocity → cancel the bounce,
  accelerate along the floor. Nesterov peeks at the *lookahead* point first.
- **AdaGrad:** divide by $\sqrt{\sum g^2}$ → big steps for rare directions; but the
  sum only grows, so the LR decays to zero.
- **RMSProp:** use an *exponential moving average* of $g^2$ instead → LR stays alive.
- **Adam:** momentum (1st moment) + RMSProp (2nd moment) + bias correction.
- **AdamW:** Adam but weight decay is applied to the weights directly, not folded
  into the gradient — better generalization.
"""),
        md(r"""
## 3. Math derivation — every update rule

Let $g_t=\nabla_\theta\mathcal L(\theta_t)$.

**SGD.** $\theta_{t+1}=\theta_t-\eta\,g_t.$

**Momentum (heavy ball).** Velocity = EMA of gradients:
$$v_t=\mu v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t.$$
Consistent directions accumulate ($\sum \mu^k$ amplifies by $\tfrac{1}{1-\mu}$);
oscillating ones cancel.

**Nesterov.** Evaluate the gradient at the *lookahead* $\tilde\theta=\theta_t-\eta\mu v_{t-1}$:
$$v_t=\mu v_{t-1}+\nabla\mathcal L(\tilde\theta),\qquad \theta_{t+1}=\theta_t-\eta v_t.$$
Looking ahead lets it brake *before* overshooting — a better-conditioned momentum.

**AdaGrad.** $G_t=G_{t-1}+g_t^2$ (elementwise), then
$$\theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{G_t}+\epsilon}\odot g_t.$$
The per-coordinate denominator $\sqrt{G_t}$ shrinks steps for frequently-large grads.

**RMSProp.** Replace the growing sum with a decaying average:
$$E_t=\rho E_{t-1}+(1-\rho)g_t^2,\qquad
  \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{E_t}+\epsilon}\odot g_t.$$

**Adam.** Track both moments and *correct their bias*:
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2.$$
Because $m_0=v_0=0$, unrolling gives $m_t=(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$.
Taking expectation under (approximately) stationary $g$:
$$\mathbb E[m_t]=(1-\beta_1^t)\,\mathbb E[g],\qquad \mathbb E[v_t]=(1-\beta_2^t)\,\mathbb E[g^2],$$
so the EMAs **under-estimate** the true moments by the factors $(1-\beta^t)$. Divide
them out:
$$\hat m_t=\frac{m_t}{1-\beta_1^{t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{t}},
  \qquad \theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$
We verify in the demo that this restores an update of size $\approx\eta$ from step 1.

**AdamW (decoupled weight decay).** Standard L2 puts $\lambda\theta$ *inside* $g_t$,
so it gets divided by $\sqrt{\hat v_t}$ along with everything else (coupling the
regularization to the adaptive scale). AdamW instead decays the weights directly:
$$\theta_{t+1}=\theta_t-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\theta_t\Big).$$
"""),
        md("## 4. NumPy implementation — each `step` is literally the update rule"),
        show(MOD, "SGD", "Momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam", "AdamW"),
        md("## 5. PyTorch implementation — same rules via `torch.optim` (validated)"),
        show(MOD, "torch_optimize"),
        md("## 6. Run — race on a quadratic, show bias correction, match torch"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — optimizer trajectories on a contour plot

The classic picture on the **Beale** function (a hard non-convex surface): watch
how momentum/Nesterov build speed and curve toward the minimum, while plain SGD
crawls. The black star is the global minimum $(3, 0.5)$.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import optimizers as M

start = (-2.0, 2.0)
# per-optimizer LRs that behave well on Beale's steep landscape
beale_cfg = {
    "sgd":      (M.SGD,      dict(lr=2e-4)),
    "momentum": (M.Momentum, dict(lr=1e-4, mu=0.9)),
    "nesterov": (M.Nesterov, dict(lr=1e-4, mu=0.9)),
    "rmsprop":  (M.RMSProp,  dict(lr=0.02, rho=0.9)),
    "adam":     (M.Adam,     dict(lr=0.3, b1=0.9, b2=0.999)),
}

# contour grid
xs = np.linspace(-3.5, 4.0, 300); ys = np.linspace(-1.5, 2.5, 300)
Xg, Yg = np.meshgrid(xs, ys)
Z = M.beale([Xg, Yg])

plt.figure(figsize=(8, 5.5))
plt.contour(Xg, Yg, np.log1p(Z), levels=30, cmap="viridis", alpha=.6)
for nm, (cls, kw) in beale_cfg.items():
    traj = M.optimize(cls, start, M.beale_grad, steps=4000, **kw)
    plt.plot(traj[:, 0], traj[:, 1], "-", lw=1.6, label=nm)
plt.scatter([3], [0.5], c="k", marker="*", s=180, zorder=5, label="min (3, 0.5)")
plt.scatter([start[0]], [start[1]], c="r", marker="o", s=40, zorder=5, label="start")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Optimizer trajectories on Beale (log-contours)")
plt.legend(loc="upper left", fontsize=8); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Momentum** is almost free and fixes SGD's zig-zag on ill-conditioned losses.
- **AdaGrad** decays its LR to zero (bad for long training); **RMSProp** fixes that
  with an EMA.
- **Adam** is the default for most deep nets; **bias correction** is essential for
  sane early steps (we measured it).
- **AdamW > Adam + L2** when you care about generalization — decouple the decay.
- Adaptive methods can converge to *worse* minima than well-tuned SGD+momentum on
  some vision tasks; there is no universally best optimizer.
- LR scheduling/warmup composes with any of these — see the transformer file and
  `06.training-techniques/README.md`.
"""),
    ]

from tools.nbreg import register, md, code, show, run_demo

MOD = "ddim"


@register("ddim", "03.generative-models/diffusion/ddim.ipynb")
def build():
    return [
        md(r"""
# DDIM — deterministic, accelerated diffusion sampling

> Tutorial pair for [`ddim.py`](ddim.py).

## 1. Intuition
A DDPM is slow to sample because it walks back through *every* one of the $T$
noising steps, one network call each. DDIM keeps the same trained noise-predictor
but changes the *sampler*: it defines a family of reverse processes that all share
the DDPM marginals, including a **deterministic** one that can take big jumps.
The result is sharp samples in 20-50 steps instead of 1000, and (at $\eta=0$) a
reproducible map from a fixed noise seed to a fixed sample.
"""),
        md(r"""
## 2. Concept (the slide)
- **Same training** as DDPM: regress $\epsilon_\theta(x_t,t)$ on the added noise.
- **Different sampling:** a *non-Markovian* reverse process with a free knob
  $\eta\in[0,1]$ controlling injected noise.
- $\eta=1$ recovers the DDPM ancestral sampler; $\eta=0$ is fully deterministic
  (an implicit ODE) and lets you **skip steps** cheaply.
- Pick any decreasing sub-sequence of timesteps $\tau_1>\tau_2>\dots$ to trade
  speed for quality.
"""),
        md(r"""
## 3. Math derivation — the non-Markovian reverse step

DDIM defines a family of inference processes indexed by $\sigma_t\ge0$ that all
keep the *same* forward marginals $q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$,
so a network trained for DDPM is valid unchanged.

**Predict $x_0$.** From $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$,
$$\hat x_0(x_t,t)=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.$$

**Step to an earlier time $s<t$.** The DDIM update is
$$\boxed{\,x_s=\sqrt{\bar\alpha_s}\,\hat x_0
 +\underbrace{\sqrt{1-\bar\alpha_s-\sigma_t^2}\;\epsilon_\theta(x_t,t)}_{\text{direction pointing to }x_t}
 +\sigma_t z\,},\qquad z\sim\mathcal N(0,I).$$
One checks that this preserves $q(x_s\mid x_0)$ for **any** choice of $\sigma_t$.

**The $\eta$ knob.** Set
$$\sigma_t=\eta\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}}\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_s}}.$$
- $\eta=1$ $\Rightarrow$ $\sigma_t$ equals the DDPM posterior std $\tilde\beta_t$
  — the **ancestral DDPM sampler**.
- $\eta=0$ $\Rightarrow$ $\sigma_t=0$, the noise term vanishes and the update is
  **deterministic**:
  $$x_s=\sqrt{\bar\alpha_s}\,\hat x_0+\sqrt{1-\bar\alpha_s}\,\epsilon_\theta(x_t,t).$$
  This is the Euler discretization of a *probability-flow ODE*; because there is
  no randomness, you can use a coarse subset of timesteps with little loss.

**Acceleration.** Choose a sub-sequence $\{\tau_i\}\subset\{1,\dots,T\}$ and apply
the update along it. With $S$ steps the cost is $S$ network calls instead of $T$.
"""),
        md("## 4. Model — locally defined DDPM-style eps-network"),
        show(MOD, "EpsMLP", "DDIM"),
        md("## 5. Training / sampling — DDPM loss + DDIM accelerated sampler"),
        show(MOD, "DDIM"),
        md("## 6. Train once, then sample with full vs few steps (eta = 0 / 1)"),
        run_demo(MOD),
        md("## 7. Visualization — sample quality vs number of steps"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import ddim as M

X, _ = make_moons(2000, noise=0.05, random_state=0)
X = ((X - X.mean(0)) / X.std(0)).astype("float32")
m = M.DDIM(data_dim=2, T=200).fit(X, epochs=400)

configs = [(200, 0.0, "det, 200 steps"),
           (20, 0.0, "det, 20 steps"),
           (10, 0.0, "det, 10 steps"),
           (20, 1.0, "stochastic, 20 steps")]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (steps, eta, title) in zip(axes, configs):
    s = m.sample(2000, steps=steps, eta=eta)
    ax.scatter(X[:, 0], X[:, 1], s=4, alpha=.2, color="gray")
    ax.scatter(s[:, 0], s[:, 1], s=4, alpha=.4, color="r")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- DDIM = **same network, smarter sampler**. No retraining needed.
- $\eta=0$ is deterministic and step-skippable; $\eta=1$ is DDPM.
- Too few steps (e.g. < 10 here) starts to blur fine structure — there is a
  speed/quality frontier.
- The deterministic flow is invertible (encode data $\to$ noise $\to$ data),
  which enables latent interpolation and is the bridge to the **score / SDE
  view** (next file): DDIM is the probability-flow ODE of that SDE.
"""),
    ]

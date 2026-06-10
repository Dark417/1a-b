from tools.nbreg import register, md, code, show, run_demo

MOD = "lsgan"


@register("lsgan", "generative-models/gan/lsgan.ipynb")
def build():
    return [
        md(r"""
# LSGAN — least-squares loss keeps the gradients flowing

> Tutorial pair for [`lsgan.py`](lsgan.py).

## 1. Intuition
With the usual sigmoid cross-entropy (BCE) loss, once a fake lands on the correct
side of the decision boundary the discriminator is "happy" and BCE **saturates**
— even if that fake is still far from the real data. **LSGAN** scores the
discriminator's raw output with a least-squares (L2) penalty, which keeps pushing
samples in proportion to how far they are from the target value. "Correct but far"
fakes still get a strong gradient pulling them toward the data.
"""),
        md(r"""
## 2. Concept (the slide)
- **Discriminator** outputs a single *unbounded* value (no sigmoid).
- Targets use the $a$–$b$–$c$ coding: D pushes real $\to b$, fake $\to a$; G pushes
  fake $\to c$. The standard choice is $a=0,\ b=1,\ c=1$.
- Replacing BCE with L2 means the loss never flattens, so the generator keeps
  receiving useful gradients.
"""),
        md(r"""
## 3. Math derivation — least squares and the Pearson $\chi^2$ divergence

LSGAN replaces the log-loss with squared error. With labels $a$ (fake), $b$
(real), $c$ (the value G wants its fakes to score), the objectives are
$$\min_D\;\tfrac12\,\mathbb E_{x\sim p_{\text{data}}}\big[(D(x)-b)^2\big]
 +\tfrac12\,\mathbb E_{z\sim p_z}\big[(D(G(z))-a)^2\big],$$
$$\min_G\;\tfrac12\,\mathbb E_{z\sim p_z}\big[(D(G(z))-c)^2\big].$$

**Optimal discriminator.** For fixed $G$, minimizing pointwise over $D(x)$ gives
$$D^\star(x)=\frac{b\,p_{\text{data}}(x)+a\,p_g(x)}{p_{\text{data}}(x)+p_g(x)}.$$

**What G minimizes.** Substitute $D^\star$ into G's objective. Choosing labels with
$b-c=1$ and $b-a=2$ (e.g. $a=-1,b=1,c=0$, or equivalently the demo's $a=0,b=1,c=1$
up to a shift) yields, after algebra,
$$2\,C(G)=\int\frac{\big((b-c)(p_{\text{data}}+p_g)-(b-a)p_g\big)^2}
 {p_{\text{data}}+p_g}\,dx
 =\chi^2_{\text{Pearson}}\big(p_{\text{data}}+p_g\,\big\Vert\,2p_g\big),$$
the **Pearson $\chi^2$ divergence** between $p_{\text{data}}+p_g$ and $2p_g$,
minimized iff $p_g=p_{\text{data}}$.

**Why this differs from vanilla GAN.** The vanilla GAN minimizes the
Jensen–Shannon divergence through a *sigmoid* output: for a confidently-classified
fake, $\partial_{\text{logit}}\,\mathrm{BCE}\to 0$, so its gradient **vanishes**.
The L2 loss has gradient $\propto (D(G(z))-c)$, which grows *linearly* with the
error and never saturates — distant fakes are penalized hardest, exactly where a
GAN most needs signal. The cost is the JS$\to\chi^2$ change of divergence.
"""),
        md("## 4. Generator / key component"),
        show(MOD, "Generator", "Discriminator"),
        md("## 5. Trainer / losses"),
        show(MOD, "LSGANTorch"),
        md("## 6. Train"),
        run_demo(MOD),
        md("## 7. Visualization"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import lsgan as M

real = M.make_ring(2000)
gan = M.LSGANTorch().fit(real, steps=1500, batch=128)
fake = gan.generate(1000)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(real[:, 0], real[:, 1], s=6, alpha=.3, label="real")
ax[0].scatter(fake[:, 0], fake[:, 1], s=6, alpha=.5, color="r", label="fake")
ax[0].set_title("Real vs generated (LSGAN)"); ax[0].legend(); ax[0].set_aspect("equal")
ax[1].plot(gan.d_hist, label="D (L2) loss", alpha=.7)
ax[1].plot(gan.g_hist, label="G (L2) loss", alpha=.7)
ax[1].set_xlabel("step"); ax[1].set_title("Least-squares losses"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Swapping BCE for L2 turns the JS-divergence game into a **Pearson $\chi^2$**
  game whose gradient grows with the error instead of saturating.
- Practical win: distant "already classified" fakes still get pulled toward the
  data, reducing vanishing-gradient stalls.
- Pitfalls: the unbounded D output can blow up without care; label coding matters
  ($b-c=1,\ b-a=2$ for the clean $\chi^2$ interpretation); still not immune to
  mode collapse.
"""),
    ]

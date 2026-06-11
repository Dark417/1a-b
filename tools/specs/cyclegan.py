from tools.nbreg import register, md, code, show, run_demo

MOD = "cyclegan"


@register("cyclegan", "03.generative-models/gan/cyclegan.ipynb")
def build():
    return [
        md(r"""
# CycleGAN — unpaired translation via cycle-consistency

> Tutorial pair for [`cyclegan.py`](cyclegan.py).

## 1. Intuition
We want to translate between two domains (horses<->zebras, photos<->paintings)
but have **no aligned pairs**. Adversarial loss alone is too weak: a generator
could send every input to one realistic target sample and still fool the critic.
The fix is **cycle-consistency** — translate $X\to Y\to X$ and you must get the
original back. That closed loop forces the mapping to preserve content, turning
an underconstrained problem into a well-posed one.
"""),
        md(r"""
## 2. Concept (the slide)
- **Two generators:** $G:X\to Y$ and $F:Y\to X$.
- **Two discriminators:** $D_Y$ scores $Y$-realism, $D_X$ scores $X$-realism.
- **Cycle-consistency:** $F(G(x))\approx x$ and $G(F(y))\approx y$ (L1).
- **Identity loss:** $G(y)\approx y$, $F(x)\approx x$ to preserve color/scale.
- We use the **least-squares** adversarial loss (LSGAN), which CycleGAN found
  more stable than BCE. Here both domains are tiny 2-D point clouds related by a
  fixed affine transform, so we can *measure* whether translation succeeded.
"""),
        md(r"""
## 3. Math derivation — adversarial + cycle + identity

**Adversarial (LSGAN form), e.g. for $G$ and $D_Y$:**
$$\mathcal L_{\text{GAN}}(G,D_Y)=\mathbb E_{y}\big[(D_Y(y)-1)^2\big]
 +\mathbb E_{x}\big[(D_Y(G(x)))^2\big],$$
with $G$ trying to push $D_Y(G(x))\to 1$. The same pair of terms applies to
$F$ and $D_X$. This alone only asks "does the output *look* like the target
domain?" — it says nothing about **which** input produced it.

**Cycle-consistency.** Add the constraint that round-trips are identity:
$$\mathcal L_{\text{cyc}}(G,F)=\mathbb E_{x}\big[\lVert F(G(x))-x\rVert_1\big]
 +\mathbb E_{y}\big[\lVert G(F(y))-y\rVert_1\big].$$
L1 (not L2) is used because it encourages sharp, less-blurry reconstructions.
This term is what prevents **mode collapse** to a single target: if many inputs
mapped to the same $y$, $F$ could not invert it to recover each distinct $x$.

**Identity.** Optionally regularize with
$$\mathcal L_{\text{id}}=\mathbb E_{y}\big[\lVert G(y)-y\rVert_1\big]
 +\mathbb E_{x}\big[\lVert F(x)-x\rVert_1\big],$$
which anchors color/scale (a sample already in the target domain should pass
through unchanged).

**Full objective.**
$$\min_{G,F}\max_{D_X,D_Y}\;\mathcal L_{\text{GAN}}(G,D_Y)+\mathcal L_{\text{GAN}}(F,D_X)
 +\lambda_{\text{cyc}}\,\mathcal L_{\text{cyc}}+\lambda_{\text{id}}\,\mathcal L_{\text{id}}.$$

**Why it differs from vanilla GAN.** A vanilla GAN matches one distribution from
noise. CycleGAN couples **two** GANs through an L1 cycle term, replacing the need
for paired supervision with a structural invertibility constraint.
"""),
        md("## 4. Generator / key component"),
        show(MOD, "Generator"),
        md("## 5. Trainer / losses"),
        show(MOD, "CycleGANTorch"),
        md("## 6. Train"),
        run_demo(MOD),
        md("## 7. Visualization"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import cyclegan as M

X = M.make_domain_x(800)
Y = M.make_domain_y(800)
gan = M.CycleGANTorch().fit(X, Y, steps=1500, batch=128)
fake_y = gan.generate(M.make_domain_x(800, seed=99))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(X[:, 0], X[:, 1], s=8, alpha=.4, label="domain X", color="C0")
ax[0].scatter(Y[:, 0], Y[:, 1], s=8, alpha=.4, label="domain Y", color="C1")
ax[0].scatter(fake_y[:, 0], fake_y[:, 1], s=8, alpha=.5, label="G(X) -> Y", color="C3")
ax[0].set_title("Unpaired domains and the learned translation")
ax[0].legend(); ax[0].set_aspect("equal")
ax[1].plot(gan.cyc_hist, alpha=.8)
ax[1].set_xlabel("step"); ax[1].set_ylabel("cycle L1")
ax[1].set_title("Cycle-consistency loss falling (X->Y->X recovers X)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Cycle-consistency replaces paired supervision: it is the L1 round-trip term
  that makes unpaired translation well-posed and resists mode collapse.
- Identity loss stabilizes color/scale; the LSGAN adversarial loss is steadier
  than BCE for this two-GAN system.
- Pitfalls: too-large $\lambda_{\text{cyc}}$ makes $G$ near-identity (no
  translation); the cycle constraint only enforces invertibility, not
  *correctness*, so geometry-changing tasks (e.g. shape changes) are hard.
"""),
    ]

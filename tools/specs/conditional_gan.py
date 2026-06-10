from tools.nbreg import register, md, code, show, run_demo

MOD = "conditional_gan"


@register("conditional_gan", "generative-models/gan/conditional_gan.ipynb")
def build():
    return [
        md(r"""
# Conditional GAN — generation you can steer with a label

> Tutorial pair for [`conditional_gan.py`](conditional_gan.py).

## 1. Intuition
A plain GAN samples from $p(x)$ with no say over *which* sample you get. A
**conditional GAN** hands both the generator and the discriminator an extra label
$y$, so the model learns $p(x\mid y)$ and you can ask it for "a sample of class 3".
The discriminator's job gets sharper too: not "is this real?" but "is this a real
example *of class $y$*?".
"""),
        md(r"""
## 2. Concept (the slide)
- A label $y$ is mapped to a learned **embedding** vector and concatenated into
  the inputs of both nets.
- **Generator** $G(z, y)$: noise + label embedding $\to$ a class-conditional sample.
- **Discriminator** $D(x, y)$: sample + label embedding $\to$ logit "real *and* of
  class $y$".
- The adversarial loss is the **non-saturating GAN loss**, but every term is now
  conditioned on $y$.
"""),
        md(r"""
## 3. Math derivation — conditioning the minimax game

Everything is conditioned on the label $y$. The value function becomes
$$\min_G\max_D\;V(D,G)=
 \mathbb E_{(x,y)\sim p_{\text{data}}}[\log D(x\mid y)]
 +\mathbb E_{y\sim p(y),\,z\sim p_z}[\log(1-D(G(z\mid y)\mid y))].$$

Fixing $G$ and maximizing pointwise *per label* gives the conditional optimal
discriminator
$$D^\star(x\mid y)=\frac{p_{\text{data}}(x\mid y)}{p_{\text{data}}(x\mid y)+p_g(x\mid y)},$$
and substituting it back shows the generator minimizes the **expected
Jensen–Shannon divergence over labels**
$$\mathbb E_{y\sim p(y)}\big[\mathrm{JSD}\big(p_{\text{data}}(\cdot\mid y)\,\Vert\,p_g(\cdot\mid y)\big)\big]
 \;-\;\log 4,$$
which is zero iff $p_g(x\mid y)=p_{\text{data}}(x\mid y)$ for every $y$ — i.e. the
generator matches the data distribution **class by class**.

**Why this differs from vanilla GAN.** The vanilla GAN matches only the marginal
$p(x)$, so a sample's class is uncontrollable and modes can be dropped. The
conditional version matches each conditional $p(x\mid y)$, which (a) gives explicit
control and (b) supplies the discriminator with the label as side information,
making "real but wrong class" detectable and discouraging class-level mode collapse.

In code, both nets use `nn.Embedding(n_classes, emb_dim)` and the trainer feeds
the *true* $y$ with real samples and a *sampled* $y$ with fakes.
"""),
        md("## 4. Generator / key component"),
        show(MOD, "Generator"),
        md("## 5. Trainer / losses"),
        show(MOD, "ConditionalGANTorch"),
        md("## 6. Train"),
        run_demo(MOD),
        md("## 7. Visualization"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import conditional_gan as M

k = 4
x, y = M.make_clusters(2000, k=k)
gan = M.ConditionalGANTorch(n_classes=k).fit(x, y, steps=1500, batch=128)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(x[:, 0], x[:, 1], s=6, alpha=.15, color="gray", label="real")
colors = ["C0", "C1", "C2", "C3"]
for c in range(k):
    s = gan.generate(200, label=c)
    ax.scatter(s[:, 0], s[:, 1], s=8, alpha=.6, color=colors[c], label=f"gen class {c}")
ax.set_title("Each requested label lands on its own cluster"); ax.legend()
ax.set_aspect("equal")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Conditioning turns $p(x)$ into $p(x\mid y)$: the generator becomes steerable and
  the discriminator matches each class distribution separately.
- Label embeddings concatenated to inputs are the simplest mechanism; richer
  schemes (projection discriminator, conditional BatchNorm) scale better.
- Pitfalls: if labels for fakes aren't sampled from the true $p(y)$, the model
  learns a skewed conditional; a too-strong D can still collapse *within* a class.
"""),
    ]

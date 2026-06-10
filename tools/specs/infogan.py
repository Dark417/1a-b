from tools.nbreg import register, md, code, show, run_demo

MOD = "infogan"


@register("infogan", "generative-models/gan/infogan.ipynb")
def build():
    return [
        md(r"""
# InfoGAN — disentangling the latent code by maximizing mutual information

> Tutorial pair for [`infogan.py`](infogan.py).

## 1. Intuition
A vanilla GAN's noise vector $z$ is **entangled**: no single coordinate has an
interpretable meaning. InfoGAN splits the input into incompressible noise $z$ and
a small set of *structured* codes $c$, then rewards the generator for making $c$
**recoverable** from the output. The result: one code coordinate ends up
controlling one human-meaningful factor of variation — here, the angle around a
2-D ring — for free, with no labels.
"""),
        md(r"""
## 2. Concept (the slide)
- **Generator** $G(z, c)$: noise $z$ plus a latent code $c$ produce a sample.
- **Mutual information** $I(c; G(z,c))$ measures how much the output tells us
  about $c$. Maximizing it forces $G$ to *use* $c$ rather than ignore it.
- $I$ is intractable (needs the true posterior $P(c\mid x)$), so we introduce an
  **auxiliary network $Q$** approximating it and maximize a **variational lower
  bound**. $Q$ shares its feature trunk with the discriminator $D$.
- For a **continuous** code, $Q$ outputs a Gaussian $(\mu, \log\sigma^2)$ and the
  bound reduces to a Gaussian log-likelihood of $c$.
"""),
        md(r"""
## 3. Math derivation — the variational MI lower bound

Vanilla GAN keeps its adversarial term unchanged:
$$\min_{G}\max_{D}\; V(D,G)=\mathbb E_{x\sim p_{\text{data}}}[\log D(x)]
 +\mathbb E_{z,c}[\log(1-D(G(z,c)))].$$
InfoGAN **adds** a mutual-information regularizer, turning the objective into
$$\min_{G,Q}\max_{D}\; V(D,G)-\lambda\, I(c; G(z,c)).$$

**Why a bound is needed.** By definition
$$I(c; G(z,c)) = H(c) - H(c\mid G(z,c))
 = H(c) + \mathbb E_{x\sim G(z,c)}\big[\mathbb E_{c'\sim P(c\mid x)}[\log P(c'\mid x)]\big].$$
The posterior $P(c\mid x)$ is unknown. Introduce an auxiliary distribution
$Q(c\mid x)$; using the non-negativity of the KL divergence
$\mathrm{KL}\!\big(P(\cdot\mid x)\,\Vert\,Q(\cdot\mid x)\big)\ge 0$ gives the
**variational lower bound**
$$I(c; G(z,c)) \ge H(c) + \mathbb E_{c\sim P(c),\,x\sim G(z,c)}\big[\log Q(c\mid x)\big]
 \;=\; L_I(G, Q).$$
Since $H(c)$ is constant (we fix the code prior), maximizing $L_I$ means
maximizing $\mathbb E[\log Q(c\mid x)]$ — i.e. training $Q$ to **reconstruct the
code** from the generated sample, jointly with $G$.

**Continuous code.** With $Q(c\mid x)=\mathcal N\big(c;\,\mu(x),\,\sigma^2(x)\big)$,
$$-\log Q(c\mid x) = \tfrac12\sum_i\Big(\log\sigma_i^2 + \tfrac{(c_i-\mu_i)^2}{\sigma_i^2}\Big) + \text{const},$$
which is exactly the Gaussian NLL minimized in the code. **Why it differs from
vanilla GAN:** the extra term ties the latent to the output so the code becomes
disentangled and interpretable, whereas a plain GAN's latent is arbitrary.
"""),
        md("## 4. Generator / key component"),
        show(MOD, "Generator", "DiscriminatorQ"),
        md("## 5. Trainer / losses"),
        show(MOD, "InfoGANTorch"),
        md("## 6. Train"),
        run_demo(MOD),
        md("## 7. Visualization"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import infogan as M

real = M.make_ring(2000)
gan = M.InfoGANTorch().fit(real, steps=1500, batch=128)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
# Left: sweeping the code should sweep position on the ring (disentanglement).
cmap = plt.get_cmap("viridis")
codes = np.linspace(-1, 1, 9)
ax[0].scatter(real[:, 0], real[:, 1], s=6, alpha=.12, color="gray", label="real")
for i, cval in enumerate(codes):
    s = gan.generate(120, code=cval)
    ax[0].scatter(s[:, 0], s[:, 1], s=10, alpha=.7, color=cmap(i / (len(codes) - 1)))
ax[0].set_title("Code -1 -> +1 sweeps the ring angle"); ax[0].set_aspect("equal")
ax[0].legend()
# Right: Q's NLL falling means the MI bound is tightening (code recoverable).
ax[1].plot(gan.mi_hist, alpha=.7)
ax[1].set_xlabel("step"); ax[1].set_ylabel("Q NLL")
ax[1].set_title("Variational MI bound tightening (lower = better)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- InfoGAN keeps the adversarial game but **adds** a variational MI term so a code
  coordinate controls a real factor of variation — disentanglement without labels.
- $Q$ sharing $D$'s trunk is nearly free; only the small posterior heads are extra.
- Pitfalls: $\lambda$ too large overpowers the adversarial loss (blurry samples);
  too small and $G$ ignores the code. Continuous codes need a sensible prior
  ($U(-1,1)$ here); discrete codes use a categorical $Q$ and cross-entropy instead.
"""),
    ]

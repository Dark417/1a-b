from tools.nbreg import register, md, code, show, run_demo

MOD = "autoencoder"


@register("autoencoder", "dl/autoencoder/autoencoder.ipynb")
def build():
    return [
        md(r"""
# Autoencoders — learning to compress by reconstructing

> Tutorial pair for [`autoencoder.py`](autoencoder.py).

## 1. Intuition
Force data through a narrow bottleneck and ask the network to rebuild it. To
reconstruct well through a small code, the encoder must discover the *structure*
that actually matters — a nonlinear generalization of PCA. The regularized
variants (denoising, sparse, contractive) shape *what* that code captures.
"""),
        md(r"""
## 2. Concept (the slide)
- **Encoder** $z=f(x)$ compresses; **decoder** $\hat x=g(z)$ rebuilds.
- **Objective:** minimize reconstruction error $\lVert \hat x-x\rVert^2$.
- **Undercomplete** ($\dim z<\dim x$) forces compression; if $f,g$ are linear and
  the loss is MSE, the optimum spans the **top principal components** (PCA).
- **Variants** add a regularizer to the code:
  - **Denoising:** corrupt the input, reconstruct the *clean* original → robust features.
  - **Sparse:** penalize the *average* activation of code units → few fire per input.
  - **Contractive:** penalize the encoder Jacobian → code insensitive to small input changes.
"""),
        md(r"""
## 3. Math derivation

**Reconstruction (MSE).** Per sample $\mathcal L_{rec}=\tfrac12\lVert\hat x-x\rVert^2$.
With a sigmoid decoder $\hat x=\sigma(z_2),\ z_2=hW_2+b_2$:
$$\frac{\partial\mathcal L_{rec}}{\partial z_2}=(\hat x-x)\odot\sigma'(z_2)
  =(\hat x-x)\odot\hat x\odot(1-\hat x),$$
then $\partial\mathcal L/\partial W_2=h^\top\,\partial\mathcal L/\partial z_2$ and the
error flows back to the hidden layer via $\partial\mathcal L/\partial h=(\partial\mathcal L/\partial z_2)W_2^\top$.

**Denoising.** Replace the input by a corrupted $\tilde x$ (here random masking),
but keep the *clean* $x$ as the target: minimize $\lVert g(f(\tilde x))-x\rVert^2$.
This stops the AE from learning the identity and forces it to fill in missing
structure.

**Sparse (KL).** Let $\hat\rho_j=\frac1N\sum_n h_j^{(n)}$ be a unit's average
activation; pick a small target $\rho$ (e.g. $0.05$) and add
$$\beta\sum_j \mathrm{KL}(\rho\,\|\,\hat\rho_j),\quad
  \mathrm{KL}=\rho\log\frac{\rho}{\hat\rho_j}+(1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}.$$
Its derivative w.r.t. the activation adds
$\beta\big(-\frac{\rho}{\hat\rho_j}+\frac{1-\rho}{1-\hat\rho_j}\big)$ to $\partial\mathcal L/\partial h_j$.
(An $L_1$ penalty $\lambda\sum_j|h_j|$ is a simpler alternative.)

**Contractive.** Penalize the Frobenius norm of the encoder Jacobian
$J_{jk}=\partial h_j/\partial x_k$. For a sigmoid encoder $h=\sigma(xW_1+b_1)$,
$J_{jk}=h_j(1-h_j)\,W_{1,kj}$, so
$$\lVert J\rVert_F^2=\sum_j \big(h_j(1-h_j)\big)^2\sum_k W_{1,kj}^2.$$
Adding $\lambda\lVert J\rVert_F^2$ makes the code change little when the input is
perturbed — directly the robustness we measure in the demo. The gradient (worked
out in the code) uses $\frac{d}{dz_1}\big(h(1-h)\big)^2 = 2\,s\,s(1-2h)$ with
$s=h(1-h)$.
"""),
        md("## 4. NumPy implementation — encoder/decoder + all four variant gradients"),
        show(MOD, "AutoencoderNumPy"),
        md("## 5. PyTorch implementation — deeper net; contractive via autograd"),
        show(MOD, "AutoencoderTorch"),
        md("## 6. Train on 8x8 digits — recon error, code sparsity, robustness"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — original vs reconstructed digits

Top row: original digits. Bottom row: the vanilla autoencoder's reconstruction
from a 16-dim code. The blur is the information lost through the bottleneck —
exactly what makes the code a useful compressed representation.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import autoencoder as M
from sklearn.datasets import load_digits

d = load_digits(); X = d.data/16.0
rng = np.random.default_rng(0); perm = rng.permutation(len(X)); X = X[perm]
Xtr, Xte = X[:1400], X[1400:]
ae = M.AutoencoderNumPy(64, 16, lr=0.5, mode="vanilla", seed=0).fit(Xtr, epochs=40)
recon = ae.forward(Xte[:8])

fig, axes = plt.subplots(2, 8, figsize=(11, 3))
for i in range(8):
    axes[0, i].imshow(Xte[i].reshape(8, 8), cmap="gray"); axes[0, i].axis("off")
    axes[1, i].imshow(recon[i].reshape(8, 8), cmap="gray"); axes[1, i].axis("off")
axes[0, 0].set_ylabel("orig"); axes[1, 0].set_ylabel("recon")
fig.suptitle("Autoencoder: originals (top) vs 16-dim reconstructions (bottom)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- A linear undercomplete AE recovers PCA; the nonlinearity is what buys you more.
- **Denoising** is a simple, powerful self-supervised objective (and the conceptual
  ancestor of masked-image / masked-language pretraining).
- **Sparse** codes are interpretable and disentangle factors; tune $\beta,\rho$ —
  too strong and reconstruction collapses (we see recon error rise).
- **Contractive** explicitly trades reconstruction for robustness of the code.
- Plain autoencoders are *not* generative — sampling a random code rarely decodes
  to a valid image. For that you need a probabilistic latent: see
  [`generative-models/vae/vae`](../../generative-models/vae/vae.ipynb).
"""),
    ]

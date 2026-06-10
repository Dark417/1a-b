from tools.nbreg import register, md, code, show, run_demo

MOD = "resnet"


@register("resnet", "dl/cnn/resnet.ipynb")
def build():
    return [
        md(r"""
# ResNet — residual connections beat the degradation problem

> Tutorial pair for [`resnet.py`](resnet.py).

## 1. Intuition
Stacking more layers *should* never hurt: the extra layers could just learn the
identity. In practice, plain very-deep CNNs do worse — training error goes **up**
past a certain depth. This is the **degradation problem**, and it is an
optimization failure (gradients struggle to reach early layers), not overfitting.
ResNet's fix is tiny: have each block learn a **residual** $F(x)=H(x)-x$ and emit
$x+F(x)$. The identity shortcut hands every layer a direct gradient highway, so
100+ layer nets train fine.
"""),
        md(r"""
## 2. Concept (the slide)
- **Residual block:** `out = x + F(x)`, where $F$ is a couple of conv-BN-ReLU
  layers. Learning a *perturbation* of the identity is easier than learning the
  whole mapping from scratch.
- **BasicBlock** (ResNet-18/34): two $3\times3$ convs + skip.
- **Bottleneck** (ResNet-50/101/152): $1\times1 \to 3\times3 \to 1\times1$, so the
  expensive $3\times3$ runs in a *reduced* channel space (cheap), then channels
  are restored.
- **Projection shortcut:** when a block changes spatial size or channel count, the
  skip uses a $1\times1$ conv so the shapes match before adding.
- **BatchNorm** keeps activations well-scaled (see `training-techniques/README.md`).
"""),
        md(r"""
## 3. Math — why the residual gradient never vanishes

**Forward.** A residual block computes
$$y = x + F(x;\,\mathcal W).$$

**Backward (the whole point).** Differentiate w.r.t. the input:
$$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}
   = I + F'(x).$$
The $+I$ is an **identity gradient path**. Now chain $L$ blocks; with upstream
loss $\mathcal L$ and intermediate activations $x_l$,
$$\frac{\partial \mathcal L}{\partial x_0}
  = \frac{\partial \mathcal L}{\partial x_L}\prod_{l=1}^{L}\big(I + F_l'(x_{l-1})\big).$$
Expanding the product, **one term is the bare identity** $\prod I = I$: the
gradient at the deepest layer reaches the input *undiminished*, plus correction
terms. Contrast the **plain** net, where
$$\frac{\partial \mathcal L}{\partial x_0}
  = \frac{\partial \mathcal L}{\partial x_L}\prod_{l=1}^{L} F_l'(x_{l-1}),$$
a pure product of Jacobians whose magnitude scales **geometrically** — if each
$\lVert F_l'\rVert<1$ it vanishes, if $>1$ it explodes. The $+I$ shortcut converts
that multiplicative chain into an additive one, which is the cure.

**Bottleneck parameter math.** A $3\times3$ conv with $C$ in/out channels costs
$9C^2$ weights. The bottleneck wraps it as $1\times1$ (reduce $C\to C/4$),
$3\times3$ (in $C/4$), $1\times1$ (restore $C/4\to C$):
$$C\cdot\tfrac{C}{4} + 9\cdot\big(\tfrac{C}{4}\big)^2 + \tfrac{C}{4}\cdot C
  = \tfrac{C^2}{4}+\tfrac{9C^2}{16}+\tfrac{C^2}{4}
  = \tfrac{17}{16}C^2 \approx 1.06\,C^2,$$
versus a *two* $3\times3$ BasicBlock at $18C^2$ — far cheaper for the same depth.
"""),
        md("## 4. Key building block — BasicBlock & Bottleneck"),
        show(MOD, "BasicBlock", "Bottleneck"),
        md("## 5. Full architecture (PyTorch) — a small ResNet from stages of blocks"),
        show(MOD, "TinyResNet"),
        md(r"""
## 6. Run — measure the gradient highway

The demo (a) backprops a unit gradient through a 50-layer tanh net in pure NumPy
with vs without skips, then (b) compares per-conv-layer gradient norms in a
16-conv PyTorch plain net vs its residual twin, then (c) builds the Basic and
Bottleneck ResNets and trains a few steps.
"""),
        run_demo(MOD),
        md(r"""
## 7. Visualization — per-layer gradient norm: plain vs residual

The money plot: in the deep plain net the gradient **collapses** toward the input
layer; the residual twin keeps it $O(1)$ all the way down, exactly as $I+F'$
predicts.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import torch
import resnet as M

torch.manual_seed(0); torch.set_num_threads(1)

# (a) NumPy 50-layer net: activation-gradient norm vs depth
plain_g, resid_g = M.residual_gradient_demo(n_layers=50, dim=16)

# (b) PyTorch 16-conv nets: per-conv-weight grad norm, input->output
x = torch.randn(8, 1, 8, 8); y = torch.randint(0, 10, (8,))
pn = M.layerwise_grad_norms(M.PlainNet(depth=16, width=8), x, y)
rn = M.layerwise_grad_norms(M.ResidualNet(depth=16, width=8), x, y)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(plain_g, "s-", label="plain")
ax[0].semilogy(resid_g, "o-", label="residual (+I)")
ax[0].set_title("NumPy 50-layer tanh net"); ax[0].set_xlabel("layer (0=input)")
ax[0].set_ylabel(r"$\||\partial L/\partial x\||$ (log)"); ax[0].legend(); ax[0].grid(True, alpha=.3)

ax[1].semilogy(range(1, len(pn)+1), pn, "s-", label="plain")
ax[1].semilogy(range(1, len(rn)+1), rn, "o-", label="residual")
ax[1].set_title("PyTorch 16-conv nets"); ax[1].set_xlabel("conv layer (1=input)")
ax[1].set_ylabel(r"$\||dW\||$ (log)"); ax[1].legend(); ax[1].grid(True, alpha=.3)
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Residual = additive, not multiplicative.** $\partial y/\partial x = I + F'$
  gives a guaranteed gradient path; that is *the* fix for the degradation problem.
- **Bottlenecks** make depth affordable: do the expensive $3\times3$ in a squeezed
  channel space.
- **Pitfall — shape mismatch:** if a block changes stride or channels you must
  project the shortcut (a $1\times1$ conv), or the addition will not align.
- **Pitfall — pre/post-activation:** the original block applies ReLU *after* the
  add; "Identity Mappings" (2016) moves BN/ReLU *before* the convs for an even
  cleaner identity path. Both appear in the literature.
- Related fixes for vanishing gradients live in `training-techniques/README.md`
  (init, BatchNorm, gated RNN memory).
"""),
    ]

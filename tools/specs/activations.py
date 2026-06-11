from tools.nbreg import register, md, code, show, run_demo

MOD = "activations"


@register("activations", "02.dl/mlp/activations.ipynb")
def build():
    return [
        md(r"""
# Activation Functions — the nonlinearity that makes depth matter

> Tutorial pair for [`activations.py`](activations.py).

## 1. Intuition
Stack only linear layers and you get... another linear layer: $W_2(W_1x)=(W_2W_1)x$.
The **activation** is the pointwise nonlinearity squeezed between affine maps; it is
the entire reason a deep network can represent something a single matrix cannot.
Its *derivative* is equally important: backprop multiplies by $f'$ at **every** layer,
so the shape of $f'$ decides whether gradients survive depth or **vanish**.
"""),
        md(r"""
## 2. Concept (the slide)
- A layer computes $a = f(z),\ z = aW + b$. The choice of $f$ trades off:
  - **Range / centering** (sigmoid is in $(0,1)$ and not zero-centred; tanh is in $(-1,1)$).
  - **Saturation** — where $f'\!\to 0$. A saturated unit passes *no* gradient.
  - **Smoothness** — ReLU has a kink; GELU/Swish are smooth, which helps optimization.
- **Saturating** (sigmoid, tanh): $f'$ bounded well below 1 over most of the input range.
- **Non-saturating** (ReLU family, GELU, Swish): $f'\approx 1$ on the active region.
- **Softmax** is the odd one out: vector-valued, used at the output to make a distribution;
  its derivative is a full **Jacobian**, not a scalar.
"""),
        md(r"""
## 3. Math derivation — each function and its derivative

**Sigmoid.** $\sigma(x)=\dfrac{1}{1+e^{-x}}$. Differentiate:
$$\sigma'(x)=\frac{e^{-x}}{(1+e^{-x})^2}=\sigma(x)\bigl(1-\sigma(x)\bigr)\le\tfrac14.$$
The bound $\tfrac14$ is the crux: each layer scales the gradient by at most $0.25$.

**Tanh.** $\tanh(x)=\dfrac{e^x-e^{-x}}{e^x+e^{-x}}$, and
$$\tanh'(x)=1-\tanh^2(x)\in[0,1],$$
with max $1$ at $x=0$ but $\to 0$ for $|x|$ large (still saturates, but zero-centred).

**ReLU.** $\mathrm{ReLU}(x)=\max(0,x)$, so
$$\mathrm{ReLU}'(x)=\begin{cases}1 & x>0\\ 0 & x<0\end{cases}$$
(undefined at $0$; we take $0$). No shrinkage on the active side — but a unit stuck at
$x<0$ is **dead** (zero gradient forever).

**LeakyReLU.** $f(x)=x$ for $x>0$, else $\alpha x$; derivative $1$ or $\alpha$. The small
$\alpha$ keeps negatives alive, fixing dead units.

**ELU.** $f(x)=x$ for $x>0$, else $\alpha(e^x-1)$. Then
$$f'(x)=\begin{cases}1 & x>0\\ \alpha e^x = f(x)+\alpha & x\le 0\end{cases}$$
— smooth, negative-saturating to $-\alpha$ (pushes mean activations toward 0).

**GELU** (tanh approximation, used in BERT/GPT). With
$u(x)=\sqrt{2/\pi}\,(x+0.044715x^3)$,
$$\mathrm{GELU}(x)=\tfrac12 x\,(1+\tanh u).$$
Product + chain rule, using $u'(x)=\sqrt{2/\pi}\,(1+3\cdot0.044715\,x^2)$ and
$\frac{d}{du}\tanh u = 1-\tanh^2u$:
$$\mathrm{GELU}'(x)=\tfrac12(1+\tanh u)+\tfrac12 x\,(1-\tanh^2u)\,u'(x).$$

**Swish / SiLU.** $f(x)=x\,\sigma(\beta x)$ ($\beta=1$ is SiLU). Product rule:
$$f'(x)=\sigma(\beta x)+\beta x\,\sigma(\beta x)\bigl(1-\sigma(\beta x)\bigr).$$
Note $f'$ can exceed $1$ — Swish/GELU are **non-monotone** and slightly self-gating.

**Softmax** (vector-valued). $s_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$. The Jacobian:
$$\frac{\partial s_i}{\partial z_j}=s_i(\delta_{ij}-s_j)\;\Longrightarrow\;
  J=\operatorname{diag}(s)-ss^\top.$$
For backprop you never form $J$; the **vector-Jacobian product** is
$$(g^\top J)_j = s_j\Bigl(g_j-\textstyle\sum_k g_k s_k\Bigr).$$

### The vanishing-gradient link
Backprop through $L$ layers multiplies $L$ copies of $f'$ (times weight factors). With
sigmoid, $\prod f' \le 0.25^L \to 0$ geometrically — early layers stop learning. ReLU/GELU/Swish
keep $f'\approx 1$, so the signal survives. We **measure** exactly this below.
"""),
        md("## 4. NumPy implementation — forward + analytic derivative for each"),
        show(MOD, "Sigmoid", "Tanh", "ReLU", "LeakyReLU", "ELU", "GELU", "Swish", "Softmax"),
        md("## 5. PyTorch implementation — same functions; derivatives via autograd"),
        show(MOD, "torch_forward_and_grad"),
        md("## 6. Run — verify every derivative & **measure saturation**"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — each activation and its derivative

Top row: the activations. Bottom row: their derivatives. Watch how sigmoid/tanh
derivatives collapse toward 0 away from the origin (saturation), while the ReLU
family / GELU / Swish keep their derivative near 1 on the active side.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import activations as M

x = np.linspace(-6, 6, 600)
acts = M.ELEMENTWISE
fig, axes = plt.subplots(2, len(acts), figsize=(2.1*len(acts), 4.6), sharex=True)
for j, (nm, act) in enumerate(acts.items()):
    axes[0, j].plot(x, act.forward(x), color="C0")
    axes[0, j].set_title(nm); axes[0, j].grid(True, alpha=.3); axes[0, j].axhline(0, color="k", lw=.5)
    axes[1, j].plot(x, act.df(x), color="C3")
    axes[1, j].grid(True, alpha=.3); axes[1, j].axhline(0, color="k", lw=.5)
axes[0, 0].set_ylabel("f(x)"); axes[1, 0].set_ylabel("f'(x)")
fig.suptitle("Activations (top) and their derivatives (bottom)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- The derivative $f'$ is what backprop propagates — its shape *is* the trainability story.
- **Sigmoid** saturates hard ($f'\le0.25$) and is not zero-centred → avoid in hidden layers.
- **ReLU** is cheap and non-saturating but can produce **dead units**; LeakyReLU/ELU fix that.
- **GELU/Swish** are smooth, non-monotone, and dominate modern Transformers/CNNs.
- **Softmax** belongs at the *output*; pair it with cross-entropy so the gradient
  simplifies to $a-y$ (see [`mlp.ipynb`](mlp.ipynb)).
- Vanishing gradients motivate ReLU/He init, normalization, and residuals — see
  `06.training-techniques/README.md`.
"""),
    ]

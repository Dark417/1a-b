from tools.nbreg import register, md, code, show, run_demo

MOD = "mlp"


@register("mlp", "dl/mlp/mlp.ipynb")
def build():
    return [
        md(r"""
# The MLP & Backpropagation — the heart of deep learning

> Tutorial pair for [`mlp.py`](mlp.py).

## 1. Intuition
Stack neurons in layers, put a nonlinearity between them, and you can
approximate essentially any function (universal approximation). The magic is
**backpropagation**: an efficient application of the chain rule that computes the
gradient of the loss w.r.t. *every* weight in one backward sweep.
"""),
        md(r"""
## 2. Concept (the slide)
- **Forward:** $a^{[l]}=f(z^{[l]}),\ z^{[l]}=a^{[l-1]}W^{[l]}+b^{[l]}$, with
  $a^{[0]}=x$ and a softmax output.
- **Loss:** cross-entropy.
- **Backward:** push the error signal $\delta$ from the output back to the input,
  reusing each layer's cached activations.
- **Init matters:** bad initialization → vanishing/exploding gradients before
  training even starts.
"""),
        md(r"""
## 3. Math derivation — backpropagation

Let the loss be $\mathcal L$ and define the **error signal**
$\delta^{[l]}=\partial\mathcal L/\partial z^{[l]}$.

**Output layer (softmax + cross-entropy).** A small miracle:
$$\delta^{[L]}=a^{[L]}-y_{\text{onehot}}.$$

**Hidden layers (chain rule).** Since $z^{[l+1]}=a^{[l]}W^{[l+1]}+b^{[l+1]}$ and
$a^{[l]}=f(z^{[l]})$,
$$\boxed{\;\delta^{[l]}=\big(\delta^{[l+1]}W^{[l+1]\top}\big)\odot f'(z^{[l]})\;}$$
— multiply the *next* layer's error by the transposed weights, then gate by the
local activation derivative.

**Parameter gradients.**
$$\frac{\partial\mathcal L}{\partial W^{[l]}}=\frac1n\,a^{[l-1]\top}\delta^{[l]},
  \qquad
  \frac{\partial\mathcal L}{\partial b^{[l]}}=\frac1n\sum_i\delta^{[l]}_i.$$

### Weight initialization (why it matters)
Each $\delta^{[l]}$ carries a factor $W^{[l+1]\top}$ *and* $f'(z^{[l]})$. Chain
$L$ of them and the magnitude scales like a **product**. To keep it $\approx 1$:
- **Xavier/Glorot** (tanh/sigmoid): $\operatorname{Var}(W)=\frac{2}{n_{in}+n_{out}}$.
- **He/Kaiming** (ReLU): $\operatorname{Var}(W)=\frac{2}{n_{in}}$.

Sigmoid saturates ($f'\le 0.25$), so its gradients **shrink** every layer →
*vanishing gradients*. ReLU has $f'\in\{0,1\}$, and with He init the signal
survives. We **measure exactly this** below.
"""),
        md("## 4. NumPy implementation — forward + backprop by hand"),
        show(MOD, "MLPNumPy"),
        md("## 5. PyTorch implementation — `autograd` reproduces the same gradients"),
        show(MOD, "MLPTorch"),
        md("## 6. Train (moons) and **measure vanishing gradients** in a deep net"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — per-layer gradient magnitude: sigmoid vs ReLU

This is the money plot: in a 6-layer net, sigmoid gradients decay by orders of
magnitude toward the input layer; ReLU+He stay flat.
"""),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import mlp as M

X, y = make_classification(n_samples=400, n_features=20, n_informative=10,
                           n_classes=3, random_state=0)
X = (X - X.mean(0)) / X.std(0)
deep = [20, 64, 64, 64, 64, 64, 3]

plt.figure(figsize=(7,4))
for act, init, style in [("sigmoid","xavier","o-"), ("relu","he","s-")]:
    net = M.MLPNumPy(deep, activation=act, init=init)
    net.forward(X); net.backward(y)
    plt.semilogy(range(1, len(net.grad_norms)+1), net.grad_norms, style, label=f"{act}+{init}")
plt.xlabel("layer (1=input … 6=output)"); plt.ylabel("||dW||  (log scale)")
plt.title("Vanishing gradients: sigmoid vs ReLU+He"); plt.legend(); plt.grid(True, alpha=.3)
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways
- Backprop = chain rule + caching; the softmax+CE output gives $\delta=a-y$.
- **Initialization is not a detail** — it decides whether deep nets train at all.
- Vanishing gradients motivate ReLU/He, BatchNorm, and residual connections
  (see `training-techniques/README.md`).

**Next:** add weight sharing over space → [CNNs](../cnn/cnn.ipynb); over time →
[RNNs](../rnn/rnn.ipynb).
"""),
    ]

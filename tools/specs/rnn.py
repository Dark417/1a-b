from tools.nbreg import register, md, code, show, run_demo

MOD = "rnn"


@register("rnn", "02.dl/rnn/rnn.ipynb")
def build():
    return [
        md(r"""
# Recurrent Neural Networks & BPTT — and the vanishing-gradient problem

> Tutorial pair for [`rnn.py`](rnn.py).

## 1. Intuition
To handle sequences, keep a hidden "memory" vector and update it at every step
with the **same** weights. Unrolled in time it's a very deep net that shares
parameters — and that depth is exactly why long-range learning is hard.
"""),
        md(r"""
## 2. Concept (the slide)
- **Recurrence:** $h_t=\tanh(x_tW_{xh}+h_{t-1}W_{hh}+b_h)$.
- **Read-out (many-to-one):** classify from the last state $h_T$.
- **Training:** Backpropagation Through Time (BPTT) = backprop on the unrolled graph.
- **Problem:** gradients to early steps **vanish** (or **explode**) because they
  are a long *product* of Jacobians.
"""),
        md(r"""
## 3. Math derivation — BPTT and why gradients vanish

Unroll $T$ steps. The loss depends on $h_t$ through all later states, so

$$\frac{\partial\mathcal L}{\partial h_t}
 =\frac{\partial\mathcal L}{\partial h_T}\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}},
 \qquad
 \frac{\partial h_k}{\partial h_{k-1}}=\operatorname{diag}\!\big(1-h_k^2\big)\,W_{hh}^{\top}.$$

The product of $T-t$ such Jacobians governs everything. Bound its norm by
$\big(\gamma\,\lVert W_{hh}\rVert\big)^{T-t}$ where $\gamma=\max|\tanh'|\le 1$:

$$\lVert W_{hh}\rVert<1/\gamma \Rightarrow \text{**vanishing** (geometric decay)},\qquad
  \lVert W_{hh}\rVert>1/\gamma \Rightarrow \text{**exploding**}.$$

So information from $t=0$ reaches the loss multiplied by a factor that is
exponentially small (or large) in the sequence length — early steps barely learn.

**Gradient clipping** (the exploding fix). Rescale the whole gradient when its
norm exceeds $\tau$:
$$g\leftarrow g\cdot\min\!\Big(1,\frac{\tau}{\lVert g\rVert}\Big).$$
It caps magnitude without changing direction. (Vanishing needs an architectural
fix → the LSTM, next file.)
"""),
        md("## 4. NumPy implementation — BPTT with per-step gradient tracking & clipping"),
        show(MOD, "RNNNumPy"),
        md("## 5. PyTorch implementation (`nn.RNN` + `clip_grad_norm_`)"),
        show(MOD, "RNNTorch"),
        md("## 6. Train a long-memory task, then **measure the vanishing decay**"),
        run_demo(MOD),
        md("## 7. Visualization — gradient norm vs distance back in time"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
import rnn as M

X, y = M.make_memory_task(n=1, T=30)
net = M.RNNNumPy(2, 16, 2)
net.forward(X[0]); net.backward(y[0])
norms = np.array(net.bptt_norms)            # index 0 = t=0 (distant)

plt.figure(figsize=(7,4))
plt.semilogy(range(len(norms)), norms, "o-")
plt.xlabel("time step t (0 = distant past, T = recent)")
plt.ylabel("||∂L/∂h_t||  (log)")
plt.title("BPTT: gradient to distant steps vanishes")
plt.grid(True, alpha=.3); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways
- BPTT is just backprop on the time-unrolled graph with **shared** weights.
- The repeated Jacobian product → exponential vanishing/exploding in $T$.
- **Clip** to tame explosion; change the **architecture** to beat vanishing.

**Next:** the [LSTM](lstm.ipynb) adds a gated, *additive* memory path so
gradients survive across many steps.
"""),
    ]

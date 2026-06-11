from tools.nbreg import register, md, code, show, run_demo

MOD = "lstm"


@register("lstm", "02.dl/rnn/lstm.ipynb")
def build():
    return [
        md(r"""
# LSTM (& GRU) — gated memory that beats vanishing gradients

> Tutorial pair for [`lstm.py`](lstm.py). Read [`rnn.ipynb`](rnn.ipynb) first.

## 1. Intuition
A vanilla RNN overwrites its memory every step, so old information (and its
gradient) decays. The LSTM adds a **cell state** that it can *carry unchanged*
and edits with learned **gates** — a "conveyor belt" for information across time.
"""),
        md(r"""
## 2. Concept (the slide)
Three gates decide, each step, what to **forget**, what to **write**, and what to
**output**:
- forget $f_t$, input $i_t$, output $o_t$ (all sigmoids in $[0,1]$),
- candidate content $g_t=\tanh(\cdot)$,
- cell update $c_t=f_t\odot c_{t-1}+i_t\odot g_t$ (**additive!**),
- hidden $h_t=o_t\odot\tanh(c_t)$.
The **GRU** merges these into two gates — cheaper, often as good.
"""),
        md(r"""
## 3. Math derivation — the constant error carousel

With $z=[x_t,h_{t-1}]$:
$$f_t=\sigma(W_fz),\;\; i_t=\sigma(W_iz),\;\; g_t=\tanh(W_gz),\;\; o_t=\sigma(W_oz),$$
$$c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t).$$

**Why gradients survive.** The gradient along the cell-state path is

$$\frac{\partial c_t}{\partial c_{t-1}}=\operatorname{diag}(f_t).$$

Unlike the RNN's $\operatorname{diag}(1-h^2)W_{hh}^\top$ (which shrinks), this is a
**direct, gateable multiplier**. When the forget gate stays open ($f_t\approx1$),
the product $\prod_k f_k\approx1$ — the *constant error carousel* — so gradients
flow across hundreds of steps without vanishing. We initialize the forget-gate
bias to $+1$ so the cell **remembers by default** early in training.

**Backprop.** The cell gradient accumulates both the readout path and the future:
$$\delta c_t=\delta h_t\odot o_t\odot(1-\tanh^2 c_t)+\delta c_{t+1}\odot f_{t+1}.$$
Each gate's pre-activation gradient is then standard sigmoid/tanh backprop (coded
in full in the module).
"""),
        md("## 4. NumPy implementation — LSTM cell + BPTT by hand"),
        show(MOD, "LSTMNumPy"),
        md("## 5. PyTorch implementation — LSTM and GRU"),
        show(MOD, "RecurrentTorch"),
        md("## 6. Train the long-memory task (where the vanilla RNN struggled)"),
        run_demo(MOD),
        md("## 7. Visualization — gates over time on one sequence"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
import lstm as M

seqs, labels = M.make_memory_task(n=200, T=25)
net = M.LSTMNumPy(2, 24, 2, lr=0.1).fit(seqs[:160], labels[:160], epochs=30)
net.forward(seqs[0])
F = np.array([c[1].mean() for c in net.cache])   # mean forget-gate activation
I = np.array([c[2].mean() for c in net.cache])   # mean input-gate activation

plt.figure(figsize=(7,4))
plt.plot(F, label="forget gate ⟨f_t⟩"); plt.plot(I, label="input gate ⟨i_t⟩")
plt.axvline(0, ls="--", c="k", alpha=.4, label="signal at t=0")
plt.xlabel("time step"); plt.ylabel("mean gate activation"); plt.ylim(0,1)
plt.title("LSTM learns to hold memory (forget≈1) after reading the signal")
plt.legend(); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways
- The **additive** cell update + gates create a gradient highway → long memory.
- Forget-gate bias $=1$ is a useful default.
- **GRU** ≈ LSTM with fewer parameters; try both.
- Attention (next major topic) drops recurrence entirely for direct,
  constant-path access to every step → [Transformers](../../05.transformers/architectures/transformer.ipynb).
"""),
    ]

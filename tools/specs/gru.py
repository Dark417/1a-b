from tools.nbreg import register, md, code, show, run_demo

MOD = "gru"


@register("gru", "dl/rnn/gru.ipynb")
def build():
    return [
        md(r"""
# GRU — the Gated Recurrent Unit

> Tutorial pair for [`gru.py`](gru.py). Read [`rnn.ipynb`](rnn.ipynb) and
> [`lstm.ipynb`](lstm.ipynb) first.

## 1. Intuition
The LSTM beats vanishing gradients with a separate cell state and **three** gates.
The **GRU** (Cho et al., 2014) asks: can we get the same long-memory behaviour
more cheaply? It keeps a **single** hidden state and just **two** gates — a
*reset* gate and an *update* gate. The update gate linearly blends the old state
with a fresh candidate, so when it decides to "carry", the state (and its
gradient) passes through almost untouched.
"""),
        md(r"""
## 2. Concept (the slide)
At each step (with $z=[x_t,h_{t-1}]$):
- **reset gate** $r_t=\sigma(\cdot)$ — how much of the past to *mix into* the
  candidate (when $r_t\approx0$ the candidate ignores history),
- **update gate** $u_t=\sigma(\cdot)$ — how much of the past state to *keep*,
- **candidate** $n_t=\tanh(\cdot)$ — the proposed new content (sees $r_t\odot h_{t-1}$),
- **state** $h_t=(1-u_t)\odot n_t + u_t\odot h_{t-1}$ — a **convex combination**.

No separate cell state, no output gate. Fewer parameters than the LSTM (2 gates
vs 3), often comparable accuracy.
"""),
        md(r"""
## 3. Math derivation — why the update gate is a gradient highway

**Equations.** With $z_t=[x_t,h_{t-1}]$ and candidate input $\tilde z_t=[x_t,\,r_t\odot h_{t-1}]$:
$$
r_t=\sigma(W_r z_t+b_r),\qquad
u_t=\sigma(W_u z_t+b_u),
$$
$$
n_t=\tanh(W_n \tilde z_t + b_n),\qquad
h_t=(1-u_t)\odot n_t + u_t\odot h_{t-1}.
$$

**The carry path.** Differentiate the state update w.r.t. the previous state. Two
terms appear — a *direct* copy term and an *indirect* term through the candidate:
$$
\frac{\partial h_t}{\partial h_{t-1}}
=\underbrace{\operatorname{diag}(u_t)}_{\text{direct carry}}
+\;\underbrace{\operatorname{diag}(1-u_t)\,\frac{\partial n_t}{\partial h_{t-1}}}_{\text{through candidate}} .
$$
When the update gate stays open ($u_t\approx1$) the second term is suppressed by
$1-u_t\approx0$ and the Jacobian collapses to
$$
\frac{\partial h_t}{\partial h_{t-1}}\approx\operatorname{diag}(u_t)\approx I,
$$
so over $k$ steps the product is $\prod u\approx 1$ — the gradient flows back
**unattenuated**. This is the same idea as the LSTM's *constant error carousel*
($\partial c_t/\partial c_{t-1}=\operatorname{diag}(f_t)$), and contrasts sharply
with the vanilla RNN's $\operatorname{diag}(1-h^2)W_{hh}^\top$, whose repeated
multiplication shrinks (vanishing) or grows (exploding) geometrically.

**Comparison at a glance:**

| cell | carry Jacobian | states / gates |
|---|---|---|
| RNN | $\operatorname{diag}(1-h_t^2)\,W_{hh}^\top$ (decays) | 1 / 0 |
| LSTM | $\operatorname{diag}(f_t)$ on $c_t$ | 2 / 3 |
| GRU | $\operatorname{diag}(u_t)$ on $h_t$ | 1 / 2 |

**Backprop (BPTT) sketch.** Let $\delta h_t=\partial\mathcal L/\partial h_t$. From
$h_t=(1-u_t)n_t+u_t h_{t-1}$:
$$
\delta n_t=\delta h_t\odot(1-u_t),\quad
\delta u_t=\delta h_t\odot(h_{t-1}-n_t),\quad
\delta h_{t-1}\mathrel{+}=\delta h_t\odot u_t .
$$
Push $\delta n_t$ through $\tanh$ ($\times(1-n_t^2)$) into $W_n$ and into
$r_t\odot h_{t-1}$ (adding $\delta r_t = (\cdot)\odot h_{t-1}$ and another
contribution $(\cdot)\odot r_t$ to $\delta h_{t-1}$); push $\delta u_t,\delta r_t$
through their sigmoids ($\times g(1-g)$) into $W_u,W_r$ and back into
$\delta h_{t-1}$. The module codes every one of these terms by hand.
"""),
        md("## 4. NumPy implementation — GRU cell + BPTT by hand"),
        show(MOD, "GRUNumPy"),
        md("## 5. PyTorch implementation — idiomatic `nn.GRU` + clipping"),
        show(MOD, "GRUTorch"),
        md("## 6. Train / run — the long-memory toy task"),
        run_demo(MOD),
        md("## 7. Visualization — the update gate holds the memory open"),
        code(r"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gru as M

M.seed_everything()
seqs, labels = M.make_memory_task(n=160, T=12)
net = M.GRUNumPy(2, 16, 2, lr=0.2).fit(seqs[:120], labels[:120], epochs=40)

# Run one sequence through and read the gates out of the cache.
net.forward(seqs[0])
U = np.array([c[3].mean() for c in net.cache])   # mean update-gate activation
R = np.array([c[2].mean() for c in net.cache])   # mean reset-gate activation

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(net.history)
ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
ax1.set_title("GRU training loss")
ax1.grid(True, alpha=.3)

ax2.plot(U, "o-", label="update gate ⟨u_t⟩  (carry)")
ax2.plot(R, "s-", label="reset gate ⟨r_t⟩")
ax2.axvline(0, ls="--", c="k", alpha=.4, label="signal at t=0")
ax2.set_xlabel("time step"); ax2.set_ylabel("mean gate activation")
ax2.set_ylim(0, 1)
ax2.set_title("u_t≈1 ⇒ state copied forward (gradient highway)")
ax2.legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Convex-combination update** $h_t=(1-u_t)n_t+u_t h_{t-1}$ gives the carry
  Jacobian $\operatorname{diag}(u_t)$ — a gradient highway, like the LSTM cell but
  with one state and two gates.
- **GRU vs LSTM:** fewer parameters and often as accurate; the LSTM's extra
  output gate can help on some tasks. Try both.
- **Pitfall — the reset gate placement:** the candidate sees $r_t\odot h_{t-1}$
  *before* the matmul $W_n$, not after. Getting this wrong silently breaks the
  reset gate (and the gradient `d(r⊙h)/dr = h_{t-1}` term in BPTT).
- **Pitfall — gate saturation:** if $u_t$ saturates near 0 early, the model
  behaves like a plain RNN and gradients vanish; clipping + good init help.
- **Next:** drop recurrence entirely and attend directly to every step →
  [Transformers](../../transformers/architectures/transformer.ipynb).
"""),
    ]

from tools.nbreg import register, md, code, show, run_demo

MOD = "char_rnn"


@register("char_rnn", "04.nlp/language-models/char_rnn.ipynb")
def build():
    return [
        md(r"""
# Character-level RNN — generating text one character at a time

> Tutorial pair for [`char_rnn.py`](char_rnn.py). The recurrent machinery is the
> vanilla RNN from [`rnn.ipynb`](../../02.dl/rnn/rnn.ipynb).

## 1. Intuition
Forget words — let the model read and write **single characters**. The vocabulary
is just the distinct characters in the text, yet a small RNN, fed one character
at a time, learns to spell words, insert spaces, and even mimic grammar, purely
from the statistics of which character follows which. This is Karpathy's famous
*min-char-rnn*: tiny, from scratch, and unreasonably effective.
"""),
        md(r"""
## 2. Concept (the slide)
- **One-hot characters** in, **softmax over characters** out.
- A hidden state $h_t$ is a running summary of everything read so far; it is
  updated each step and used to predict the next character.
- Train with **backpropagation through time** (BPTT) over short chunks, carrying
  the hidden state across chunks.
- RNN gradients **explode** along the recurrence, so we **clip** them.
- **Generate** by feeding the model's own output back in, sampling with a
  *temperature* that trades coherence for diversity.
"""),
        md(r"""
## 3. Math derivation

**Char-level factorization.** A string $c_1\dots c_T$ factorizes exactly like any
sequence,
$$P(c_1,\dots,c_T)=\prod_{t=1}^{T} P(c_t\mid c_1,\dots,c_{t-1}),$$
and the RNN models each conditional through its hidden state.

**Forward (vanilla RNN).** With $x_t$ the one-hot of $c_t$:
$$h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b_h),\qquad
y_t=W_{hy}h_t+b_y,\qquad p_t=\operatorname{softmax}(y_t).$$
The whole history is squeezed into the fixed-size vector $h_{t-1}$.

**Loss.** Sum the per-step cross-entropies:
$$\mathcal L=\sum_{t=1}^{T}-\log p_t[c_{t+1}].$$

**Backprop through time.** The output gradient is the usual softmax-minus-onehot,
$\delta y_t=p_t-\mathbf 1_{c_{t+1}}$. The hidden gradient receives two streams —
from the output at time $t$ and from the *future* through the recurrence:
$$\delta h_t=W_{hy}^\top \delta y_t+W_{hh}^\top \delta a_{t+1},\qquad
\delta a_t=(1-h_t^{2})\odot \delta h_t,$$
where $\delta a_t$ is the gradient *before* the $\tanh$. Accumulating over time,
$$\delta W_{hh}=\sum_t \delta a_t\,h_{t-1}^\top,\quad
\delta W_{xh}=\sum_t \delta a_t\,x_t^\top,\quad
\delta W_{hy}=\sum_t \delta y_t\,h_t^\top.$$

**Why clip?** Unrolling multiplies by $W_{hh}^\top$ once per step; if its spectral
norm $>1$ the gradient blows up exponentially. We **clip** each gradient to
$[-5,5]$ (and use Adagrad's adaptive step) to keep training stable — the canonical
RNN training technique (see [`rnn.ipynb`](../../02.dl/rnn/rnn.ipynb)). LSTMs replace
this fragile path with a gated additive one
([`lstm.ipynb`](../../02.dl/rnn/lstm.ipynb)).

**Temperature sampling.** To generate, draw $c\sim\operatorname{softmax}(y/T)$.
Low $T$ -> peaky, repetitive, "safe"; $T=1$ -> the model's own distribution;
high $T$ -> adventurous and error-prone. Each sampled character is fed back as the
next input.
"""),
        md("## 4. NumPy implementation — vanilla RNN with manual BPTT (min-char-rnn)"),
        show(MOD, "CharRNNNumPy"),
        md("## 5. PyTorch implementation — an LSTM char-LM"),
        show(MOD, "CharLSTMTorch"),
        md("## 6. Train / run — fit a short string, then generate at two temperatures"),
        run_demo(MOD),
        md("## 7. Visualization — the BPTT loss curve as the RNN learns to spell"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import char_rnn as M

text = M.toy_text()
rnn = M.CharRNNNumPy(hidden=100, seq_len=25, lr=0.1).fit(text, epochs=40)

plt.figure(figsize=(7, 4))
plt.plot(rnn.history)
plt.xlabel("epoch"); plt.ylabel("mean cross-entropy / char (nats)")
plt.title("char-RNN BPTT loss: from random characters to readable words")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- A char-RNN learns spelling and structure with **no word knowledge at all** —
  the hidden state is doing all the bookkeeping.
- **Gradient clipping is not optional** for vanilla RNNs; without it BPTT
  diverges. Carrying the hidden state across chunks (truncated BPTT) lets the
  model see beyond one chunk.
- **Temperature** is the single most important generation knob; tune it per use.
- Vanilla RNNs still struggle with **long-range** dependencies (vanishing
  gradients) — switch to the [LSTM](../../02.dl/rnn/lstm.ipynb), or to
  [Transformers](../../05.transformers/architectures/transformer.ipynb) for parallel,
  long-context modeling. On a tiny repeated string the LSTM can memorize it (loss
  near 0), which is expected here.
"""),
    ]

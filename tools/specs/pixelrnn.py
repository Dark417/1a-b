from tools.nbreg import register, md, code, show, run_demo

MOD = "pixelrnn"


@register("pixelrnn", "generative-models/autoregressive/pixelrnn.ipynb")
def build():
    return [
        md(r"""
# PixelRNN (Row LSTM) — autoregressive images with recurrence

> Tutorial pair for [`pixelrnn.py`](pixelrnn.py).

## 1. Intuition
PixelRNN shares PixelCNN's promise -- model an image pixel by pixel in raster
order and get an exact likelihood -- but uses an **LSTM** to summarize the context
instead of a fixed-size masked convolution. The recurrence carries information
down the rows, so in principle a pixel can depend on *all* the rows above it, not
just a small convolutional window. This file implements the simpler **Row LSTM**
variant: scan top to bottom, one LSTM step per row.
"""),
        md(r"""
## 2. Concept (the slide)
- Same chain-rule factorization over pixels as PixelCNN.
- **Row LSTM:** an LSTM whose hidden state moves down the image one row at a time.
- The input to row $i$ is a *causal 1-D convolution* of the row above, and the
  recurrence uses row $i-1$ to predict row $i$ (a one-row shift) -- so a pixel
  never sees its own row, preserving causality.
- Train by maximum likelihood with teacher forcing; sample pixel by pixel.
"""),
        md(r"""
## 3. Math derivation — factorization & the row recurrence

**Autoregressive factorization** (identical chain rule as PixelCNN). With the
$N=H\cdot W$ pixels in raster order,
$$\boxed{\,p(x)=\prod_{i=1}^{N}p\big(x_i\mid x_{<i}\big)\,},
\qquad p(x_i\mid x_{<i})=\sigma(\ell_i)^{x_i}(1-\sigma(\ell_i))^{1-x_i}.$$
The exact NLL is again a sum of per-pixel binary cross-entropies.

**Row LSTM recurrence.** Group the conditionals by row. Let $r_i\in\{0,1\}^W$ be
row $i$. We compute a state that carries the context of rows $<i$ downward:
$$\text{input}_i=\mathrm{Conv1d}^{\text{causal}}_{\text{cols}}\big(\phi(r_{i-1})\big),
\qquad (h_i,c_i)=\mathrm{LSTM}\big(\text{input}_i,(h_{i-1},c_{i-1})\big),$$
$$\ell_i=\mathrm{readout}(h_i)\in\mathbb R^{W}.$$
Two design points guarantee the autoregressive property:
1. **One-row shift:** row $i$ is predicted from $r_{i-1}$ and the recurrent state,
   which only ever absorbed rows $\le i-1$. So $x_i$ never enters its own
   prediction.
2. **Causal 1-D conv** along the width: with kernel size $k$ the convolution mixes
   a few neighbouring columns of the row above, giving each pixel a *triangular*
   receptive field that widens with depth in the rows -- the LSTM extends it
   unboundedly far up the image.

The recurrence is exactly an RNN unrolled over $H$ steps, so training uses
**backpropagation through time** (with gradient clipping, since BPTT can explode --
the same issue showcased in `dl/rnn/`).

**Why recurrence over convolution?** A masked conv has a bounded receptive field;
the LSTM's state can in principle propagate dependencies across the *entire* image
height, at the cost of an inherently sequential forward pass.
"""),
        md("## 4. Model — the Row-LSTM PixelRNN"),
        show(MOD, "RowLSTM"),
        md("## 5. Training / sampling — NLL loss (BPTT + clipping) and raster sampler"),
        show(MOD, "RowLSTM"),
        md("## 6. Train & sample on binarized 8×8 digits"),
        run_demo(MOD),
        md("## 7. Visualization — training NLL and generated digits"),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import pixelrnn as M

X = load_digits().data.reshape(-1, 1, 8, 8) / 16.0
X = (X > 0.3).astype("float32")
m = M.RowLSTM(size=8, hidden=64).fit(X, epochs=35)
samples = m.sample(16)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(m.history)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("NLL (nats/image)")
ax[0].set_title("Row-LSTM training likelihood")

# show samples in an inset grid
ax[1].axis("off"); ax[1].set_title("Generated digits (pixel by pixel)")
grid = np.zeros((2 * 8, 8 * 8))
for idx in range(16):
    r, c = divmod(idx, 8)
    grid[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = samples[idx, 0]
ax[1].imshow(grid, cmap="gray")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- PixelRNN gives an **exact** likelihood, like PixelCNN, but with an unbounded
  vertical receptive field via the LSTM state.
- The **one-row shift + causal conv** is what keeps it autoregressive; get the
  shift wrong and the model trivially copies the target.
- Recurrence makes both training and sampling sequential -- slower than PixelCNN,
  which is why the field moved toward convolutional / transformer autoregressive
  models. **BPTT** here needs gradient clipping (cf. `dl/rnn/`).
- The full paper's **Diagonal BiLSTM** removes the Row-LSTM's blind spot; we
  implement only the simpler Row LSTM.
"""),
    ]

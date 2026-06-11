from tools.nbreg import register, md, code, show, run_demo

MOD = "gpt"


@register("gpt", "05.transformers/architectures/gpt.ipynb")
def build():
    return [
        md(r"""
# GPT — decoder-only causal Transformer

> Tutorial pair for [`gpt.py`](gpt.py). Read
> [`transformer.ipynb`](transformer.ipynb) and
> [`attention.ipynb`](../attention/attention.ipynb) first.

## 1. Intuition
Keep only the *decoder* stack of the Transformer and remove cross-attention. Each
block does **masked self-attention** (a token may look only at itself and earlier
tokens) followed by a position-wise MLP. Reading left to right, the model predicts
the **next token** at every position. Train it on next-token prediction, then let
it write text by sampling one token at a time and feeding it back in.
"""),
        md(r"""
## 2. Concept (the slide)
- **Causal mask:** the only structural difference from a BERT encoder — block the
  upper triangle so position $i$ cannot see $j>i$.
- **Pre-norm blocks:** LayerNorm *before* each sublayer, residual around it
  (GPT-2 style) — stable to train deep.
- **Weight tying:** the output projection reuses the input embedding matrix.
- **Generation:** autoregressive sampling with a **temperature** (flatten/sharpen
  the distribution) and optional **top-k** truncation.
"""),
        md(r"""
## 3. Math derivation

### 3.1 Autoregressive factorization
Any joint distribution over a sequence factorizes exactly by the chain rule:
$$p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T} p_\theta(x_t \mid x_1,\dots,x_{t-1}).$$
GPT parameterizes each conditional $p_\theta(x_t\mid x_{<t})$ with the same network.
Training minimizes the negative log-likelihood, i.e. the mean next-token
cross-entropy:
$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).$$

### 3.2 Causal masking makes the factorization valid
The conditional for position $t$ must depend **only** on $x_{<t}$. Self-attention
would otherwise let $x_t$ peek at the answer. We enforce this with an additive
mask on the scores before softmax:
$$\tilde S_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}} + M_{ij},\qquad
  M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases},$$
so $\operatorname{softmax}(\tilde S)_{ij}=0$ for $j>i$. With this mask a single
forward pass computes **all** $T$ conditionals in parallel (teacher forcing) — each
row $i$ of the output is a valid next-token distribution for prefix $x_{\le i}$.

### 3.3 Sampling with temperature and top-k
At inference we draw $x_t \sim p_\theta(\cdot\mid x_{<t})$. With logits $z$ and
**temperature** $\tau$,
$$p_i=\frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}.$$
$\tau\to0$ approaches greedy argmax (sharp, repetitive); $\tau>1$ flattens
(more random). **Top-k** first keeps only the $k$ largest logits (sets the rest to
$-\infty$) to avoid sampling the long tail of unlikely tokens.
"""),
        md("## 4. Key component — causal self-attention (the mask) + decoder block"),
        show(MOD, "causal_mask_numpy", "CausalSelfAttention", "DecoderBlock"),
        md("## 5. Full model — GPT (token+pos embeddings, blocks, tied head, generate)"),
        show(MOD, "GPT"),
        md("## 6. Train / run — learn a cyclic pattern, then generate from it"),
        run_demo(MOD),
        md("## 7. Visualization — the causal attention mask & a training loss curve"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, torch, matplotlib.pyplot as plt
import gpt as M

torch.manual_seed(M.SEED); np.random.seed(M.SEED); torch.set_num_threads(1)

# (a) the causal mask: lower-triangular "who can attend to whom"
L = 10
mask = M.causal_mask_numpy(L)

# (b) a short training run to draw a loss curve
period, V, Lc = 5, 5, 12
data = torch.tensor(M.make_pattern_data(256, Lc + 1, period))
x, y = data[:, :-1], data[:, 1:]
model = M.GPT(V, d_model=48, n_heads=4, d_ff=96, n_layers=2, max_len=Lc)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
lossfn = torch.nn.CrossEntropyLoss()
losses = []
for _ in range(120):
    logits = model(x)
    loss = lossfn(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].imshow(mask, cmap="Greys", vmin=0, vmax=1)
ax[0].set_title("Causal mask (1 = may attend)")
ax[0].set_xlabel("key pos j"); ax[0].set_ylabel("query pos i")
ax[1].plot(losses); ax[1].set_xlabel("step"); ax[1].set_ylabel("cross-entropy")
ax[1].set_title("Next-token loss going down")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- GPT = a Transformer **decoder stack with causal masking**; the chain rule turns
  next-token prediction into full-sequence generation.
- The causal mask is the single line that separates a generator (GPT) from a
  bidirectional encoder (BERT).
- Pitfalls: forgetting the mask leaks the future and gives deceptively low train
  loss but useless generation; very low temperature causes repetition loops;
  positions beyond `max_len` are undefined for learned positional embeddings (crop
  the context, as `generate` does).
- Next: **[BERT](bert.ipynb)** removes the mask for bidirectional pretraining;
  **[T5](t5.ipynb)** keeps the full encoder-decoder.
"""),
    ]

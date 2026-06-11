from tools.nbreg import register, md, code, show, run_demo

MOD = "transformer"


@register("transformer", "05.transformers/architectures/transformer.ipynb")
def build():
    return [
        md(r"""
# The Transformer — *Attention Is All You Need*

> Tutorial pair for [`transformer.py`](transformer.py). Read
> [`attention.ipynb`](../attention/attention.ipynb) first.

## 1. Intuition
Take multi-head attention, wrap it with a position-wise MLP, a residual
connection and LayerNorm, and stack the block. Add positional encodings (since
attention is order-agnostic) and you get a fully-parallel sequence model that
trains far better than RNNs — the backbone of BERT, GPT, T5, and ViT.
"""),
        md(r"""
## 2. Concept (the slide)
- **Encoder block:** self-attention → FFN, each with **residual + LayerNorm**.
- **Decoder block:** *masked* self-attention → **cross-attention** to the encoder
  → FFN.
- **Positional encoding** injects order; **causal mask** enforces autoregression.
- **Teacher forcing** during training; **greedy/beam** decoding at inference.
- **noam LR schedule:** warm up, then decay — critical for stable training.
"""),
        md(r"""
## 3. Math derivation — the pieces

**Positional encoding.** Attention is permutation-equivariant, so we add position
info:
$$PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d}\big),\quad
  PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d}\big).$$
Because $\sin(a{+}b),\cos(a{+}b)$ are linear in $\sin a,\cos a$, a fixed relative
shift $k$ acts as a **linear map** on $PE$ — attention can learn relative offsets.

**Residual + LayerNorm (pre-norm).** Each sublayer computes
$x \leftarrow x + \text{Sublayer}(\text{LN}(x))$. The residual gives the gradient a
direct $+1$ path (the vanishing-gradient fix from ResNet), and
$\text{LN}(x)=\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ (statistics over
the feature dim, **per token**) keeps activations well-scaled regardless of batch.

**Position-wise FFN.** $\text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$ applied identically
at every position — the per-token "compute" between attention mixing steps.

**Masked self-attention.** In the decoder, a lower-triangular mask blocks future
positions so the model is a valid autoregressive factorization
$p(y)=\prod_t p(y_t\mid y_{<t},x)$. Training uses **teacher forcing**: feed the
true $y_{<t}$ in parallel and predict all $y_t$ at once.

**noam learning-rate schedule.**
$$\text{lr}(t)=d_{\text{model}}^{-1/2}\cdot\min\!\big(t^{-1/2},\, t\cdot t_{\text{warmup}}^{-3/2}\big).$$
It rises linearly for `warmup` steps then decays as $t^{-1/2}$. Warmup prevents
huge early updates from destabilizing the freshly-initialized attention; the decay
anneals to a good minimum. (See `06.training-techniques/README.md`.)
"""),
        md("## 4. NumPy — positional encoding (the closed-form bit)"),
        show(MOD, "positional_encoding"),
        md("## 5. PyTorch — attention, encoder/decoder blocks, full model"),
        show(MOD, "EncoderLayer", "DecoderLayer", "Transformer"),
        md("## 6. Train a seq2seq **reverse** task (teacher forcing + noam) and decode"),
        code("# ~30–60s on CPU: trains 1000 steps then greedily decodes one example.\n"
             "import transformer as M\nM.demo()"),
        md("## 7. Visualization — positional encoding & the noam schedule"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
import transformer as M

pe = M.positional_encoding(80, 64)
steps = np.arange(1, 4000)
lr = [M.noam_lr(s, 64, warmup=300) for s in steps]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
im = ax[0].imshow(pe.T, aspect="auto", cmap="RdBu")
ax[0].set_xlabel("position"); ax[0].set_ylabel("dimension")
ax[0].set_title("Sinusoidal positional encoding"); fig.colorbar(im, ax=ax[0])
ax[1].plot(steps, lr); ax[1].axvline(300, ls="--", c="r", label="warmup end")
ax[1].set_xlabel("step"); ax[1].set_ylabel("learning rate")
ax[1].set_title("noam schedule (warmup → decay)"); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & what's next
- The Transformer = attention + FFN + **residual + LayerNorm**, stacked, with
  positional encodings and careful **LR warmup**.
- Masking turns the decoder into an autoregressive model; teacher forcing makes
  training parallel.
- Specializations (next files in `architectures/`, see `MAP.md`):
  - **BERT** = encoder-only, masked-LM pretraining (bidirectional).
  - **GPT** = decoder-only, causal LM (generation).
  - **ViT** = patches-as-tokens for images.
"""),
    ]

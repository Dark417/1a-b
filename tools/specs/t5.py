from tools.nbreg import register, md, code, show, run_demo

MOD = "t5"


@register("t5", "transformers/architectures/t5.ipynb")
def build():
    return [
        md(r"""
# T5 — the text-to-text encoder-decoder Transformer

> Tutorial pair for [`t5.py`](t5.py). Read
> [`transformer.ipynb`](transformer.ipynb) first.

## 1. Intuition
T5's thesis: **cast every NLP task as "text in -> text out."** Translation,
summarization, classification, even regression all become string-to-string
problems handled by *one* model with *one* loss. Architecturally it is the
original Transformer — a **bidirectional encoder** reads the input, a **causal
decoder** writes the output while cross-attending to the encoder. It is pretrained
by **span corruption**: drop random spans from the input and make the decoder
regenerate them.
"""),
        md(r"""
## 2. Concept (the slide)
- **Encoder–decoder:** encoder = bidirectional self-attention; decoder = causal
  self-attention **+ cross-attention** to the encoder memory.
- **Shared vocabulary & embedding** for input and output (tied to the output
  projection).
- **Span corruption** pretraining uses **sentinel tokens** (`<extra_id_0>`, …):
  each dropped span is one sentinel in the input, and the target is the sentinels
  followed by their missing content.
- **Teacher forcing + causal mask** in the decoder, just like the base
  Transformer.
"""),
        md(r"""
## 3. Math derivation

### 3.1 The text-to-text framing
Every task is a conditional sequence model
$$p_\theta(y_{1:T_y}\mid x_{1:T_x})=\prod_{t=1}^{T_y} p_\theta(y_t\mid y_{<t},\,x_{1:T_x}),$$
trained with the same token-level cross-entropy
$\mathcal{L}=-\sum_t \log p_\theta(y_t\mid y_{<t},x)$ regardless of whether $y$ is a
translation, a class name ("positive"), or a number rendered as text. The encoder
provides the conditioning $x$ via **cross-attention**: in the decoder,
$$\text{CrossAttn}(Q_{\text{dec}},K_{\text{enc}},V_{\text{enc}})
=\operatorname{softmax}\!\Big(\tfrac{Q_{\text{dec}}K_{\text{enc}}^\top}{\sqrt{d_k}}\Big)V_{\text{enc}},$$
where queries come from the decoder and keys/values from the encoder output.

### 3.2 Span corruption (the pretraining objective)
Sample a corruption rate $\rho$ (T5 uses $\approx 15\%$). Mark tokens i.i.d. with
probability $\rho$, then merge **consecutive** marked tokens into spans. Replace
each span $s_k$ in the input by a unique sentinel $\langle X_k\rangle$, and define
the target as the sentinels interleaved with the dropped spans:
$$
\underbrace{\text{the}\ \langle X_0\rangle\ \text{walked}\ \langle X_1\rangle\ \text{dog}}_{\text{encoder input}}
\;\longrightarrow\;
\underbrace{\langle X_0\rangle\ \text{cat}\ \langle X_1\rangle\ \text{the}\ \langle\text{EOS}\rangle}_{\text{decoder target}}.
$$
The loss is the usual decoder cross-entropy over this short target. Because the
target contains only the *missing* spans (not the whole input), each example is
cheap and dense in signal — a middle ground between BERT's MLM (predict isolated
tokens) and full sequence reconstruction.

### 3.3 Why an encoder-decoder (vs decoder-only)?
The encoder sees the source **bidirectionally** (good for understanding), while
the decoder stays autoregressive (good for generation). Cross-attention is the
bridge: every output step can query the entire, fully-contextualized source.
"""),
        md("## 4. Key component — span corruption + the cross-attending decoder layer"),
        show(MOD, "span_corrupt", "DecoderLayer"),
        md("## 5. Full model — T5 encoder-decoder (shared/tied embedding)"),
        show(MOD, "T5"),
        md("## 6. Train / run — span-corruption demo + a text-to-text REVERSE task"),
        run_demo(MOD),
        md("## 7. Visualization — span corruption + the seq2seq training loss"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, torch, matplotlib.pyplot as plt
import t5 as M

torch.manual_seed(M.SEED); np.random.seed(M.SEED)

# (a) train the REVERSE task and record the loss curve
V, L = 20, 6
src, tin, tout = M.make_reverse_data(512, L, V)
src = torch.tensor(src); tin = torch.tensor(tin); tout = torch.tensor(tout)
model = M.T5(V, d_model=48, n_heads=4, d_ff=96, n_layers=2, max_len=L + 2)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
lossfn = torch.nn.CrossEntropyLoss(ignore_index=M.PAD)
tmask = M.causal_mask(tin.size(1), torch.device("cpu"))
losses = []
for _ in range(150):
    logits = model(src, tin, tgt_mask=tmask)
    loss = lossfn(logits.reshape(-1, V), tout.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

# (b) the decoder causal mask used for teacher forcing
cm = M.causal_mask(L + 1, torch.device("cpu"))[0, 0].numpy()
cm = np.where(np.isneginf(cm), np.nan, 0.0) + np.tril(np.ones_like(cm))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(losses); ax[0].set_xlabel("step"); ax[0].set_ylabel("cross-entropy")
ax[0].set_title("Seq2seq REVERSE loss going down")
ax[1].imshow(cm, cmap="Greys", vmin=0, vmax=1)
ax[1].set_title("Decoder causal mask (teacher forcing)")
ax[1].set_xlabel("key pos j"); ax[1].set_ylabel("query pos i")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- T5 unifies NLP under one **text-to-text** interface and one cross-entropy loss;
  task identity lives in the input text (and, in real T5, a task prefix).
- **Span corruption** is the encoder-decoder analogue of MLM: drop spans, emit
  sentinels, regenerate — cheaper and denser than reconstructing the whole input.
- The encoder is bidirectional (understanding) and the decoder autoregressive
  (generation); **cross-attention** lets each output token query the full source.
- Pitfalls: mismatched sentinel ids between input and target; forgetting the
  decoder causal mask (leaks the target); encoder-decoder doubles the parameter
  count vs a decoder-only model of equal depth.
- Compare: **[GPT](gpt.ipynb)** (decoder-only) and **[BERT](bert.ipynb)**
  (encoder-only).
"""),
    ]

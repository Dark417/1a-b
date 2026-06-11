from tools.nbreg import register, md, code, show, run_demo

MOD = "seq2seq_attention"


@register("seq2seq_attention", "04.nlp/seq2seq/seq2seq_attention.ipynb")
def build():
    return [
        md(r"""
# Seq2Seq with Attention — Bahdanau & Luong

> Tutorial pair for [`seq2seq_attention.py`](seq2seq_attention.py). Recurrent
> backbone from [`lstm.ipynb`](../../02.dl/rnn/lstm.ipynb); the attention idea is
> generalized in [`attention.ipynb`](../../05.transformers/attention/attention.ipynb).

## 1. Intuition
An encoder-decoder reads the whole source, crushes it into **one** vector, and the
decoder must generate the entire output from that single summary. For anything
longer than a few tokens this bottleneck loses information. **Attention** removes
it: at every output step the decoder gets to *look back* at all the encoder
states and build a custom, weighted summary — a soft "alignment" between output
and input positions. We test on a **reverse** task (output = input reversed),
which forces the model to align output position $i$ with source position
$\text{len}-1-i$.
"""),
        md(r"""
## 2. Concept (the slide)
- **Encoder** RNN -> a sequence of hidden states $h_1,\dots,h_{T}$ (the *keys/
  values*).
- **Decoder** RNN with state $s_t$ (the *query*). Each step:
  1. score every encoder state against the query -> alignment $e_{t,j}$;
  2. softmax -> weights $\alpha_{t,j}$ (a distribution over source positions);
  3. context $c_t=\sum_j \alpha_{t,j} h_j$;
  4. predict the next token from $[s_t; c_t]$.
- **Bahdanau (additive):** a small MLP scores using the *previous* decoder state;
  context is fed **into** the decoder RNN.
- **Luong (multiplicative):** a dot/bilinear score using the *current* decoder
  state; context is combined **after** the RNN.
- Train with **teacher forcing**: feed the gold previous token (sometimes) instead
  of the model's own prediction.
"""),
        md(r"""
## 3. Math derivation

**Encoder.** $h_j=\mathrm{RNN}_{enc}(x_j,h_{j-1})$, $j=1\dots T_x$.

**Alignment scores.** Let the query be a decoder state. The three classic scoring
functions:
$$e_{t,j}=\begin{cases}
s_t^\top h_j & \textbf{Luong dot}\\[2pt]
s_t^\top W_a h_j & \textbf{Luong general (bilinear)}\\[2pt]
v_a^\top \tanh\!\big(W_a[s_t;h_j]\big) & \textbf{Bahdanau (additive)}
\end{cases}$$
Bahdanau uses the *previous* decoder state $s_{t-1}$ as the query; Luong uses the
*current* $s_t$. Bahdanau's additive form is a tiny one-hidden-layer MLP, so it
can score even when query and key live in different spaces.

**Attention weights** (a softmax over source positions — they sum to 1):
$$\alpha_{t,j}=\frac{\exp(e_{t,j})}{\sum_{k=1}^{T_x}\exp(e_{t,k})}.$$

**Context vector** is the attention-weighted average of the encoder states:
$$c_t=\sum_{j=1}^{T_x}\alpha_{t,j}\,h_j.$$
This is the heart of attention: a *content-based*, differentiable lookup. Because
$c_t$ is a convex combination of the $h_j$, gradients flow directly to every
source position — no information squeezed through a single vector.

**Output.** Combine context with the decoder state and project to the vocabulary:
$$P(y_t\mid y_{<t},x)=\operatorname{softmax}\big(W_o[s_t;c_t]+b_o\big).$$

**Bahdanau vs Luong, side by side.** Bahdanau computes $c_t$ from $s_{t-1}$ and
feeds $[y_{t-1};c_t]$ *into* the RNN to produce $s_t$; Luong computes $s_t$ first,
then $c_t$, then blends them at the output. Both learn the same alignments here;
Luong-dot is cheaper (no extra parameters).

**Teacher forcing.** The decoder is trained to predict $y_t$ given the *true*
prefix $y_{<t}$ with some probability, otherwise its own prediction. High forcing
speeds early learning but causes *exposure bias* (train/inference mismatch);
mixing the two is the usual compromise. (See
[`06.training-techniques/README.md`](../../06.training-techniques/README.md).)
"""),
        md("## 4. NumPy implementation — the attention scoring math, explicit"),
        show(MOD, "attention_numpy"),
        md("## 5. PyTorch implementation — encoder, decoder, and both attentions"),
        show(MOD, "Attention", "Seq2SeqAttention"),
        md("## 6. Train / run — both variants solve the reverse task"),
        run_demo(MOD),
        md("## 7. Visualization — the attention matrix is anti-diagonal (reversal)"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, torch, matplotlib.pyplot as plt
import seq2seq_attention as M

np.random.seed(0); torch.manual_seed(0)
src, tgt = M.make_reverse_data(n=512)
model = M.Seq2SeqAttention(attn_mode="bahdanau")
M.train(model, src[:448], tgt[:448], epochs=40)

s = src[460]
smax = max(len(x) for x in src[:448])
S = torch.tensor(M._pad([s], smax), device=next(model.parameters()).device)
pred, attn = model.greedy_decode(S, max_len=len(s) + 1)
A = attn[0, :len(s), :len(s)].cpu().numpy()   # (out positions, src positions)

plt.figure(figsize=(5, 4))
plt.imshow(A, cmap="viridis", aspect="auto")
plt.xlabel("source position"); plt.ylabel("output step")
plt.title("Attention alignment for the reverse task (anti-diagonal)")
plt.colorbar(label="attention weight"); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Attention is a **content-based soft lookup**: scores -> softmax weights ->
  weighted average of values. The decoder no longer depends on one bottleneck
  vector, so long sequences survive.
- **Bahdanau (additive)** vs **Luong (multiplicative)** differ mainly in the
  score function and *when* the context is mixed in; both learn clean alignments.
- **Teacher forcing** trades faster convergence for exposure bias — schedule it.
- Real systems should **mask PAD positions** out of the softmax; with fixed-length
  padding here we keep the padding consistent between training and decoding.
- Stack attention everywhere and drop the RNN entirely and you get
  **self-attention / [Transformers](../../05.transformers/architectures/transformer.ipynb)** —
  same scores-softmax-weighted-sum recipe, applied in parallel.
"""),
    ]

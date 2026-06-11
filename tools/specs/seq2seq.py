from tools.nbreg import register, md, code, show, run_demo

MOD = "seq2seq"


@register("seq2seq", "02.dl/rnn/seq2seq.ipynb")
def build():
    return [
        md(r"""
# Sequence-to-Sequence: Encoder–Decoder, Teacher Forcing & Attention

> Tutorial pair for [`seq2seq.py`](seq2seq.py). Builds on
> [`rnn.ipynb`](rnn.ipynb), [`lstm.ipynb`](lstm.ipynb), [`gru.ipynb`](gru.ipynb).

## 1. Intuition
How do you map one sequence to **another** sequence, possibly of a different
length (translate a sentence, reverse a string, summarize a paragraph)? Squeeze
the whole input into a vector with an **encoder** RNN, then **decode** it one
token at a time with a second RNN that conditions on what it has produced so far.
Two ideas make it actually train: **teacher forcing** (feed the decoder the
correct previous token during training) and **attention** (let the decoder peek
back at *every* encoder state instead of a single bottleneck vector).
"""),
        md(r"""
## 2. Concept (the slide)
- **Encoder:** read source $x_{1:S}$, produce hidden states $h^{enc}_{1:S}$ and a
  summary/context $c$ (e.g. the last state).
- **Decoder:** an RNN initialized from $c$ that, at step $t$, takes the previous
  target token and emits a distribution over the vocabulary for $y_t$.
- **Teacher forcing:** during *training* the "previous target token" is the
  ground-truth $y_{t-1}$, not the model's own guess — fast, stable gradients.
- **Inference:** no ground truth, so we feed the model's own previous prediction
  (greedy / beam search) — this is the *exposure-bias* gap.
- **Attention:** instead of one fixed $c$, recompute a *per-step* context
  $c_t=\sum_s \alpha_{t,s}h^{enc}_s$ from learned alignment weights $\alpha_{t,s}$.
"""),
        md(r"""
## 3. Math derivation

**Conditional factorization.** A seq2seq model is an autoregressive model of the
target given the source. By the chain rule of probability,
$$
p(y_{1:T}\mid x_{1:S})=\prod_{t=1}^{T} p\big(y_t \mid y_{<t},\,x_{1:S}\big).
$$
The decoder parameterizes each factor: with decoder state $s_t$ and context $c$
(or $c_t$ with attention),
$$
p(y_t\mid y_{<t},x)=\operatorname{softmax}\!\big(W_o[s_t;c_t]+b_o\big),\qquad
s_t=\mathrm{RNN}(\,[\,\mathrm{emb}(y_{t-1});c_t\,],\,s_{t-1}).
$$
Training maximizes the log-likelihood, i.e. minimizes token-level cross-entropy
$$
\mathcal L=-\sum_{t=1}^{T}\log p\big(y_t\mid y_{<t},x\big).
$$

**Teacher forcing.** The factorization conditions $y_t$ on the *true* prefix
$y_{<t}$. If we always plug in the ground-truth $y_{<t}$ during training, every
factor becomes an **independent classification** problem given $(x, y_{<t})$, so
gradients are clean and learning is fast. The alternative — feeding the model's
sampled $\hat y_{<t}$ — couples errors across steps and slows early training. The
price is **exposure bias**: at test time the model sees its own (imperfect)
history, a distribution it never trained on. A common compromise is *scheduled
sampling*: feed the gold token with probability $p$ and the model's own with
probability $1-p$ (the `teacher_forcing` ratio in the code).

**Attention context vector.** A single fixed $c$ is an information bottleneck for
long inputs. Attention computes, at each decoder step $t$, an alignment score
between the decoder state $s_t$ (or $s_{t-1}$) and each encoder state $h^{enc}_s$:
$$
e_{t,s}=\mathrm{score}(s_t,h^{enc}_s),\qquad
\alpha_{t,s}=\frac{\exp(e_{t,s})}{\sum_{s'}\exp(e_{t,s'})},\qquad
c_t=\sum_{s=1}^{S}\alpha_{t,s}\,h^{enc}_s.
$$
Two classic score functions:
$$
\textbf{Bahdanau (additive):}\quad e_{t,s}=v^\top\tanh\!\big(W_d s_t + W_e h^{enc}_s\big),
$$
$$
\textbf{Luong (dot):}\quad e_{t,s}=s_t^\top h^{enc}_s .
$$
$c_t$ is concatenated with $s_t$ before the output projection, giving the decoder
a direct, gradient-short path to *any* source position — the seed of the
Transformer's self-attention.
"""),
        md("## 4. NumPy implementation — compact GRU encoder–decoder + manual BPTT + teacher forcing"),
        show(MOD, "Seq2SeqNumPy"),
        md("## 5. PyTorch implementation — encoder–decoder, teacher forcing, Bahdanau/Luong attention"),
        show(MOD, "Attention", "Seq2SeqTorch"),
        md("## 6. Train / run — the tiny reverse task (no-attn vs Bahdanau vs Luong)"),
        run_demo(MOD),
        md("## 7. Visualization — loss curves and an attention alignment map"),
        code(r"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import seq2seq as M

M.seed_everything()
vocab, length = 6, 4
src, tgt = M.make_reverse_task(n=400, vocab=vocab, length=length)
tr, te = slice(0, 320), slice(320, 400)

# Train a no-attention baseline and a Bahdanau-attention model.
M.seed_everything()
plain = M.Seq2SeqTorch(vocab, emb=24, hidden=32, cell="gru", attention=None)
plain.fit(src[tr], tgt[tr], epochs=40, teacher_forcing=0.9)
M.seed_everything()
attn = M.Seq2SeqTorch(vocab, emb=24, hidden=32, cell="gru", attention="bahdanau")
attn.fit(src[tr], tgt[tr], epochs=40, teacher_forcing=0.9)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(plain.history, label="no attention")
ax1.plot(attn.history, label="Bahdanau attention")
ax1.set_xlabel("epoch"); ax1.set_ylabel("train CE loss")
ax1.set_title("Attention trains faster / lower")
ax1.legend(); ax1.grid(True, alpha=.3)

# Pull the attention matrix for one example: rows = decoder steps, cols = source.
dev = next(attn.parameters()).device
s = torch.as_tensor(src[te][0:1], dtype=torch.long, device=dev)
enc_out, hn = attn.encoder(attn.emb(s))
state = attn._enc_to_dec_state(hn)
mask = (s != M.PAD)
prev = torch.tensor([M.BOS], device=dev)
rows = []
for _ in range(length + 1):
    e = attn.emb(prev)
    context, a = attn.attn(state, enc_out, mask)
    rows.append(a.detach().cpu().numpy().ravel())
    dec_in = torch.cat([e, context], dim=1)
    state = attn.decoder_cell(dec_in, state)
    prev = attn.out(torch.cat([state, context], dim=1)).argmax(1)
A = np.array(rows)                                   # (Tdec, S)

im = ax2.imshow(A, aspect="auto", cmap="viridis")
ax2.set_xlabel("source position s"); ax2.set_ylabel("decoder step t")
ax2.set_title("Attention α (reverse task ⇒ anti-diagonal)")
fig.colorbar(im, ax=ax2, fraction=.046)
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- A seq2seq model factorizes $p(y_{1:T}\mid x)=\prod_t p(y_t\mid y_{<t},x)$ and is
  trained with token-level cross-entropy.
- **Teacher forcing** = condition on the *gold* prefix while training: fast,
  stable, but causes **exposure bias** at inference (mitigate with scheduled
  sampling / a teacher-forcing ratio $<1$).
- **Attention** removes the fixed-context bottleneck; for the reverse task the
  learned alignment is the **anti-diagonal**, a nice sanity check.
- **Pitfalls:** train/infer mismatch (always decode with the model's own outputs,
  never teacher force at test time); mask `<pad>` positions in attention and in
  the loss (`ignore_index`); and clip gradients — the decoder is deep in time.
- **Next:** stack attention everywhere and drop recurrence →
  [Transformers](../../05.transformers/architectures/transformer.ipynb).
"""),
    ]

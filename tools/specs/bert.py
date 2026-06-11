from tools.nbreg import register, md, code, show, run_demo

MOD = "bert"


@register("bert", "05.transformers/architectures/bert.ipynb")
def build():
    return [
        md(r"""
# BERT — encoder-only Transformer with masked-LM pretraining

> Tutorial pair for [`bert.py`](bert.py). Read
> [`transformer.ipynb`](transformer.ipynb) and
> [`attention.ipynb`](../attention/attention.ipynb) first.

## 1. Intuition
GPT reads left-to-right; BERT reads **both directions at once**. It keeps only the
Transformer *encoder* (no causal mask), so every token's representation is built
from the entire sentence. You cannot train such a model with next-token
prediction (it would trivially see the answer), so BERT instead **masks ~15% of
the tokens** and learns to fill them back in — the Masked Language Model objective.
A special **[CLS]** token aggregates the sequence for classification.
"""),
        md(r"""
## 2. Concept (the slide)
- **Bidirectional self-attention:** no causal mask; only a *padding* mask hides
  [PAD] positions.
- **Three embeddings summed:** token + learned positional + **segment** (sentence
  A vs B), enabling sentence-pair inputs `[CLS] A [SEP] B [SEP]`.
- **MLM head:** predict the original id at every masked position (tied to the input
  embedding matrix).
- **NSP head (conceptual):** classify from the pooled [CLS] vector whether
  sentence B follows A — uses the segment embeddings.
"""),
        md(r"""
## 3. Math derivation

### 3.1 Bidirectional attention
A BERT layer is exactly scaled-dot-product self-attention **without** a causal
mask:
$$A=\operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}}+M_{\text{pad}}\Big)V,$$
where $M_{\text{pad}}[i,j]=-\infty$ iff key $j$ is a [PAD] token, else $0$. Because
nothing blocks the upper triangle, token $i$'s output depends on the **whole**
sequence — left and right context.

### 3.2 The masked-language-model objective
Let $\mathcal{M}\subset\{1,\dots,L\}$ be the randomly chosen masked positions and
$\hat x$ the corrupted input. BERT maximizes the log-probability of the *original*
tokens at those positions:
$$\mathcal{L}_{\text{MLM}}
=-\sum_{t\in\mathcal{M}}\log p_\theta\!\big(x_t \mid \hat x\big).$$
Only the masked positions contribute (others have label $-100$ and are ignored).
This is a denoising autoencoder over discrete tokens.

### 3.3 The 80/10/10 masking scheme
For each chosen position $t\in\mathcal{M}$:
$$\hat x_t=\begin{cases}
\texttt{[MASK]} & \text{with prob } 0.8\\
\text{random token} & \text{with prob } 0.1\\
x_t\ \text{(unchanged)} & \text{with prob } 0.1.
\end{cases}$$
Why not always `[MASK]`? Because `[MASK]` never appears at fine-tuning time; the
10% random and 10% unchanged cases force the encoder to build a good
representation of **every** token (it can't tell which positions will be scored),
reducing the pretrain/finetune mismatch.

### 3.4 Next-Sentence Prediction (conceptual)
A binary head over the pooled [CLS] representation
$h_{\text{CLS}}=\tanh(W\,h_0)$ predicts whether segment B truly follows A:
$\mathcal{L}_{\text{NSP}}=-\log p_\theta(\text{IsNext}\mid h_{\text{CLS}})$. The
total pretraining loss is $\mathcal{L}_{\text{MLM}}+\mathcal{L}_{\text{NSP}}$.
(Later work — RoBERTa — drops NSP.)
"""),
        md("## 4. Key component — the MLM masking scheme + bidirectional attention"),
        show(MOD, "mlm_mask_numpy", "BidirectionalSelfAttention", "EncoderBlock"),
        md("## 5. Full model — BERT (token/pos/segment emb, encoder, MLM + NSP heads)"),
        show(MOD, "BERT"),
        md("## 6. Train / run — tiny masked-LM on toy `[CLS] … [SEP]` sequences"),
        run_demo(MOD),
        md("## 7. Visualization — masking scheme & the MLM training loss curve"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, torch, matplotlib.pyplot as plt
import bert as M

torch.manual_seed(M.SEED); np.random.seed(M.SEED); torch.set_num_threads(1)

# (a) which positions get masked, on a small batch
V, L, n = 24, 8, 12
seqs = M.make_sequences(n, L, V)
corrupt, labels = M.mlm_mask_numpy(seqs, V, p=0.20)
chosen = (labels != -100).astype(float)          # 1 where a token is predicted

# (b) a short MLM training run for the loss curve
seqs_tr = M.make_sequences(384, L, V)
corr_tr, lab_tr = M.mlm_mask_numpy(seqs_tr, V, p=0.20, seed=1)
pad = (seqs_tr != M.PAD).astype(np.int64)
X = torch.tensor(corr_tr); Y = torch.tensor(lab_tr)
S = torch.zeros_like(X); P = torch.tensor(pad)
model = M.BERT(V, d_model=48, n_heads=4, d_ff=96, n_layers=2, max_len=L + 2)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
lossfn = torch.nn.CrossEntropyLoss(ignore_index=-100)
losses = []
for _ in range(120):
    mlm_logits, _ = model(X, S, P)
    loss = lossfn(mlm_logits.reshape(-1, V), Y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
im = ax[0].imshow(chosen, cmap="Reds", aspect="auto")
ax[0].set_title("Masked positions (red = predicted)")
ax[0].set_xlabel("position"); ax[0].set_ylabel("sequence")
ax[1].plot(losses); ax[1].set_xlabel("step"); ax[1].set_ylabel("MLM cross-entropy")
ax[1].set_title("Masked-LM loss going down")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- BERT = a Transformer **encoder** trained by **masked-token reconstruction**;
  bidirectional context is its whole point and the reason it can't be a generator.
- The 80/10/10 scheme is not cosmetic — it removes the train/finetune `[MASK]`
  mismatch and forces the model to represent every token.
- Only ~15% of tokens produce a loss signal, so MLM pretraining is sample-hungry
  compared to causal LM (every token supervises GPT).
- Pitfalls: masking special tokens ([CLS]/[SEP]/[PAD]); forgetting the padding
  mask (the model attends to pad slots); using BERT for generation (it has no
  causal structure).
- Compare: **[GPT](gpt.ipynb)** (causal, generative) and the full
  **[T5](t5.ipynb)** encoder-decoder.
"""),
    ]

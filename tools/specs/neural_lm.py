from tools.nbreg import register, md, code, show, run_demo

MOD = "neural_lm"


@register("neural_lm", "nlp/language-models/neural_lm.ipynb")
def build():
    return [
        md(r"""
# Neural Language Models — embeddings + softmax over the vocabulary

> Tutorial pair for [`neural_lm.py`](neural_lm.py). Contrast with the count-based
> [`ngram_lm.ipynb`](ngram_lm.ipynb).

## 1. Intuition
A count-based n-gram model treats every word as an unrelated symbol, so seeing
"the cat ran" tells it nothing about "the dog ran". A **neural** LM first maps
each word to a dense **embedding**, so similar words sit near each other and
share statistical strength. It then summarizes the history with a neural net and
predicts the next word with a softmax. Two flavors: a fixed-window MLP (Bengio
2003) and an LSTM that reads the whole history.
"""),
        md(r"""
## 2. Concept (the slide)
- **Embed:** each word id $\to$ a learned vector $E[w]\in\mathbb R^{d}$.
- **Feed-forward LM:** concatenate the last $n-1$ embeddings, push through an MLP,
  softmax over the vocabulary. Context length is fixed.
- **Recurrent (LSTM) LM:** feed words one at a time; the hidden state carries an
  *unbounded* history. Predict the next token at every step.
- **Train** by minimizing cross-entropy (= negative log-likelihood of the true
  next word). **Perplexity** $=\exp(\text{cross-entropy})$.
- **Generate** by sampling from the softmax, sharpened/flattened by a
  *temperature*.
"""),
        md(r"""
## 3. Math derivation

**Setup.** Same chain-rule factorization as any LM,
$P(w_1^T)=\prod_t P(w_t\mid w_1^{t-1})$, but now each conditional is produced by a
neural network with parameters $\theta$.

**Feed-forward LM.** With context $c=(w_{t-n+1},\dots,w_{t-1})$ and embedding
table $E$:
$$x=\big[E[w_{t-n+1}];\dots;E[w_{t-1}]\big],\quad
h=\tanh(W_1 x+b_1),\quad z=W_2 h+b_2.$$

**Softmax over the vocabulary.** The logits $z\in\mathbb R^{|V|}$ become a
distribution
$$P(w=i\mid c)=\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_{j} e^{z_j}}.$$

**Cross-entropy loss.** For the true next word $y$,
$$\mathcal L=-\log P(y\mid c)=-\log\operatorname{softmax}(z)_y= -z_y+\log\textstyle\sum_j e^{z_j}.$$
Its gradient w.r.t. the logits is the clean softmax-minus-one-hot:
$$\frac{\partial\mathcal L}{\partial z}=\operatorname{softmax}(z)-\mathbf 1_y,$$
which backpropagates through $W_2,h,W_1$ and finally **scatters** into the
embedding rows of the context words (each gets $\partial\mathcal L/\partial x$
for its slice). The module codes this by hand.

**Perplexity.** The standard intrinsic metric is the exponentiated average loss:
$$\mathrm{PP}=\exp\!\Big(\frac1N\sum_t\mathcal L_t\Big)
=\exp\!\Big(-\frac1N\sum_t\log P(w_t\mid w_1^{t-1})\Big).$$
So perplexity is *literally* $e$ raised to the cross-entropy — minimizing one
minimizes the other.

**Temperature sampling.** To generate, sample $w\sim\operatorname{softmax}(z/T)$.
$T\!\to\!0$ is greedy/argmax (safe, repetitive); $T=1$ is the model's own
distribution; $T>1$ flattens it (more surprising, more mistakes).

**Recurrent LM.** The LSTM replaces the fixed window: $h_t=\mathrm{LSTM}(E[w_t],
h_{t-1})$ and $z_t=W h_t+b$. The same softmax cross-entropy applies at every step;
because the recurrence is deep in time we **clip gradients** to stop them
exploding (see [`lstm.ipynb`](../../dl/rnn/lstm.ipynb)).
"""),
        md("## 4. NumPy implementation — feed-forward LM with manual backprop"),
        show(MOD, "FeedForwardLMNumPy"),
        md("## 5. PyTorch implementation — an LSTM language model"),
        show(MOD, "LSTMLanguageModel"),
        md("## 6. Train / run — perplexity + sampled text from both models"),
        run_demo(MOD),
        md("## 7. Visualization — training perplexity curve (cross-entropy -> ppl)"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import neural_lm as M

tok = M._tokenize(M.toy_text())
ff = M.FeedForwardLMNumPy(n=3, dim=16, hidden=32, lr=0.3).fit(tok, epochs=150)

plt.figure(figsize=(7, 4))
plt.plot(np.exp(ff.history))         # history stores mean cross-entropy/epoch
plt.xlabel("epoch"); plt.ylabel("training perplexity  exp(CE)")
plt.title("Neural n-gram LM: perplexity falls as the softmax sharpens")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- The neural LM's superpower is **generalization through embeddings** — unseen
  word combinations still get reasonable probability because similar words share
  geometry. No explicit smoothing needed.
- **Perplexity $=\exp(\text{cross-entropy})$**: report it, and compare only with
  matching vocabulary/tokenization.
- The **feed-forward** LM has a fixed context window (like an n-gram); the
  **LSTM** has unbounded context but needs gradient clipping and is sequential.
- On a tiny repeated toy text the LSTM can essentially **memorize** it (ppl near
  1) — on real data you must watch for overfitting (held-out perplexity, dropout,
  weight tying).
- Attention-based LMs ([Transformers](../../transformers/architectures/transformer.ipynb))
  drop recurrence for parallel, long-range context — today's state of the art.
"""),
    ]

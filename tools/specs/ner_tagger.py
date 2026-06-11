from tools.nbreg import register, md, code, show, run_demo

MOD = "ner_tagger"


@register("ner_tagger", "04.nlp/seq2seq/ner_tagger.ipynb")
def build():
    return [
        md(r"""
# Sequence Tagging — BiLSTM + linear-chain CRF

> Tutorial pair for [`ner_tagger.py`](ner_tagger.py). Uses the
> [LSTM](../../02.dl/rnn/lstm.ipynb) as its backbone.

## 1. Intuition
Tasks like **named-entity recognition** label every token: is "Paris" a location,
a person, or just a word? A BiLSTM reads each sentence forwards *and* backwards
and emits a score for each tag at each position. But tagging each token
independently lets the model produce *illegal* label sequences — an `I-PER`
("inside a person span") with no `B-PER` before it. A **CRF** layer fixes this by
scoring the *whole label sequence jointly*, learning which tag transitions are
allowed. It decodes with **Viterbi** and trains via the **forward algorithm**.
"""),
        md(r"""
## 2. Concept (the slide)
- **Emissions** $E[t,k]$: how much token $t$ likes tag $k$ (from the BiLSTM).
- **Transitions** $T[i,j]$: learned score of following tag $i$ with tag $j$.
- A label path's score is $\sum_t E[t,y_t]+\sum_t T[y_{t-1},y_t]$ (+ start/end).
- **Training:** maximize $P(y\mid E)=e^{\text{score}(y)}/Z$. The hard part is the
  partition function $Z$ (a sum over $K^L$ paths) — the **forward algorithm**
  computes $\log Z$ in $O(LK^2)$.
- **Decoding:** the **Viterbi** algorithm (same recursion with $\max$) returns the
  single best legal path.
"""),
        md(r"""
## 3. Math derivation

**Potentials & path score.** For a length-$L$ sentence with $K$ tags, given
emission scores $E[t,k]$ and a transition matrix $T[i,j]$, the unnormalized score
of a tag path $y=(y_1,\dots,y_L)$ is
$$\operatorname{score}(E,y)=\pi_{y_1}+E[1,y_1]+\sum_{t=2}^{L}\Big(T[y_{t-1},y_t]+E[t,y_t]\Big)+\rho_{y_L},$$
with start/end biases $\pi,\rho$.

**Conditional probability (structured softmax).**
$$P(y\mid E)=\frac{\exp\operatorname{score}(E,y)}{Z(E)},\qquad
Z(E)=\sum_{y'\in K^{L}}\exp\operatorname{score}(E,y').$$
$Z$ sums over **exponentially many** paths — intractable by brute force.

**Forward algorithm for $\log Z$.** Define $\alpha_t[k]=\log\sum_{y_{1:t}:\,y_t=k}
\exp\operatorname{score}$. The chain structure gives a recursion:
$$\alpha_1[k]=\pi_k+E[1,k],\qquad
\alpha_t[k]=E[t,k]+\operatorname*{logsumexp}_{i}\big(\alpha_{t-1}[i]+T[i,k]\big),$$
$$\log Z=\operatorname*{logsumexp}_{k}\big(\alpha_L[k]+\rho_k\big).$$
This is $O(LK^2)$. We use the **log-sum-exp trick**
$\log\sum_i e^{x_i}=m+\log\sum_i e^{x_i-m}$ ($m=\max_i x_i$) for numerical
stability — exponentiating raw scores would overflow.

**CRF loss.** Negative log-likelihood of the gold path:
$$\mathcal L=-\log P(y\mid E)=\log Z(E)-\operatorname{score}(E,y).$$
Its gradient w.r.t. the transition $T[i,j]$ is the classic
**expected-minus-empirical counts**:
$$\frac{\partial\mathcal L}{\partial T[i,j]}=\underbrace{\sum_t P(y_{t-1}=i,y_t=j\mid E)}_{\text{model expectation (forward-backward)}}-\underbrace{\sum_t \mathbb 1[y_{t-1}=i,y_t=j]}_{\text{gold counts}}.$$
The edge marginals come from **forward-backward**:
$\alpha$ (forward) and $\beta$ (backward) messages give
$P(y_{t-1}=i,y_t=j\mid E)\propto e^{\alpha_{t-1}[i]+T[i,j]+E[t,j]+\beta_t[j]-\log Z}$.
(In the neural model PyTorch autograd computes this gradient for us.)

**Viterbi decoding.** Replace $\operatorname{logsumexp}$ with $\max$ and store
backpointers:
$$\delta_t[k]=E[t,k]+\max_i\big(\delta_{t-1}[i]+T[i,k]\big),\quad
\psi_t[k]=\arg\max_i(\cdot),$$
then start from $\arg\max_k(\delta_L[k]+\rho_k)$ and follow $\psi$ back to recover
the single most probable **legal** tag sequence.
"""),
        md("## 4. NumPy implementation — linear-chain CRF (forward algorithm + Viterbi)"),
        show(MOD, "LinearChainCRF"),
        md("## 5. PyTorch implementation — BiLSTM emissions + a differentiable CRF"),
        show(MOD, "CRFTorch", "BiLSTMCRF"),
        md("## 6. Train / run — verify log Z, then tag with BiLSTM-CRF"),
        run_demo(MOD),
        md("## 7. Visualization — the learned CRF transition matrix encodes grammar"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, torch, matplotlib.pyplot as plt
import ner_tagger as M

np.random.seed(0); torch.manual_seed(0)
sents, tags = M.make_tagging_data(n=200)
model = M.BiLSTMCRF(len(M.WORDS), len(M.TAGS), use_crf=True).fit(sents[:150], tags[:150], epochs=25)
T = model.crf.T.detach().numpy()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(T, cmap="RdBu_r", vmin=-abs(T).max(), vmax=abs(T).max())
ax.set_xticks(range(len(M.TAGS))); ax.set_xticklabels(M.TAGS, rotation=45, ha="right")
ax.set_yticks(range(len(M.TAGS))); ax.set_yticklabels(M.TAGS)
ax.set_xlabel("to tag  (y_t)"); ax.set_ylabel("from tag  (y_{t-1})")
ax.set_title("Learned CRF transitions: bright = encouraged, dark = forbidden")
fig.colorbar(im, ax=ax, label="transition score"); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- A CRF turns independent per-token decisions into a **structured** one: it learns
  that `I-PER` can't follow `O`, that `B-LOC` precedes `I-LOC`, etc., so it never
  emits malformed spans the way a plain softmax tagger can.
- The whole method rests on two dynamic programs over the chain: the **forward
  algorithm** (training, sums paths -> $\log Z$) and **Viterbi** (decoding,
  maximizes over paths). They share the same recursion, swapping
  $\operatorname{logsumexp}\leftrightarrow\max$.
- Always use the **log-sum-exp trick** — naive exponentiation overflows.
- On an easy toy task even the plain BiLSTM reaches 100%; the CRF's advantage
  shows up on **ambiguous tokens and long spans**, where the transition grammar
  breaks ties. Inspect the transition matrix to see the grammar it learned.
- This is the architecture behind classic neural NER (Lample et al., 2016); modern
  systems swap the BiLSTM for a [Transformer](../../05.transformers/architectures/transformer.ipynb)
  encoder but often keep the CRF head.
"""),
    ]

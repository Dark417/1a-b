from tools.nbreg import register, md, code, show, run_demo

MOD = "word2vec"


@register("word2vec", "nlp/embeddings/word2vec.ipynb")
def build():
    return [
        md(r"""
# word2vec — word vectors from context (Skip-gram & CBOW)

> Tutorial pair for [`word2vec.py`](word2vec.py).

## 1. Intuition
"You shall know a word by the company it keeps." Words appearing in similar
contexts should get similar vectors. word2vec slides a window over text and
trains vectors so that a word predicts its neighbours (Skip-gram) — and the
resulting geometry encodes meaning (analogies become vector arithmetic).
"""),
        md(r"""
## 2. Concept (the slide)
- **Skip-gram:** given the center word, predict each context word.
- **CBOW:** given the (averaged) context, predict the center word.
- Two embedding tables: "input" vectors $v_w$ and "output" vectors $u_w$.
- **Negative sampling** turns an expensive $|V|$-way softmax into a few cheap
  binary classifications.
"""),
        md(r"""
## 3. Math derivation

**Full Skip-gram objective.** Maximize the probability of context words:
$$\frac1T\sum_t\sum_{-m\le j\le m,\,j\ne0}\log p(w_{t+j}\mid w_t),\qquad
  p(o\mid c)=\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}.$$
The denominator sums over the **whole vocabulary** — too expensive.

**Negative sampling (SGNS).** Replace it with: make the true pair score high and a
few random ("negative") pairs score low. For center $c$, true context $o$, and
negatives $k\sim P_n$:
$$\mathcal L=-\log\sigma(u_o^\top v_c)-\sum_{k}\mathbb E_{k\sim P_n}\log\sigma(-u_k^\top v_c).$$

**Gradients** (same clean form as logistic regression — let $s_w=\sigma(u_w^\top v_c)$):
$$\frac{\partial\mathcal L}{\partial u_o}=(s_o-1)v_c,\quad
  \frac{\partial\mathcal L}{\partial u_k}=(s_k-0)v_c,\quad
  \frac{\partial\mathcal L}{\partial v_c}=\sum_{w\in\{o\}\cup\text{neg}}(s_w-y_w)\,u_w.$$

**Noise distribution.** Negatives are drawn from the **unigram raised to the 3/4
power**, $P_n(w)\propto \text{count}(w)^{0.75}$ — empirically the sweet spot
between sampling frequent and rare words. Frequent words are also randomly
**subsampled** so "the" doesn't dominate.

**CBOW** is the mirror image: average the context vectors into $v_c=\frac1{|ctx|}\sum v_w$
and predict the center; the gradient w.r.t. each context vector is the averaged
$\partial\mathcal L/\partial v_c$.
"""),
        md("## 4. NumPy implementation — Skip-gram/CBOW with negative sampling"),
        show(MOD, "Word2VecNumPy"),
        md("## 5. PyTorch implementation — SGNS with `nn.Embedding`"),
        show(MOD, "SGNSTorch"),
        md("## 6. Train on a toy corpus — neighbours respect topic (animals vs fruits)"),
        run_demo(MOD),
        md("## 7. Visualization — embeddings cluster by meaning (PCA to 2-D)"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
import word2vec as M

sents = M.toy_corpus()
w = M.Word2VecNumPy(dim=20, window=2, neg=5).fit(sents, epochs=60)
words = list(w.itos)
V = np.stack([w.vec(t) for t in words])
V = V - V.mean(0)
U, S, Vt = np.linalg.svd(V, full_matrices=False)   # PCA via SVD
P = V @ Vt[:2].T

animals = set("dog cat lion tiger horse cow".split())
plt.figure(figsize=(6, 5))
for (x, y), t in zip(P, words):
    color = "tab:red" if t in animals else "tab:green"
    plt.scatter(x, y, color=color); plt.annotate(str(t), (x, y), fontsize=9)
plt.title("word2vec embeddings (red=animals, green=fruits)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- SGNS = many small logistic regressions; gradient is the familiar $(\hat y-y)$.
- The $0.75$-power noise distribution and frequent-word subsampling matter in
  practice.
- Static embeddings give one vector per word — no sense disambiguation. That's
  what **contextual** models (ELMo → BERT) fix, using the
  [Transformer](../../transformers/architectures/transformer.ipynb).
"""),
    ]

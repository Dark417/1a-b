from tools.nbreg import register, md, code, show, run_demo

MOD = "fasttext"


@register("fasttext", "nlp/embeddings/fasttext.ipynb")
def build():
    return [
        md(r"""
# fastText — embeddings made of subword pieces

> Tutorial pair for [`fasttext.py`](fasttext.py). Builds on
> [`word2vec.ipynb`](word2vec.ipynb).

## 1. Intuition
word2vec treats every word as an atom: "play", "plays", and "playing" are three
unrelated symbols, and a word it never saw at training time has *no vector at
all*. That is wasteful — those words obviously share structure. fastText breaks a
word into **character n-grams** ("playing" -> `pla`, `lay`, `ayi`, ...) and makes
the word vector the **sum** of its piece vectors. Shared pieces pull related
words together, and any brand-new word can be assembled from pieces it knows.
"""),
        md(r"""
## 2. Concept (the slide)
- Wrap a word in boundary markers: `where` -> `<where>`, so prefixes and suffixes
  are distinguishable.
- Extract all character n-grams for $n\in[\text{minn},\text{maxn}]$ (plus the whole
  word as one special token).
- A word's input vector is $v_w=\sum_{g\in G(w)} z_g$ (sum of subword vectors).
- Train exactly like **skip-gram with negative sampling**, but the center vector
  is this subword sum; the gradient flows back to every subword.
- **OOV words** get a vector for free by summing whatever subwords they contain.
"""),
        md(r"""
## 3. Math derivation

**Subword set.** For word $w$ let $G(w)$ be its set of character n-grams (with the
boundary-marked whole word included). fastText composes the word's input vector
additively from a shared subword table $z_g$:
$$\boxed{\,v_w=\sum_{g\in G(w)} z_g.\,}$$
This is the entire modeling change over word2vec — everything else is skip-gram.

**Scoring.** The compatibility of center $w$ with a context word $c$ uses a
per-word **output** vector $u_c$:
$$s(w,c)=v_w^\top u_c=\Big(\sum_{g\in G(w)} z_g\Big)^{\!\top} u_c.$$

**Negative-sampling loss.** For a true context $o$ and negatives $k\sim P_n$ drawn
from the unigram$^{3/4}$ distribution:
$$\mathcal L=-\log\sigma(v_w^\top u_o)-\sum_{k}\log\sigma(-v_w^\top u_k).$$

**Gradients.** Let $s_x=\sigma(v_w^\top u_x)$ and label $y_o=1,\,y_k=0$. As in
plain SGNS,
$$\frac{\partial\mathcal L}{\partial u_x}=(s_x-y_x)\,v_w,\qquad
\frac{\partial\mathcal L}{\partial v_w}=\sum_{x\in\{o\}\cup\text{neg}}(s_x-y_x)\,u_x.$$
The new step: $v_w$ is a **sum** of subwords, so by the chain rule the same
gradient lands on *each* subword that built it:
$$\frac{\partial\mathcal L}{\partial z_g}=\frac{\partial\mathcal L}{\partial v_w}
\quad\text{for every }g\in G(w).$$
That sharing is why morphologically related words (and unseen words) inherit
meaning: every time "playing" is updated, the subword `play` improves, which in
turn improves "played", "player", and the unseen "playful".

**Implementation note.** The subword table is large, so fastText **hashes** each
n-gram string into a fixed number of buckets; `nn.EmbeddingBag(mode="sum")`
realises the sum $v_w=\sum z_g$ in one pooled lookup.
"""),
        md("## 4. NumPy implementation — subword skip-gram with negative sampling"),
        show(MOD, "char_ngrams", "FastTextNumPy"),
        md("## 5. PyTorch implementation — `EmbeddingBag(mode='sum')` = Σ subwords"),
        show(MOD, "FastTextTorch"),
        md("## 6. Train / run — morphology clusters and an OOV word gets a vector"),
        run_demo(MOD),
        md("## 7. Visualization — word families cluster; an OOV word lands among them"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import fasttext as M

sents = M.toy_corpus()
ft = M.FastTextNumPy(dim=30, window=2, neg=5).fit(sents, epochs=40)
words = list(ft.itos) + ["playful"]   # last one is OOV (never trained)
V = np.stack([ft.word_vector(w) for w in words]); V = V - V.mean(0)
_, _, Vt = np.linalg.svd(V, full_matrices=False)
P = V @ Vt[:2].T

fams = {"play": "tab:red", "walk": "tab:blue", "jump": "tab:green"}
plt.figure(figsize=(7, 5))
for (x, y), t in zip(P, words):
    color = "k"
    for stem, c in fams.items():
        if t.startswith(stem):
            color = c
    marker = "*" if t == "playful" else "o"
    size = 220 if t == "playful" else 40
    plt.scatter(x, y, color=color, marker=marker, s=size)
    plt.annotate(t, (x, y), fontsize=8)
plt.title("fastText: morphological families cluster; OOV 'playful' (star) joins 'play'")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- The one idea — **word vector = sum of subword vectors** — buys two big wins:
  morphological generalization and **open-vocabulary** (OOV) coverage.
- Choosing `minn..maxn` matters: too small captures only spelling, too large
  approaches word-level word2vec. Typical default is $3\!-\!6$.
- Subwords are **hashed** into buckets, so collisions happen; more buckets =
  fewer collisions at the cost of memory.
- It still produces **static** embeddings (one vector per word/subword, no
  context). Subword tokenization (BPE/WordPiece) carries this idea into
  [Transformers](../../transformers/architectures/transformer.ipynb), but there
  the *contextual* representation comes from attention, not a fixed sum.
"""),
    ]

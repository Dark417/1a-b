from tools.nbreg import register, md, code, show, run_demo

MOD = "glove"


@register("glove", "04.nlp/embeddings/glove.ipynb")
def build():
    return [
        md(r"""
# GloVe — Global Vectors from co-occurrence statistics

> Tutorial pair for [`glove.py`](glove.py). Compare with
> [`word2vec.ipynb`](word2vec.ipynb).

## 1. Intuition
word2vec walks a window across text and learns from one local context at a time.
GloVe says: why not use **all** the statistics at once? Count how often every
pair of words co-occurs in the whole corpus — that giant co-occurrence matrix
already contains the meaning. GloVe then finds word vectors whose **dot product
equals the log of how often the words co-occur**. The magic comes from *ratios*:
the ratio of co-occurrence probabilities is what separates "ice" from "steam".
"""),
        md(r"""
## 2. Concept (the slide)
- Build the co-occurrence matrix $X$, where $X_{ij}$ = (weighted) number of times
  word $j$ appears in the context of word $i$. Optionally weight a context word
  $d$ positions away by $1/d$ (closer words matter more).
- Learn word vectors $w_i$, context vectors $\tilde w_j$, and biases so that
  $$w_i^\top \tilde w_j + b_i + \tilde b_j \approx \log X_{ij}.$$
- Fit by **weighted** least squares, where the weight $f(X_{ij})$ caps the
  influence of extremely frequent pairs (like "the the").
- Optimize with **AdaGrad**; the final embedding is $w_i + \tilde w_i$.
"""),
        md(r"""
## 3. Math derivation — why *log* co-occurrence?

Let $P_{ij}=P(j\mid i)=X_{ij}/X_i$ be the probability that word $j$ appears in the
context of $i$. Pennington et al. observe that **meaning lives in ratios**: for a
probe word $k$, the ratio $P_{ik}/P_{jk}$ is large when $k$ relates to $i$ not
$j$, small in the reverse case, and $\approx 1$ when $k$ relates to both or
neither. We want a function $F$ of the vectors to reproduce that ratio:
$$F\big(w_i,w_j,\tilde w_k\big)=\frac{P_{ik}}{P_{jk}}.$$

Vector spaces are linear, so make $F$ depend on the **difference** $w_i-w_j$, and
since the right side is a scalar, on its dot product with $\tilde w_k$:
$$F\big((w_i-w_j)^\top \tilde w_k\big)=\frac{P_{ik}}{P_{jk}}.$$

We want $F$ to turn subtraction in the argument into division on the right — that
is the homomorphism $F(a-b)=F(a)/F(b)$, whose solution is $F=\exp$. Then
$$\exp(w_i^\top \tilde w_k)=P_{ik}=\frac{X_{ik}}{X_i}
\;\Longrightarrow\; w_i^\top \tilde w_k=\log X_{ik}-\log X_i.$$
The $\log X_i$ term depends only on $i$, so absorb it (and a context-side
constant) into **bias** terms $b_i,\tilde b_k$:
$$\boxed{\,w_i^\top \tilde w_k + b_i + \tilde b_k = \log X_{ik}.\,}$$

**The weighted least-squares objective.** Turn that target into a regression,
weighting each residual by $f(X_{ij})$ so rare noisy pairs and ultra-frequent
pairs don't dominate:
$$J=\sum_{i,j} f(X_{ij})\,\big(w_i^\top \tilde w_j + b_i + \tilde b_j-\log X_{ij}\big)^2,$$
$$f(x)=\begin{cases}(x/x_{\max})^{\alpha} & x<x_{\max}\\ 1 & x\ge x_{\max}\end{cases}
\quad(\alpha=3/4,\;x_{\max}=100).$$
$f(0)=0$ skips the (huge number of) zero entries, and $f$ rises then *plateaus*
so "the" can't swamp the loss.

**Gradient.** For one entry, with residual
$r_{ij}=w_i^\top\tilde w_j+b_i+\tilde b_j-\log X_{ij}$:
$$\frac{\partial J}{\partial w_i}=f(X_{ij})\,r_{ij}\,\tilde w_j,\qquad
\frac{\partial J}{\partial \tilde w_j}=f(X_{ij})\,r_{ij}\,w_i,\qquad
\frac{\partial J}{\partial b_i}=\frac{\partial J}{\partial \tilde b_j}=f(X_{ij})\,r_{ij}.$$
GloVe trains these with **AdaGrad** (per-parameter adaptive step
$\eta/\sqrt{\sum g^2}$), which suits the wildly varying word frequencies.
"""),
        md("## 4. NumPy implementation — co-occurrence build + AdaGrad SGD"),
        show(MOD, "GloVeNumPy"),
        md("## 5. PyTorch implementation — same objective via autograd + `Adagrad`"),
        show(MOD, "GloVeTorch"),
        md("## 6. Train / run — fit on a toy corpus and recover log X_ij"),
        run_demo(MOD),
        md("## 7. Visualization — embeddings split by topic (PCA to 2-D)"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import glove as M

sents = M._tokenize(M.toy_corpus())
g = M.GloVeNumPy(dim=20, window=3, x_max=30, lr=0.05).fit(sents, epochs=150)
words = list(g.itos)
V = np.stack([g.vec(t) for t in words]); V = V - V.mean(0)
_, _, Vt = np.linalg.svd(V, full_matrices=False)   # PCA via SVD
P = V @ Vt[:2].T

animals = set("dog cat lion tiger wolf bark hunt run chase prey fur paws".split())
plt.figure(figsize=(6.5, 5))
for (x, y), t in zip(P, words):
    color = "tab:red" if t in animals else "tab:green"
    plt.scatter(x, y, color=color); plt.annotate(t, (x, y), fontsize=8)
plt.title("GloVe embeddings (red = animal topic, green = fruit topic)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- GloVe = **matrix factorization of log co-occurrence** with a clever weighting;
  word2vec is implicitly factorizing a (shifted-PMI) matrix too, so the two are
  close cousins reaching similar geometry from opposite directions
  (global counts vs local windows).
- The **weighting $f(X)$** is essential: without it, function words dominate; with
  $f(0)=0$ you also skip the overwhelmingly common zero entries.
- Use the **sum** $w_i+\tilde w_i$ as the final vector — it averages out noise.
- Static embeddings still give one vector per word — context-dependent meaning
  needs [Transformers](../../05.transformers/architectures/transformer.ipynb).
- On a *tiny* toy corpus the co-occurrence matrix is sparse and noisy; real GloVe
  shines on billions of tokens.
"""),
    ]

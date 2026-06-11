from tools.nbreg import register, md, code, show, run_demo

MOD = "ngram_lm"


@register("ngram_lm", "04.nlp/language-models/ngram_lm.ipynb")
def build():
    return [
        md(r"""
# n-gram Language Models — counting your way to P(next word)

> Tutorial pair for [`ngram_lm.py`](ngram_lm.py).

## 1. Intuition
A language model answers one question: *given the words so far, what comes
next?* The oldest, simplest answer is to **count**. If "the cat sat on the" was
usually followed by "mat" in our corpus, predict "mat". An n-gram model only
looks at the last $n-1$ words (a Markov shortcut), tallies what followed them,
and turns those tallies into probabilities. The whole art is what to do about
word sequences you have **never seen** — that is what *smoothing* fixes.
"""),
        md(r"""
## 2. Concept (the slide)
- **n-gram:** a contiguous run of $n$ tokens. Bigram $n=2$, trigram $n=3$.
- **Markov assumption:** condition only on the previous $n-1$ tokens.
- **MLE:** estimate $P(w\mid h)$ as a normalized count.
- **Zero-count problem:** any unseen n-gram gets probability 0, sending
  perplexity to $\infty$. So we **smooth**:
  - *Add-k / Laplace*: pretend every n-gram was seen $k$ extra times.
  - *Stupid backoff*: if the n-gram is unseen, fall back to a shorter one,
    scaled by a constant $\alpha$ (cheap, unnormalized).
  - *Kneser-Ney*: discount a fixed mass $D$ and redistribute it using a
    **continuation** probability (how many distinct contexts a word completes).
- **Perplexity:** the geometric-mean branching factor — lower is better.
"""),
        md(r"""
## 3. Math derivation

**Chain rule (exact).** Every sequence factorizes as
$$P(w_1,\dots,w_T)=\prod_{t=1}^{T} P(w_t\mid w_1,\dots,w_{t-1}).$$

**n-gram (Markov) approximation.** Truncate the history to the last $n-1$ tokens:
$$P(w_t\mid w_1^{t-1})\approx P(w_t\mid w_{t-n+1}^{t-1}).$$

**Maximum-likelihood estimate.** With $c(\cdot)$ the corpus count and history
$h=w_{t-n+1}^{t-1}$,
$$P_{\mathrm{MLE}}(w\mid h)=\frac{c(h,w)}{c(h)}=\frac{c(h,w)}{\sum_{w'}c(h,w')}.$$

**Add-k smoothing.** Add a pseudo-count $k$ to every word in the vocabulary $V$:
$$P_{+k}(w\mid h)=\frac{c(h,w)+k}{c(h)+k\,|V|}.$$
$k=1$ is *Laplace*. Larger $k$ moves mass toward the uniform distribution.

**Stupid backoff** (Brants et al., 2007) is a recursive, *unnormalized* score:
$$S(w\mid h)=\begin{cases}\dfrac{c(h,w)}{c(h)} & c(h,w)>0\\[1.2em]
\alpha\, S(w\mid h') & \text{otherwise,}\end{cases}$$
where $h'$ drops the oldest word of $h$. Fast and surprisingly strong at scale.

**Kneser-Ney (interpolated).** Subtract a fixed discount $D$ from each seen count
and add an interpolation with the lower order:
$$P_{KN}(w\mid h)=\frac{\max(c(h,w)-D,\,0)}{c(h)}+\lambda(h)\,P_{KN}(w\mid h'),$$
where the back-off weight is exactly the discounted mass we removed:
$$\lambda(h)=\frac{D\,\big|\{w:c(h,w)>0\}\big|}{c(h)}.$$
The **key idea** is the lowest-order term. Instead of the unigram frequency, KN
uses the **continuation probability** — how many *distinct* words $w$ follows:
$$P_{\mathrm{cont}}(w)=\frac{N_{1+}(\bullet,w)}{N_{1+}(\bullet,\bullet)},\qquad
N_{1+}(\bullet,w)=\big|\{v:c(v,w)>0\}\big|.$$
"Francisco" is frequent but almost only after "San", so $N_{1+}(\bullet,w)$ is
tiny — KN rightly gives it low probability in a novel context.

**Perplexity.** The model's per-token uncertainty, the exponentiated average
negative log-likelihood:
$$\mathrm{PP}(W)=\exp\!\Big(-\frac1N\sum_{i=1}^{N}\log P(w_i\mid h_i)\Big)
=\Big(\prod_i P(w_i\mid h_i)\Big)^{-1/N}.$$
A perplexity of $K$ means the model is as confused as if choosing uniformly among
$K$ words at each step.
"""),
        md("## 4. NumPy implementation"),
        show(MOD, "NgramLM"),
        md("## 5. Reference — comparing smoothing methods (no neural net needed)"),
        show(MOD, "compare_smoothing"),
        md("## 6. Train / run — perplexity by smoother, the effect of k, and sampling"),
        run_demo(MOD),
        md("## 7. Visualization — how add-k strength trades off perplexity"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import ngram_lm as M

sents = M._tokenize(M.toy_corpus())
train, test = sents[:8], sents[8:]
ks = np.logspace(-3, 1, 25)

plt.figure(figsize=(7, 4))
for n, style in [(2, "-o"), (3, "-s")]:
    pps = []
    for k in ks:
        lm = M.NgramLM(n=n, mode="addk", k=float(k)).fit(train)
        pps.append(lm.perplexity(test))
    plt.plot(ks, pps, style, ms=4, label=f"{n}-gram add-k")
plt.xscale("log"); plt.xlabel("add-k pseudo-count k"); plt.ylabel("held-out perplexity")
plt.title("Too little smoothing overfits; too much -> uniform. There is a sweet spot.")
plt.legend(); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Always smooth.** A single unseen n-gram gives a test sentence probability 0
  and perplexity $\infty$.
- **Add-k** is easy but blunt; with a big vocabulary it steals far too much mass.
  **Kneser-Ney** is the count-based gold standard because of the continuation
  trick — context diversity, not raw frequency, drives the back-off.
- **Sparsity explodes with $n$.** Higher-order models capture more context but
  almost everything becomes unseen; you lean entirely on back-off.
- Count models can't generalize across *similar* words ("dog" vs "puppy"). That
  is exactly what neural LMs with embeddings fix — see
  [`neural_lm.ipynb`](neural_lm.ipynb).
- Compare perplexities only with the **same vocabulary and tokenization**.
"""),
    ]

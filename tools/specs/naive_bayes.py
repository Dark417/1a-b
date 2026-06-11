from tools.nbreg import register, md, code, show, run_demo

MOD = "naive_bayes"


@register("naive_bayes", "01.ml/naive-bayes/naive_bayes.ipynb")
def build():
    return [
        md(r"""
# Naive Bayes — Bayes' rule with a (very) convenient assumption

> Tutorial pair for [`naive_bayes.py`](naive_bayes.py).

## 1. Intuition
We want $P(\text{class}\mid\text{features})$. Bayes' rule flips it into something
we *can* estimate: $P(\text{features}\mid\text{class})\,P(\text{class})$. The hard
part is the joint $P(x_1,\dots,x_d\mid c)$. Naive Bayes makes a sweeping
assumption — given the class, features are **independent** — so that joint
collapses into a product of one-dimensional pieces we can count directly. The
assumption is almost always false, yet the *argmax* it produces is often right,
which is why NB is a famously strong, dirt-cheap baseline (especially for text).
"""),
        md(r"""
## 2. Concept (the slide)
- **Generative model:** each class has a story for generating features;
  classify by asking which class most likely generated $x$.
- **Conditional independence:** $P(x\mid c)=\prod_j P(x_j\mid c)$ — turns a
  $d$-dimensional density into $d$ tiny ones.
- **Event models (the three flavours):**
  - *Gaussian* — continuous $x_j$: $P(x_j\mid c)=\mathcal N(\mu_{cj},\sigma^2_{cj})$.
  - *Multinomial* — counts (bag-of-words): $P(x\mid c)\propto\prod_j P(j\mid c)^{x_j}$.
  - *Bernoulli* — binary present/absent, with an explicit *absence* term.
- **Fit = count.** Parameters are closed-form MLEs; no gradient descent.
- **Work in log-space** and **smooth** to dodge underflow and zero-probabilities.
"""),
        md(r"""
## 3. Math derivation — Bayes' rule, the naive factorization, MLE, smoothing

### Posterior via Bayes
$$P(y=c\mid x)=\frac{P(y=c)\,P(x\mid y=c)}{P(x)}
 =\frac{P(y=c)\,P(x\mid y=c)}{\sum_{c'}P(y=c')\,P(x\mid y=c')}.$$
The denominator $P(x)$ is constant across $c$, so for classification
$$\hat y=\arg\max_c\;P(y=c)\,P(x\mid y=c).$$

### The naive (conditional independence) assumption
$$P(x\mid y=c)=\prod_{j=1}^d P(x_j\mid y=c)
\;\Rightarrow\;
\boxed{\;\log P(y=c\mid x)=\log P(y=c)+\sum_{j=1}^d\log P(x_j\mid y=c)-\log Z\;}$$
We compute everything in **log-space** (sums, not products) and recover the
normalizer with log-sum-exp: $\log Z=\log\sum_c \exp(\text{joint}_c)$.

### Maximum-likelihood parameters (per event model)
**Gaussian.** For class $c$, feature $j$:
$$\mu_{cj}=\frac1{n_c}\sum_{i:y_i=c}x_{ij},\qquad
\sigma^2_{cj}=\frac1{n_c}\sum_{i:y_i=c}(x_{ij}-\mu_{cj})^2,$$
$$\log P(x_j\mid c)=-\tfrac12\Big(\log(2\pi\sigma^2_{cj})+\frac{(x_j-\mu_{cj})^2}{\sigma^2_{cj}}\Big).$$

**Multinomial.** $\hat\theta_{cj}=P(j\mid c)$ maximizes $\sum_j N_{cj}\log\theta_{cj}$
under $\sum_j\theta_{cj}=1$. A Lagrange multiplier gives $\theta_{cj}=N_{cj}/N_c$;
with **Laplace ($+\alpha$) smoothing**:
$$\hat\theta_{cj}=\frac{N_{cj}+\alpha}{N_c+\alpha\,d},\qquad
\log P(x\mid c)=\sum_j x_j\log\hat\theta_{cj}.$$

**Bernoulli.** $p_{cj}=P(x_j=1\mid c)$, smoothed:
$$p_{cj}=\frac{(\#\,c\text{-docs with }j)+\alpha}{n_c+2\alpha},\qquad
\log P(x\mid c)=\sum_j\big[x_j\log p_{cj}+(1-x_j)\log(1-p_{cj})\big].$$
The $(1-x_j)\log(1-p_{cj})$ term — *crediting absent words* — is exactly what
distinguishes Bernoulli from Multinomial.

### Why smoothing is mandatory
Without it, a single feature unseen in class $c$ gives $P(j\mid c)=0$, which
makes the *entire* product zero ($\log\to-\infty$) and vetoes class $c$ no matter
what the other features say. Adding $\alpha$ pseudo-counts (a Dirichlet/Beta prior
→ MAP estimate) keeps every probability strictly positive.
"""),
        md("## 4. NumPy implementation — Gaussian / Multinomial / Bernoulli from scratch"),
        show(MOD, "GaussianNB", "MultinomialNB", "BernoulliNB"),
        md("## 5. PyTorch implementation — vectorized log-probabilities on tensors"),
        show(MOD, "GaussianNBTorch"),
        md("## 6. Train — Gaussian NB on iris; Multinomial/Bernoulli on bag-of-words"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — Gaussian NB decision regions & smoothing effect

Left: Gaussian NB on two iris features — note the smooth, quadratic class
boundaries (each class is an axis-aligned Gaussian). Right: per-word
$P(\text{word}\mid\text{class})$ for the toy corpus, showing Laplace smoothing
keeping unseen words nonzero.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import naive_bayes as M

iris = load_iris()
X2, y = iris.data[:, [0, 2]], iris.target          # 2 features for plotting
g = M.GaussianNB().fit(X2, y)

xx, yy = np.meshgrid(np.linspace(X2[:,0].min()-.5, X2[:,0].max()+.5, 300),
                     np.linspace(X2[:,1].min()-.5, X2[:,1].max()+.5, 300))
Z = g.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].contourf(xx, yy, Z, alpha=.3, cmap="viridis")
ax[0].scatter(X2[:,0], X2[:,1], c=y, cmap="viridis", s=18, edgecolors="k", linewidths=.3)
ax[0].set_xlabel(iris.feature_names[0]); ax[0].set_ylabel(iris.feature_names[2])
ax[0].set_title("Gaussian NB decision regions")

X, yt, vocab = M._toy_text()
mnb = M.MultinomialNB(alpha=1.0).fit(X, yt)
probs = np.exp(mnb.feature_log_prob_)              # (2 classes, vocab)
w = np.arange(len(vocab)); width = .38
ax[1].bar(w - width/2, probs[0], width, label="sports")
ax[1].bar(w + width/2, probs[1], width, label="tech")
ax[1].set_xticks(w); ax[1].set_xticklabels(vocab, rotation=30)
ax[1].set_ylabel("P(word | class)"); ax[1].set_title("Multinomial NB (Laplace-smoothed)")
ax[1].legend()
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Pick the event model for the data**: continuous → Gaussian; word *counts* →
  Multinomial; binary *occurrence* → Bernoulli. They differ in $P(x_j\mid c)$.
- **Always log-space + log-sum-exp**: products of many small probabilities
  underflow to 0; sums of logs don't.
- **Always smooth** ($\alpha>0$): one unseen feature otherwise zeroes a class.
- Probabilities are typically **poorly calibrated** (the independence assumption
  double-counts correlated features → overconfident); trust the *ranking/argmax*,
  not the exact $P$. Calibrate (Platt/isotonic) if you need real probabilities.
- It is linear in $\log$-space and trains in one pass — an excellent fast,
  low-variance baseline, especially for high-dimensional sparse text.
"""),
    ]

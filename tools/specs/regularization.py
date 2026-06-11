from tools.nbreg import register, md, code, show, run_demo

MOD = "regularization"


@register("regularization", "02.dl/regularization/regularization.ipynb")
def build():
    return [
        md(r"""
# Regularization — Dropout, BatchNorm, LayerNorm, weight decay & friends

> Tutorial pair for [`regularization.py`](regularization.py).
> This is the **canonical home** for Dropout, BatchNorm and LayerNorm
> (see `06.training-techniques/README.md`).

## 1. Intuition
A network with enough parameters can memorize its training set perfectly and still
fail on new data — that gap is **overfitting**. Regularization deliberately makes
the training objective a little harder (penalize big weights, randomly drop units,
normalize activations, smooth the targets) so the model is forced to find a
*simpler*, more general solution.
"""),
        md(r"""
## 2. Concept (the slide)
- **L2 / L1 weight decay:** add a penalty on weight magnitude → smaller weights →
  smoother function. L1 also induces *sparsity* (exact zeros).
- **Dropout:** randomly zero units during training → an implicit ensemble of
  sub-networks; no single unit can be relied upon.
- **BatchNorm / LayerNorm:** re-center & re-scale activations so each layer sees a
  stable distribution → faster, more stable training (and a mild regularizer).
- **Early stopping:** monitor a validation metric and halt when it stops improving.
- **Label smoothing:** replace hard one-hot targets with slightly soft ones →
  less over-confident, better-calibrated models.
"""),
        md(r"""
## 3. Math derivation

**L2 (ridge).** Add $\frac{\lambda}{2}\lVert W\rVert_2^2$ to the loss; gradient
$\lambda W$. The update becomes $W\leftarrow W-\eta(\nabla_W\mathcal L+\lambda W)
=(1-\eta\lambda)W-\eta\nabla_W\mathcal L$ — literally *decaying* the weights each step.

**L1 (lasso).** Add $\lambda\lVert W\rVert_1$; subgradient $\lambda\,\mathrm{sign}(W)$.
The constant pull toward 0 sends small weights *exactly* to 0 → sparsity.

**Dropout (inverted).** With keep-probability $q=1-p$, draw a mask
$m_i\sim\mathrm{Bernoulli}(q)/q$ and output $y=m\odot x$. Because
$\mathbb E[m_i]=1$, the expected activation is unchanged, so **test time needs no
rescaling** (just $y=x$). Backward: $m$ is constant w.r.t. $x$, so
$\frac{\partial \mathcal L}{\partial x}=m\odot\frac{\partial \mathcal L}{\partial y}$.

**BatchNorm.** Over a batch of $N$, per feature:
$$\mu=\tfrac1N\textstyle\sum_n x_n,\quad \sigma^2=\tfrac1N\sum_n (x_n-\mu)^2,\quad
  \hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y=\gamma\hat x+\beta.$$
The backward pass must account for $\mu,\sigma^2$ depending on **every** $x_n$:
$$\frac{\partial\mathcal L}{\partial\hat x}=\mathrm{d}y\,\gamma,\quad
  \frac{\partial\mathcal L}{\partial\sigma^2}=\sum_n \mathrm{d}\hat x_n\,(x_n-\mu)\cdot
  \big(-\tfrac12\big)(\sigma^2+\epsilon)^{-3/2},$$
$$\frac{\partial\mathcal L}{\partial\mu}=\sum_n \frac{-\mathrm{d}\hat x_n}{\sqrt{\sigma^2+\epsilon}}
   +\frac{\partial\mathcal L}{\partial\sigma^2}\cdot\frac{1}{N}\sum_n -2(x_n-\mu),$$
$$\frac{\partial\mathcal L}{\partial x_n}=\frac{\mathrm{d}\hat x_n}{\sqrt{\sigma^2+\epsilon}}
   +\frac{\partial\mathcal L}{\partial\sigma^2}\frac{2(x_n-\mu)}{N}
   +\frac{\partial\mathcal L}{\partial\mu}\frac1N.$$
At **test** time use running averages of $\mu,\sigma^2$ (an EMA collected in training).

**LayerNorm.** Same formulas but the statistics are over the **feature** axis,
*per sample* — so there is no batch dependence and no train/test difference (this
is why Transformers use it). The backward collapses to the compact form
$$\mathrm{d}x=\frac{1}{\sigma}\Big(\mathrm{d}\hat x-\overline{\mathrm{d}\hat x}
   -\hat x\,\overline{\mathrm{d}\hat x\odot\hat x}\Big),$$
with bars denoting means over the feature axis and $\mathrm{d}\hat x=\mathrm{d}y\,\gamma$.

**Label smoothing.** Replace the one-hot target with
$q_k=(1-\varepsilon)[k=y]+\frac{\varepsilon}{K-1}[k\ne y]$ and minimize the
cross-entropy $-\sum_k q_k\log p_k$. The optimum logits are finite (not $\pm\infty$),
preventing over-confidence.
"""),
        md("## 4. NumPy implementation — Dropout / BatchNorm / LayerNorm (+ penalties)"),
        show(MOD, "Dropout", "BatchNorm1d", "LayerNorm", "label_smoothing_targets",
             "EarlyStopping"),
        md("## 5. PyTorch implementation — the idiomatic layers & options"),
        show(MOD, "RegMLPTorch"),
        md("## 6. Run — verify norm backprop, then SHOW overfitting shrink"),
        run_demo(MOD),
        md(r"""
## 7. Visualization — train vs test accuracy across regularizers

The bars make the trade-off concrete: stronger regularization usually *lowers*
training accuracy but *raises* test accuracy, narrowing the overfitting gap.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import regularization as M
from sklearn.datasets import load_digits

d = load_digits(); Xall = d.data/16.0; yall = d.target.copy()
rng = np.random.default_rng(0); perm = rng.permutation(len(Xall))
Xall, yall = Xall[perm], yall[perm]
Xtr, ytr = Xall[:80].copy(), yall[:80].copy()
Xte, yte = Xall[200:700], yall[200:700]
noisy = rng.random(len(ytr)) < 0.20; ytr[noisy] = rng.integers(0, 10, noisy.sum())

configs = [("none", dict(l2=0,dropout=0)), ("L2", dict(l2=5e-3,dropout=0)),
           ("dropout", dict(l2=0,dropout=0.6)), ("drop+L2", dict(l2=5e-3,dropout=0.6))]
tr_acc, te_acc = [], []
for _, kw in configs:
    net = M.RegMLP(64, 256, 10, lr=0.2, seed=0, **kw).fit(Xtr, ytr, epochs=200, batch=16)
    tr_acc.append(net.acc(Xtr, ytr)); te_acc.append(net.acc(Xte, yte))

x = np.arange(len(configs)); w = 0.38
plt.figure(figsize=(7, 4))
plt.bar(x - w/2, tr_acc, w, label="train")
plt.bar(x + w/2, te_acc, w, label="test")
plt.xticks(x, [c[0] for c in configs]); plt.ylim(0, 1.05); plt.ylabel("accuracy")
plt.title("Regularization narrows the train/test gap"); plt.legend(); plt.grid(True, axis="y", alpha=.3)
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Dropout** uses *inverted* scaling so inference is just identity — and remember
  to switch to eval mode (`model.eval()` / `train=False`) at test time.
- **BatchNorm** couples samples in a batch (its backward sums over the batch) and
  needs *running stats* at test time; it struggles with tiny batches → use
  **LayerNorm** (per-sample) in Transformers / RNNs.
- **Weight decay** ≈ L2; with adaptive optimizers prefer *decoupled* decay (AdamW,
  see `02.dl/optimizers/optimizers.ipynb`).
- **Early stopping** is the cheapest regularizer — always keep a validation split.
- **Label smoothing** trades a touch of accuracy for much better calibration.
- These compose; don't stack so much regularization that the model underfits.
"""),
    ]

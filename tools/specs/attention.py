from tools.nbreg import register, md, code, show, run_demo

MOD = "attention"


@register("attention", "transformers/attention/attention.ipynb")
def build():
    return [
        md(r"""
# Attention — content-based lookup over a sequence

> Tutorial pair for [`attention.py`](attention.py).

## 1. Intuition
Each position emits a **query** ("what am I looking for?"); every position offers
a **key** ("what do I contain?") and a **value** ("what I'll pass on"). A position's
output is a weighted average of all values, weighted by how well its query matches
each key. Unlike an RNN, this is a **direct, parallel** path between any two
positions — no information has to survive many recurrent steps.
"""),
        md(r"""
## 2. Concept (the slide)
- **Scaled dot-product attention:** $\text{Attn}(Q,K,V)=\operatorname{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}\big)V$.
- **Self-attention:** $Q,K,V$ all come from the same sequence.
- **Cross-attention:** $Q$ from one sequence, $K,V$ from another (decoder ↔ encoder).
- **Multi-head:** run $h$ attentions in parallel subspaces, concat, project — many
  relations at once.
- **Masking:** add $-\infty$ to forbidden scores (causal for autoregression, padding for variable lengths).
"""),
        md(r"""
## 3. Math derivation

**The operation.** With $Q\in\mathbb R^{L_q\times d_k}$, $K\in\mathbb R^{L_k\times d_k}$,
$V\in\mathbb R^{L_k\times d_v}$:
$$S=\frac{QK^\top}{\sqrt{d_k}}\in\mathbb R^{L_q\times L_k},\quad
  A=\operatorname{softmax}(S)\ \text{(row-wise)},\quad
  \text{out}=AV.$$
Row $i$ of $A$ is a probability distribution over positions, so output $i$ is a
convex combination of the value vectors.

**Why divide by $\sqrt{d_k}$?** If query/key entries are independent with unit
variance, the dot product $q\!\cdot\!k=\sum_{j=1}^{d_k}q_jk_j$ has variance $d_k$.
Large logits push softmax into saturated regions where its Jacobian
$\operatorname{diag}(a)-aa^\top$ is tiny → **vanishing gradients**. Scaling by
$1/\sqrt{d_k}$ restores unit variance and healthy gradients.

**Multi-head.** Split $d_{\text{model}}$ into $h$ heads of size $d_k=d_{\text{model}}/h$:
$$\text{head}_i=\text{Attn}(QW_i^Q,KW_i^K,VW_i^V),\quad
  \text{MHA}=[\text{head}_1;\dots;\text{head}_h]\,W^O.$$
Each head can specialize (syntax, coreference, position…), and the cost matches a
single full-width attention.

**Masking.** Add an additive mask $M$ (0 keep, $-\infty$ block) *before* softmax:
$A=\operatorname{softmax}(S+M)$. A causal (upper-triangular $-\infty$) mask makes
position $i$ attend only to $\le i$ — required for left-to-right generation.

**Complexity.** $O(L^2 d)$ time and $O(L^2)$ memory — quadratic in sequence
length (the motivation for efficient-attention research).
"""),
        md("## 4. NumPy implementation — the primitive + multi-head + masks"),
        show(MOD, "scaled_dot_product_attention", "MultiHeadAttentionNumPy"),
        md("## 5. PyTorch implementation"),
        show(MOD, "MultiHeadAttentionTorch"),
        md("## 6. Run — self, causal, cross attention; shapes & mask check"),
        run_demo(MOD),
        md("## 7. Visualization — an attention weight matrix"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
import attention as M

np.random.seed(1)
L, d, h = 8, 16, 4
x = np.random.randn(1, L, d).astype("float32")
mha = M.MultiHeadAttentionNumPy(d, h)
mha(x, x, x, mask=M.causal_mask(L))               # causal self-attention

fig, axes = plt.subplots(1, h, figsize=(13, 3.2))
for i, ax in enumerate(axes):
    ax.imshow(mha.weights[0, i], cmap="viridis")
    ax.set_title(f"head {i}"); ax.set_xlabel("key pos"); ax.set_ylabel("query pos")
plt.suptitle("Causal attention: lower-triangular (no peeking ahead)")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways
- Attention = softmax-weighted average of values, keyed by query·key similarity.
- The $1/\sqrt{d_k}$ scale is essential for stable softmax gradients.
- Multi-head = several relations in parallel; masking controls who sees whom.
- Stack attention + FFN + residual/LayerNorm → the
  **[Transformer](../architectures/transformer.ipynb)**.
"""),
    ]

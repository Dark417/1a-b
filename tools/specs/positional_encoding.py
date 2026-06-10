from tools.nbreg import register, md, code, show, run_demo

MOD = "positional_encoding"


@register("positional_encoding", "transformers/attention/positional_encoding.ipynb")
def build():
    return [
        md(r"""
# Positional Encodings — sinusoidal, learned, RoPE, ALiBi

> Tutorial pair for [`positional_encoding.py`](positional_encoding.py). Read
> [`attention.ipynb`](attention.ipynb) first.

## 1. Intuition
Self-attention is **permutation-equivariant**: if you shuffle the input tokens,
the outputs simply shuffle the same way. Attention by itself has *no idea* which
token came first. We must therefore *inject position*. Four classic recipes:

- **Sinusoidal** — add fixed sine/cosine waves of geometrically-spaced
  wavelengths (no parameters).
- **Learned** — a trainable lookup table indexed by position (BERT/GPT).
- **RoPE** — *rotate* the query/key vectors by a position-dependent angle so that
  their dot product depends only on the **relative** offset.
- **ALiBi** — add a distance-proportional **negative bias** to attention scores;
  no embeddings at all.
"""),
        md(r"""
## 2. Concept (the slide)
- **Absolute** schemes (sinusoidal, learned) add a per-position vector to the
  token embedding *before* attention.
- **Relative** schemes (RoPE, ALiBi) act *inside* attention and encode the
  distance $i-j$ between query $i$ and key $j$ directly — which tends to
  **extrapolate** to longer sequences than seen in training.
- RoPE injects relative position by **rotation**; ALiBi by an additive **linear
  bias**. Learned encodings cannot extrapolate past `max_len`; sinusoidal can in
  principle but in practice degrades.
"""),
        md(r"""
## 3. Math derivation

### 3.1 Sinusoidal (Vaswani et al. 2017)
For position $pos$ and dimension index $i$,
$$PE_{(pos,2i)}=\sin\!\Big(\tfrac{pos}{10000^{2i/d}}\Big),\qquad
  PE_{(pos,2i+1)}=\cos\!\Big(\tfrac{pos}{10000^{2i/d}}\Big).$$
Dimension pair $i$ is a sinusoid of angular frequency
$\omega_i = 10000^{-2i/d}$, with wavelengths growing geometrically from $2\pi$ to
$10000\cdot 2\pi$. The key property: a shift by $k$ is a **fixed linear map**.
Writing $\theta = \omega_i\, pos$,
$$\begin{bmatrix}\sin\omega_i(pos{+}k)\\ \cos\omega_i(pos{+}k)\end{bmatrix}
=\underbrace{\begin{bmatrix}\cos\omega_i k & \sin\omega_i k\\ -\sin\omega_i k & \cos\omega_i k\end{bmatrix}}_{\text{depends only on }k}
\begin{bmatrix}\sin\omega_i\,pos\\ \cos\omega_i\,pos\end{bmatrix},$$
so attention can implement "attend $k$ positions back" with a position-independent
transform.

### 3.2 Learned absolute
Replace $PE$ with a trainable matrix $E_{\text{pos}}\in\mathbb{R}^{L_{\max}\times d}$:
$x_t \leftarrow x_t + E_{\text{pos}}[t]$. Maximally flexible, but undefined for
$t \ge L_{\max}$ (no extrapolation).

### 3.3 RoPE — relative position via rotation (Su et al. 2021)
Split the $d$-dim vector into $d/2$ pairs. For pair $j$ use frequency
$\theta_j = base^{-2j/d}$ and rotate the pair of query/key components at position
$m$ by angle $m\theta_j$:
$$R(m)_j=\begin{bmatrix}\cos m\theta_j & -\sin m\theta_j\\ \sin m\theta_j & \cos m\theta_j\end{bmatrix}.$$
Let $\tilde q_m = R(m)q$, $\tilde k_n = R(n)k$. Because rotations compose,
$R(m)^\top R(n) = R(n-m)$, the score is
$$\langle \tilde q_m, \tilde k_n\rangle
= q^\top R(m)^\top R(n)\, k
= q^\top R(n-m)\, k,$$
which depends only on the **relative offset** $n-m$. Absolute rotation, relative
effect. In code we use the "rotate-half" identity
$R\,x = x\odot\cos + \text{rot}(x)\odot\sin$ where
$\text{rot}(x)=(-x_1,x_0,-x_3,x_2,\dots)$.

### 3.4 ALiBi — Attention with Linear Biases (Press et al. 2022)
Add, per head $h$, a bias proportional to the query–key distance *before* softmax:
$$\text{score}_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}} \; - \; m_h\,(i-j),\quad j\le i,$$
with head-specific slopes forming a geometric sequence $m_h = 2^{-8h/H}$. No
learned parameters and no embeddings; distant keys are simply penalized more, and
different heads get different effective context windows. Excellent length
extrapolation.
"""),
        md("## 4. Key component — the four NumPy formulas + the rotary module"),
        show(MOD, "sinusoidal_encoding", "rope_angles", "apply_rope_numpy",
             "alibi_bias", "RotaryPositionalEncoding"),
        md("## 5. Full model — a position-aware attention block (any scheme)"),
        show(MOD, "PosAwareAttention"),
        md("## 6. Train / run — verify the math + a tiny position-sensitive task"),
        code("import torch; torch.set_num_threads(1)  # tiny CPU demo: avoid thread oversubscription\n"
             f"import {MOD} as M\nM.demo()"),
        md("## 7. Visualization — sinusoidal heatmap, RoPE rotation, ALiBi bias"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import positional_encoding as M

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# (a) sinusoidal positional-encoding heatmap
pe = M.sinusoidal_encoding(80, 64)
im = ax[0].imshow(pe.T, aspect="auto", cmap="RdBu")
ax[0].set_xlabel("position"); ax[0].set_ylabel("dimension")
ax[0].set_title("Sinusoidal PE"); fig.colorbar(im, ax=ax[0])

# (b) RoPE rotation: a unit vector rotated by increasing position angle
ang = M.rope_angles(12, 2)[:, 0]                 # angle of pair 0 vs position
xs, ys = np.cos(ang), np.sin(ang)
ax[1].quiver(np.zeros_like(xs), np.zeros_like(ys), xs, ys,
             ang, angles="xy", scale_units="xy", scale=1, cmap="viridis")
ax[1].set_xlim(-1.2, 1.2); ax[1].set_ylim(-1.2, 1.2); ax[1].set_aspect("equal")
ax[1].set_title("RoPE: position = rotation angle")

# (c) ALiBi additive bias for head 0 (causal): distant keys penalized
bias = M.alibi_bias(4, 12, causal=True)[0]
bias = np.where(bias < -1e8, np.nan, bias)        # hide masked future
im2 = ax[2].imshow(bias, cmap="magma")
ax[2].set_xlabel("key pos j"); ax[2].set_ylabel("query pos i")
ax[2].set_title("ALiBi bias (head 0)"); fig.colorbar(im2, ax=ax[2])

plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Attention is order-blind; **some** positional signal is mandatory.
- **Absolute** (sinusoidal / learned) adds a per-position vector; **relative**
  (RoPE / ALiBi) encodes the offset $i-j$ inside attention and extrapolates far
  better to unseen lengths.
- **RoPE** is the modern default (LLaMA, GPT-NeoX): relative position falls out of
  composing rotations, $R(m)^\top R(n)=R(n-m)$.
- **ALiBi** is parameter-free and trivially length-extrapolating, at the cost of a
  fixed distance prior.
- Pitfalls: learned encodings break past `max_len`; RoPE needs an **even** feature
  dim and is applied to $Q,K$ only (never $V$); ALiBi slopes must be tuned to head
  count or some heads become near-positional and others near-global.
"""),
    ]

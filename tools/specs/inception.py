from tools.nbreg import register, md, code, show, run_demo

MOD = "inception"


@register("inception", "02.dl/cnn/inception.ipynb")
def build():
    return [
        md(r"""
# Inception / GoogLeNet — go wider, not just deeper

> Tutorial pair for [`inception.py`](inception.py).

## 1. Intuition
Why pick *one* filter size when you can try several? The **Inception module**
runs 1x1, 3x3, 5x5 convolutions and a pooling branch **in parallel** and
**concatenates** their outputs, letting the network learn at multiple scales at
once. The danger is cost: a 5x5 conv over hundreds of input channels is huge. The
cure is the **1x1 "bottleneck" convolution**, which cheaply shrinks channel depth
*before* the expensive spatial conv.
"""),
        md(r"""
## 2. Concept (the slide)
- **Multi-branch module:** parallel $1\times1$, $3\times3$, $5\times5$, and a
  $3\times3$ max-pool branch; all padded to the same spatial size and
  **concatenated on the channel axis**.
- **1x1 convolution = a per-pixel fully-connected layer across channels.** It
  mixes channels and can *reduce* their number with almost no spatial cost.
- **Bottleneck placement:** put a $1\times1$ *reduce* before the $3\times3$ and
  $5\times5$ branches, and a $1\times1$ *project* after the pool branch.
- **GoogLeNet** = a stack of these modules + global average pooling head.
"""),
        md(r"""
## 3. Math — parameter savings from 1x1 bottlenecks

A conv with kernel $k$, $C_{in}$ inputs and $C_{out}$ outputs costs $k^2C_{in}C_{out}$
weights. Take the **5x5 branch** of GoogLeNet's inception (3a): $C_{in}=192$,
$C_{out}=32$.

**Naive** (direct 5x5):
$$5^2 \cdot 192 \cdot 32 = 25\cdot192\cdot32 = 153{,}600 \text{ weights}.$$

**Bottlenecked** (1x1 reduce $192\to16$, then 5x5 on 16 channels):
$$\underbrace{1^2\cdot192\cdot16}_{\text{reduce}=3072}
  + \underbrace{5^2\cdot16\cdot32}_{\text{spatial}=12800}
  = 15{,}872 \text{ weights}.$$

A **$9.7\times$** reduction on that branch alone. Summed over the whole module the
1x1 bottlenecks cut the parameter count from $\approx 393{\rm k}$ to $\approx 163{\rm k}$
— about **58% fewer**, with negligible loss in accuracy. Intuitively the 1x1 conv
projects the $C_{in}$ channels onto a small informative subspace; the costly
$k^2$ spatial mixing then happens in that low-dimensional space.
"""),
        md("## 4. Key building block — the Inception module (with 1x1 bottlenecks)"),
        show(MOD, "Inception", "inception_param_comparison"),
        md("## 5. Full architecture (PyTorch) — a tiny GoogLeNet"),
        show(MOD, "TinyGoogLeNet"),
        md(r"""
## 6. Run — the bottleneck saving, naive vs reduced shapes, a few training steps
"""),
        run_demo(MOD),
        md(r"""
## 7. Visualization — where the parameters go, naive vs 1x1-bottlenecked

Per-branch parameter counts for one Inception (3a) module. The naive 3x3 and 5x5
branches dominate; adding cheap 1x1 reduces collapses them dramatically.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import inception as M

cmp = M.inception_param_comparison()
naive, reduced = cmp["naive"], cmp["reduced"]
keys = ["1x1", "3x3", "5x5", "pool_proj"]
# fold the *_reduce costs into their branch for a fair side-by-side
red_branch = {
    "1x1": reduced["1x1"],
    "3x3": reduced["3x3_reduce"] + reduced["3x3"],
    "5x5": reduced["5x5_reduce"] + reduced["5x5"],
    "pool_proj": reduced["pool_proj"],
}
x = np.arange(len(keys)); w = 0.38
plt.figure(figsize=(8, 4.2))
plt.bar(x - w/2, [naive[k] for k in keys], w, label=f"naive ({cmp['naive_total']:,})")
plt.bar(x + w/2, [red_branch[k] for k in keys], w, label=f"with 1x1 ({cmp['reduced_total']:,})")
plt.xticks(x, keys); plt.ylabel("parameters (weights)")
plt.title("Inception (3a): 1x1 bottlenecks shrink the 3x3 & 5x5 branches")
plt.legend(); plt.grid(True, axis="y", alpha=.3); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Width + multi-scale:** branch concatenation lets one module see several
  receptive fields; the network picks the mix.
- **1x1 convs are the workhorse:** cheap channel mixing / dimensionality
  reduction. They reappear in ResNet bottlenecks and MobileNet pointwise convs.
- **Pitfall — alignment:** every branch must emit the *same spatial size* (use
  padding) so the channel-concatenation lines up.
- **Pitfall — channel budget:** the reduce widths (`reduce_3`, `reduce_5`) are
  hyperparameters; too small and you bottleneck information, too large and you
  lose the savings.
"""),
    ]

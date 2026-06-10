from tools.nbreg import register, md, code, show, run_demo

MOD = "vgg"


@register("vgg", "dl/cnn/vgg.ipynb")
def build():
    return [
        md(r"""
# VGG — depth from stacked 3x3 convolutions

> Tutorial pair for [`vgg.py`](vgg.py).

## 1. Intuition
VGG made one idea uniform and pushed it deep: build the *entire* network from
tiny **3x3 convolutions** (stride 1, pad 1) interleaved with **2x2 max-pools**.
Why small filters? Because a **stack** of them sees just as far as one big filter
(same **receptive field**) while using **fewer parameters** and inserting **more
nonlinearities** (a ReLU between each conv). Uniform, simple, and very deep.
"""),
        md(r"""
## 2. Concept (the slide)
- **VGG block:** $n$ stacked $3\times3$ conv-ReLU layers (padding keeps spatial
  size), then a $2\times2$ max-pool that halves resolution and doubles channels.
- **Configurations A..E** differ only in how many convs sit in each block — the
  motif is identical.
- **Two $3\times3$ convs $\equiv$ one $5\times5$ receptive field**; three
  $\equiv$ one $7\times7$. The stacked version is cheaper and more expressive.
- **He init** for the ReLU stacks (see `training-techniques/README.md`).
"""),
        md(r"""
## 3. Math — receptive field & parameter count

**Receptive field of a conv stack.** For layer $l$ with kernel $k_l$ and stride
$s_l$, the receptive field grows as
$$\mathrm{RF}_l = \mathrm{RF}_{l-1} + (k_l-1)\prod_{j<l}s_j .$$
With all strides $=1$ this collapses to
$$\mathrm{RF} = 1 + \sum_l (k_l - 1).$$
So **two** $3\times3$ convs give $\mathrm{RF}=1+2+2=5$ — a $5\times5$ field; **three**
give $7$. A single $7\times7$ conv sees the same window in *one* layer.

**Parameter count.** One conv with $C$ in and $C$ out channels and kernel $k$ has
$k^2C^2$ weights. Compare equal-receptive-field designs:
$$\underbrace{2\,(3^2C^2)}_{\text{two }3\times3}=18C^2 \;<\; \underbrace{5^2C^2}_{\text{one }5\times5}=25C^2,
\qquad
\underbrace{3\,(3^2C^2)}_{\text{three }3\times3}=27C^2 \;<\; \underbrace{7^2C^2}_{\text{one }7\times7}=49C^2.$$
That is $18/25 = 0.72$ (28% fewer) and $27/49 = 0.55$ (45% fewer) parameters —
**and** the stacked form applies a ReLU after every conv, so it represents a
strictly richer (more nonlinear) family of functions for the same field of view.
"""),
        md("## 4. Key building block — the VGG 3x3 stack + receptive-field math"),
        show(MOD, "vgg_block", "receptive_field", "small_vs_large_filter_table"),
        md("## 5. Full architecture (PyTorch) — a config-driven small VGG"),
        show(MOD, "VGGMini"),
        md(r"""
## 6. Run — the 3x3-vs-5x5 table, feature-map shapes, a few training steps
"""),
        run_demo(MOD),
        md(r"""
## 7. Visualization — parameters vs receptive field: 3x3 stacks win

For each equal-receptive-field design we plot the parameter count. The $3\times3$
stacks (filled markers) sit strictly *below* the single large convs (hollow) at
the same receptive field.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vgg as M

tbl = M.small_vs_large_filter_table(64)
plt.figure(figsize=(7, 4.2))
for name, (rf, p) in tbl.items():
    stacked = "3x3" in name
    plt.scatter(rf, p, s=140,
                marker="o" if stacked else "s",
                facecolors="C0" if stacked else "none",
                edgecolors="C3" if not stacked else "C0",
                label=name.replace("_", " "))
    plt.annotate(name.replace("_", " "), (rf, p),
                 textcoords="offset points", xytext=(6, 6), fontsize=9)
plt.xlabel("receptive field (pixels)"); plt.ylabel("parameters (C=64, weights)")
plt.title("Same receptive field, fewer params: stacked 3x3 vs one big conv")
plt.grid(True, alpha=.3); plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Small filters, stacked deep** = same receptive field, fewer params, more
  nonlinearity. This is the whole VGG thesis.
- **Uniformity** (everything is 3x3/pool) makes VGG a clean baseline, but it is
  **parameter-heavy in the FC head** — most of real VGG-16's 138M params live in
  the dense layers, not the convs. Global average pooling (used here) fixes that.
- **Pitfall:** very deep *plain* stacks still hit the degradation/vanishing-
  gradient wall — VGG was near the practical limit before residual connections
  (ResNet) made arbitrary depth trainable. See `training-techniques/README.md`.
"""),
    ]

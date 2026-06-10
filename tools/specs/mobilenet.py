from tools.nbreg import register, md, code, show, run_demo

MOD = "mobilenet"


@register("mobilenet", "dl/cnn/mobilenet.ipynb")
def build():
    return [
        md(r"""
# MobileNet — depthwise-separable convolutions for cheap CNNs

> Tutorial pair for [`mobilenet.py`](mobilenet.py).

## 1. Intuition
A standard convolution does two jobs at once: mix information **across space**
(the $k\times k$ window) and **across channels**. MobileNet **factorizes** these:
a **depthwise** conv filters each channel on its own (space only), then a
**pointwise** $1\times1$ conv mixes channels. Doing the jobs separately costs a
fraction of the parameters and FLOPs — the trick that put CNNs on phones.
"""),
        md(r"""
## 2. Concept (the slide)
- **Depthwise conv:** one $k\times k$ filter *per input channel*
  (`groups = in_channels`), no cross-channel summation -> spatial mixing only.
- **Pointwise conv:** a $1\times1$ conv that mixes channels -> channel mixing only.
- **Depthwise-separable block** = depthwise + pointwise, each with BN + ReLU.
- **Width multiplier $\alpha$:** thins every layer uniformly (quadratic cost
  control); a second knob (resolution multiplier) shrinks the input.
"""),
        md(r"""
## 3. Math — the parameter / FLOP reduction

Map a $C_{in}\times H\times W$ feature map to $C_{out}\times H\times W$ with a
$k\times k$ conv ('same' padding).

**Standard conv** (one kernel per output channel sees all input channels):
$$\text{params} = k^2\,C_{in}\,C_{out},\qquad
  \text{FLOPs} = k^2\,C_{in}\,C_{out}\,HW.$$

**Depthwise-separable** = depthwise ($k^2C_{in}$, one filter/channel) + pointwise
($C_{in}C_{out}$, a $1\times1$):
$$\text{params} = k^2 C_{in} + C_{in}C_{out},\qquad
  \text{FLOPs} = \big(k^2 C_{in} + C_{in}C_{out}\big)HW.$$

**The ratio** (separable / standard) is the famous MobileNet formula:
$$\frac{k^2 C_{in} + C_{in}C_{out}}{k^2 C_{in}C_{out}}
  = \frac{1}{C_{out}} + \frac{1}{k^2}.$$
For $k=3$ this is $\approx \tfrac{1}{C_{out}} + \tfrac{1}{9}$ — for any reasonable
$C_{out}$ the cost is dominated by the $1/9$ term, i.e. an **~8–9$\times$**
reduction. Concretely $C_{in}=32,\ C_{out}=64,\ k=3$: $18{,}432$ params vs
$2{,}336$ — about $12.7\%$ of the cost, exactly $1/64 + 1/9$.
"""),
        md("## 4. Key building block — the depthwise-separable conv + the savings math"),
        show(MOD, "DepthwiseSeparableConv", "depthwise_separable_savings"),
        md("## 5. Full architecture (PyTorch) — a tiny MobileNet (with width multiplier)"),
        show(MOD, "TinyMobileNet"),
        md(r"""
## 6. Run — the standard-vs-separable cost, shapes, $\alpha$ thinning, training
"""),
        run_demo(MOD),
        md(r"""
## 7. Visualization — parameters & FLOPs: standard conv vs separable

Side-by-side cost of a standard $3\times3$ conv vs its depthwise-separable
factorization, broken into the depthwise and pointwise pieces. The depthwise part
is almost free; the pointwise $1\times1$ carries most of the (already small) cost.
"""),
        code(r"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
import mobilenet as M

s = M.depthwise_separable_savings(c_in=32, c_out=64, k=3, hw=16)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

# parameters
ax[0].bar(["standard"], [s["standard_params"]], color="C3")
ax[0].bar(["separable"], [s["depthwise_params"]], color="C0", label="depthwise")
ax[0].bar(["separable"], [s["pointwise_params"]], bottom=[s["depthwise_params"]],
          color="C1", label="pointwise")
ax[0].set_title(f"params  (separable = {s['param_ratio']:.1%} of standard)")
ax[0].set_ylabel("weights"); ax[0].legend()

# flops
ax[1].bar(["standard"], [s["standard_flops"]], color="C3")
ax[1].bar(["separable"], [s["separable_flops"]], color="C0")
ax[1].set_title(f"FLOPs  (separable = {s['flop_ratio']:.1%} of standard)")
ax[1].set_ylabel("multiply-adds")

plt.tight_layout(); plt.show()
print("ratio matches 1/C_out + 1/k^2 =", round(s["formula_ratio"], 3))
"""),
        md(r"""
## 8. Takeaways & pitfalls
- **Factorize space and channels** -> cost drops to $\tfrac1{C_{out}}+\tfrac1{k^2}$
  of a full conv, with little accuracy loss.
- The **$1\times1$ pointwise** conv now dominates the FLOPs — the same channel-
  mixing role it plays in Inception and ResNet bottlenecks.
- **Width multiplier $\alpha$** scales cost quadratically ($\alpha^2$): a cheap,
  predictable accuracy/latency dial.
- **Pitfall — under-mixing:** depthwise convs never mix channels, so a separable
  block *needs* the pointwise step; stacking depthwise-only convs cannot combine
  channel information.
- **Pitfall — BN/ReLU placement:** MobileNet-v2 later moved to *linear*
  bottlenecks + inverted residuals (no ReLU on the narrow layer) to avoid
  destroying information in low-dimensional spaces.
"""),
    ]

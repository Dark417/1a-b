from tools.nbreg import register, md, code, show, run_demo

MOD = "cnn"


@register("cnn", "02.dl/cnn/cnn.ipynb")
def build():
    return [
        md(r"""
# Convolutional Neural Networks — weight sharing over space

> Tutorial pair for [`cnn.py`](cnn.py).

## 1. Intuition
A fully-connected layer on a 200×200 image has billions of weights and ignores
geometry. A **convolution** slides a small filter over the image, reusing the
*same* weights everywhere. This gives **translation equivariance**, far fewer
parameters, and a hierarchy of features (edges → textures → parts → objects).
"""),
        md(r"""
## 2. Concept (the slide)
- **Conv layer:** a bank of small filters convolved over the input → feature maps.
- **Local receptive fields + weight sharing** = the two core ideas.
- **Pooling** (max) downsamples for small translation invariance.
- **LeNet** pattern: [Conv→ReLU→Pool] × N → Flatten → Dense → softmax.
"""),
        md(r"""
## 3. Math derivation

**Convolution (cross-correlation) forward.** For input $X$, filter $W$ (size
$k\times k$), output channel $o$:
$$Y_{o,p,q}=b_o+\sum_{c}\sum_{i=0}^{k-1}\sum_{j=0}^{k-1}W_{o,c,i,j}\,X_{c,\,p s+i,\,q s+j}.$$

**im2col trick.** Extract every $k\times k$ patch into a row of a matrix
$\mathrm{col}\in\mathbb R^{(N\cdot O_H\cdot O_W)\times(C k^2)}$. Then the whole
convolution is **one matrix multiply** $Y=\mathrm{col}\,W_{\text{row}}^\top+b$ —
fast and easy to differentiate.

**Backward.** With upstream $\partial\mathcal L/\partial Y=\mathrm{d}Y$ reshaped to
rows,
$$\mathrm{d}W=\mathrm{d}Y^\top\,\mathrm{col},\qquad
  \mathrm{d}b=\textstyle\sum \mathrm{d}Y,\qquad
  \mathrm{d}\,\mathrm{col}=\mathrm{d}Y\,W_{\text{row}}.$$
`col2im` scatters $\mathrm{d}\,\mathrm{col}$ back to image positions, **summing**
overlapping contributions (because each pixel feeds several patches).

**Max-pool backward** routes the gradient only to the position that was the max
(an argmax mask). Output size: $O=\big\lfloor\frac{H+2P-k}{s}\big\rfloor+1$.

**Parameter count.** A conv layer has just $O\cdot C\cdot k^2+O$ weights —
independent of image size, the key efficiency win over dense layers.
"""),
        md("## 4. NumPy implementation — im2col conv, max-pool, full backward"),
        show(MOD, "Conv2D", "MaxPool2D"),
        md("## 5. PyTorch implementation — a LeNet-5"),
        show(MOD, "LeNetTorch"),
        md("## 6. Train on 8×8 digits — NumPy ConvNet vs PyTorch LeNet"),
        run_demo(MOD),
        md("## 7. Visualization — learned first-layer filters & a sample digit"),
        code(r"""
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import cnn as M

d = load_digits(); X = d.images[:,None]/16.0; y = d.target
net = M.ConvNetNumPy(img=8, lr=0.1).fit(X[:1400], y[:1400], epochs=8)

fig, axes = plt.subplots(1, 5, figsize=(12, 2.6))
axes[0].imshow(d.images[0], cmap="gray"); axes[0].set_title(f"digit={y[0]}")
for i in range(4):
    axes[i+1].imshow(net.conv.W[i,0], cmap="coolwarm"); axes[i+1].set_title(f"filter {i}")
for a in axes: a.axis("off")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- Convolution = local connectivity + weight sharing → few params, equivariance.
- im2col turns conv into a matmul (how real frameworks do it under the hood).
- Very deep CNNs still hit vanishing gradients → **BatchNorm** and **residual
  connections** (ResNet, next file) fix it — see `06.training-techniques/README.md`.
"""),
    ]

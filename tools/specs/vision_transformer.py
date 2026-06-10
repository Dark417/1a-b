from tools.nbreg import register, md, code, show, run_demo

MOD = "vision_transformer"


@register("vision_transformer", "transformers/architectures/vision_transformer.ipynb")
def build():
    return [
        md(r"""
# Vision Transformer (ViT) — an image is a sequence of patches

> Tutorial pair for [`vision_transformer.py`](vision_transformer.py). Read
> [`transformer.ipynb`](transformer.ipynb) and
> [`attention.ipynb`](../attention/attention.ipynb) first.

## 1. Intuition
CNNs bake in locality and translation-equivariance. ViT throws almost all of that
away: chop the image into a grid of non-overlapping **patches**, flatten each
patch into a vector, linearly project it to a token, and feed the resulting
**sequence of patch-tokens** to a vanilla Transformer encoder. A learnable
**[CLS]** token collects global information and its final state is classified.
With enough data, self-attention learns the spatial relationships a CNN hard-codes.
"""),
        md(r"""
## 2. Concept (the slide)
- **Patchify:** an $H\times W$ image becomes $N=(H/P)(W/P)$ patches of size
  $C\cdot P\cdot P$; a linear map sends each to a $d$-dim token (a Conv2d with
  `kernel=stride=P` is the same operation).
- **[CLS] token:** a learnable vector prepended to the sequence; its output feeds
  the classifier.
- **Positional embeddings:** learned, added to every token (attention is
  order-agnostic, and patch order = spatial layout).
- **Encoder + head:** standard pre-norm Transformer blocks, then a linear head on
  the [CLS] representation.
"""),
        md(r"""
## 3. Math derivation

### 3.1 Patch embedding
Let $x\in\mathbb{R}^{C\times H\times W}$ with $H=W=NP$. Reshape into patches
$x_p^{(k)}\in\mathbb{R}^{C\cdot P\cdot P}$ for $k=1,\dots,N$ where $N=(H/P)(W/P)$,
taken in raster order. A learnable matrix $E\in\mathbb{R}^{(C P^2)\times d}$ and a
learnable [CLS] vector $x_{\text{cls}}$ form the input sequence:
$$z_0 = \big[\,x_{\text{cls}}\,;\; x_p^{(1)}E\,;\;\dots\,;\; x_p^{(N)}E\,\big]
        \;+\; E_{\text{pos}},\qquad
  E_{\text{pos}}\in\mathbb{R}^{(N+1)\times d}.$$
A convolution with kernel size and stride both $P$ computes the $N$ projections
$x_p^{(k)}E$ simultaneously — it *is* the patch embedding.

### 3.2 Sequence-of-patches view through the encoder
The token matrix $z_0\in\mathbb{R}^{(N+1)\times d}$ is processed by $L$ pre-norm
Transformer blocks:
$$z'_\ell = z_{\ell-1} + \text{MHSA}\big(\text{LN}(z_{\ell-1})\big),\qquad
  z_\ell  = z'_\ell + \text{MLP}\big(\text{LN}(z'_\ell)\big).$$
Self-attention is **global from layer 1**: any patch can attend to any other,
unlike a CNN whose receptive field grows slowly with depth. Crucially, attention
is permutation-equivariant, so the positional embeddings $E_{\text{pos}}$ are what
carry the 2-D spatial layout.

### 3.3 Classification head
Read off the final [CLS] token $z_L^{(0)}$, normalize, and apply a linear
classifier:
$$\hat y = \operatorname{softmax}\!\big(W\,\text{LN}(z_L^{(0)}) + b\big),\qquad
  \mathcal{L} = -\log \hat y_{[\text{true class}]}.$$
The [CLS] token has no patch content of its own; through attention it learns to
*pool* whatever patch information the task needs.
"""),
        md("## 4. Key component — patchify (NumPy) + the Conv2d patch embedding"),
        show(MOD, "patchify_numpy", "PatchEmbed"),
        md("## 5. Full model — VisionTransformer ([CLS] + pos-emb + encoder + head)"),
        show(MOD, "VisionTransformer"),
        md("## 6. Train / run — tiny 8x8 shapes split into 4x4 patches"),
        run_demo(MOD),
        md("## 7. Visualization — patch grid & the classification loss curve"),
        code(r"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, torch, matplotlib.pyplot as plt
import vision_transformer as M

torch.manual_seed(M.SEED); np.random.seed(M.SEED)

size, patch = 8, 4
X_np, y_np = M.make_toy_images(384, size)

# (a) show one image and its 2x2 grid of 4x4 patches
img = X_np[0, 0]
seq = M.patchify_numpy(X_np[0], patch)            # (4, 16) -> 4 patches
ncol = size // patch

# (b) train a few steps for the loss curve
X = torch.tensor(X_np); y = torch.tensor(y_np)
model = M.VisionTransformer(1, size, patch, 3, d_model=48, n_heads=4,
                            d_ff=96, n_layers=2)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
lossfn = torch.nn.CrossEntropyLoss()
losses = []
for _ in range(120):
    logits = model(X)
    loss = lossfn(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

fig = plt.figure(figsize=(12, 4))
ax0 = fig.add_subplot(1, 3, 1)
ax0.imshow(img, cmap="gray"); ax0.set_title("input 8x8 image")
for k in range(ncol):
    ax0.axhline(k * patch - 0.5, color="r"); ax0.axvline(k * patch - 0.5, color="r")
ax1 = fig.add_subplot(1, 3, 2)
ax1.imshow(seq, aspect="auto", cmap="gray")
ax1.set_title("4 patches x 16 dims"); ax1.set_xlabel("flattened pixel"); ax1.set_ylabel("patch")
ax2 = fig.add_subplot(1, 3, 3)
ax2.plot(losses); ax2.set_xlabel("step"); ax2.set_ylabel("cross-entropy")
ax2.set_title("Classification loss going down")
plt.tight_layout(); plt.show()
"""),
        md(r"""
## 8. Takeaways & pitfalls
- ViT reframes vision as sequence modeling: **patches are tokens**, and a plain
  Transformer encoder does the rest.
- The patch embedding is just a strided convolution; the [CLS] token and **learned
  positional embeddings** supply aggregation and spatial layout.
- Self-attention is **global from the first layer** (no slowly-growing receptive
  field), but it has weak inductive bias, so ViT is data-hungry — small datasets
  favor CNNs or heavy augmentation/distillation.
- Pitfalls: image size must be divisible by patch size; positional embeddings are
  tied to a fixed grid (changing resolution needs interpolation); forgetting to
  prepend [CLS] (or pooling patches instead) changes the head.
- Same encoder as **[BERT](bert.ipynb)**; the novelty is purely the input
  representation.
"""),
    ]

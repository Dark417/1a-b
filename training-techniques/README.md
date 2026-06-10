# Training Techniques (cross-cutting)

Training techniques are **not** standalone algorithm files. They are the tricks
that make models actually learn, and they are demonstrated *inside* the
algorithms where they naturally arise. The **same technique reappears across
several algorithms** — that repetition is deliberate in a tutorial.

This page is the reference write-up. Each algorithm that uses a technique links
back here instead of re-deriving it.

| Technique | Canonical demo | Also appears in |
|---|---|---|
| Vanishing / exploding gradients | `dl/rnn/rnn` | `dl/rnn/lstm`, `dl/cnn/resnet`, `dl/mlp/mlp` |
| Gradient clipping | `dl/rnn/rnn` | `transformers/architectures/transformer` |
| Weight initialization (Xavier/He) | `dl/mlp/mlp` | every DL model |
| Batch / Layer normalization | `dl/regularization/regularization` | `dl/cnn/resnet`, `transformers` |
| Dropout | `dl/regularization/regularization` | `dl/mlp/mlp`, `transformers` |
| Skip / residual connections | `dl/cnn/resnet` | `transformers/architectures/transformer` |
| LR scheduling / warmup | `transformers/architectures/transformer` | `dl/optimizers/optimizers` |
| Reparameterization trick | `generative-models/vae/vae` | diffusion |
| Adversarial / minimax training | `generative-models/gan/vanilla_gan` | all GAN variants |
| Negative sampling | `nlp/embeddings/word2vec` | — |
| Teacher forcing | `nlp/seq2seq/seq2seq_attention` | `dl/rnn/lstm` |

---

## Vanishing / exploding gradients

In a deep or recurrent net the gradient is a **product** of many Jacobians.
For an RNN unrolled $T$ steps,

$$\frac{\partial \mathcal{L}}{\partial h_t}
   = \frac{\partial \mathcal{L}}{\partial h_T}
     \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}},
   \qquad
   \frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}\big(\sigma'(\cdot)\big)\,W^\top .$$

If the largest singular value of $W$ (times $\sigma'$) is $<1$, the product
**shrinks geometrically** → *vanishing* gradients (early layers stop learning).
If it is $>1$, the product **blows up** → *exploding* gradients (NaNs).

**Fixes shown in the repo**
- Non-saturating activations (ReLU/GELU) — `dl/mlp`.
- Careful initialization (Xavier/He) — `dl/mlp`.
- Gradient **clipping** for explosion — `dl/rnn`.
- Gated memory (LSTM/GRU) creating a near-identity path — `dl/rnn/lstm`.
- **Residual** connections (identity shortcut, gradient ≈ 1) — `dl/cnn/resnet`.
- **Normalization** layers keeping activations well-scaled — `dl/regularization`.

## Gradient clipping
Rescale the gradient when its norm exceeds a threshold $\tau$:
$$g \leftarrow g \cdot \min\!\Big(1, \frac{\tau}{\lVert g\rVert}\Big).$$
Cheap, and it tames exploding gradients without changing the descent direction.

## Weight initialization
Keep the variance of activations/gradients constant across layers.
- **Xavier/Glorot** (tanh/sigmoid): $\operatorname{Var}(W)=\dfrac{2}{n_{in}+n_{out}}$.
- **He/Kaiming** (ReLU): $\operatorname{Var}(W)=\dfrac{2}{n_{in}}$.

## Batch / Layer normalization
Normalize a layer's pre-activations, then learn a scale/shift $\gamma,\beta$:
$$\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad y=\gamma\hat{x}+\beta.$$
BatchNorm normalizes over the batch (per feature); LayerNorm over features (per
sample) — the latter is what Transformers use.

## Dropout
During training, zero each unit independently with probability $p$ and scale the
rest by $1/(1-p)$ (inverted dropout). Acts as an ensemble over sub-networks and
a strong regularizer.

## Skip / residual connections
Learn $F(x)$ and output $x+F(x)$. The gradient gets a direct $+1$ path, so deep
stacks stay trainable — the core fix for vanishing gradients in very deep nets.

## LR scheduling / warmup
Transformers use the "noam" schedule: linearly warm up then decay
$\propto \text{step}^{-0.5}$. Warmup avoids destabilizing the freshly-initialized
attention; decay anneals to a good minimum.

## Reparameterization trick
To backprop through a sample $z\sim\mathcal{N}(\mu,\sigma^2)$, write
$z=\mu+\sigma\odot\epsilon,\ \epsilon\sim\mathcal{N}(0,I)$ so the randomness is
an input and gradients flow to $\mu,\sigma$. Powers the VAE.

## Adversarial / minimax training
Two networks play a game
$\min_G\max_D \; \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$.
Unstable by nature → motivates non-saturating loss, WGAN, gradient penalty.

## Negative sampling
Replace an expensive softmax over the vocabulary with a few binary
logistic-regression problems against sampled "negative" words — makes word2vec
tractable.

## Teacher forcing
When training a sequence decoder, feed the *ground-truth* previous token instead
of the model's own prediction. Speeds convergence; mind the train/inference gap
("exposure bias").

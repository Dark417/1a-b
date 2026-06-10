# MAP — Master Catalogue of Algorithms

> This is the **source of truth** for the repository. Every algorithm listed
> here should eventually have:
>
> 1. a `<name>.py` file (plain **NumPy** *and* **PyTorch** implementations), and
> 2. a `<name>.ipynb` notebook (same code + concept explanation + math derivation).
>
> See [`AGENTS.md`](AGENTS.md) for the exact format and rules each file must follow.
>
> **Status legend:** `[ ]` planned · `[~]` scaffolded · `[x]` implemented (py + ipynb)

---

## How to read this map

```
category/
  sub-category/            # optional extra layer when a family is large
    algorithm.py           # numpy + pytorch
    algorithm.ipynb        # concept + math + code
```

A **training technique** (e.g. *vanishing gradients*, *gradient clipping*,
*batch norm*) is **not** a standalone file. It is demonstrated *inside* the
algorithms where it naturally appears, and the same technique may reappear in
several algorithms. The cross-reference table lives in
[`training-techniques/README.md`](training-techniques/README.md).

---

## 1. `ml/` — Classic Machine Learning

### 1.1 `linear-models/`
- [x] `linear_regression` — OLS, **variants:** Ridge (L2), Lasso (L1), ElasticNet, polynomial features
- [x] `logistic_regression` — binary, **variants:** multinomial/softmax, regularized
- [ ] `generalized_linear_models` — Poisson, Gamma regression

### 1.2 `perceptron/`
- [x] `perceptron` — Rosenblatt rule, **variants:** averaged, pocket, multiclass (one-vs-rest)

### 1.3 `svm/`
- [ ] `svm` — hard/soft margin (hinge loss), **variants:** linear, kernel (RBF, polynomial, sigmoid), SMO solver
- [ ] `svr` — support vector regression (ε-insensitive loss)

### 1.4 `knn/`
- [x] `knn` — classification + regression, **variants:** weighted, KD-tree vs brute force, distance metrics

### 1.5 `naive-bayes/`
- [ ] `naive_bayes` — **variants:** Gaussian, Multinomial, Bernoulli, Complement

### 1.6 `trees/`
- [x] `decision_tree` — CART, **variants:** classification (Gini/entropy), regression (MSE), pruning
- [ ] `random_forest` — bagging of trees, feature subsampling, OOB error
- [ ] `gradient_boosting` — GBM, **variants:** XGBoost-style (regularized + second-order), histogram split
- [ ] `adaboost` — SAMME / SAMME.R

### 1.7 `ensemble/`
- [ ] `bagging` — bootstrap aggregation (generic)
- [ ] `stacking` — meta-learner over base models
- [ ] `voting` — hard / soft voting

### 1.8 `clustering/`
- [x] `kmeans` — Lloyd's algorithm, **variants:** k-means++, mini-batch, elbow/silhouette selection
- [ ] `gaussian_mixture` — EM algorithm, **variants:** diagonal/full covariance
- [ ] `dbscan` — density-based, **variants:** OPTICS
- [ ] `hierarchical` — agglomerative, **variants:** single/complete/average/Ward linkage
- [ ] `mean_shift` — kernel density mode-seeking
- [ ] `spectral` — graph Laplacian clustering

### 1.9 `dimensionality-reduction/`
- [x] `pca` — eigen/SVD, **variants:** kernel PCA, incremental, whitening
- [ ] `lda` — Fisher linear discriminant (supervised)
- [ ] `tsne` — t-distributed stochastic neighbour embedding
- [ ] `umap` — uniform manifold approximation
- [ ] `svd` — truncated SVD / matrix factorization
- [ ] `ica` — independent component analysis (FastICA)

---

## 2. `dl/` — Basic Deep Learning

### 2.1 `mlp/`
- [x] `mlp` — feed-forward net from scratch (backprop), **variants:** depth, width, activations
- [ ] `activations` — sigmoid, tanh, ReLU, LeakyReLU, ELU, GELU, Swish (forward + grads)

### 2.2 `optimizers/`
- [ ] `optimizers` — SGD, Momentum, Nesterov, AdaGrad, RMSProp, **Adam**, AdamW (from scratch)

### 2.3 `regularization/`
- [ ] `regularization` — L1/L2 weight decay, **Dropout**, **BatchNorm**, **LayerNorm**, early stopping, label smoothing

### 2.4 `cnn/`
- [x] `cnn` — conv/pool/FC from scratch (im2col), **variants:** LeNet
- [ ] `resnet` — residual blocks, **focus:** *skip connections vs vanishing gradients*
- [ ] `vgg` — deep stacked 3×3 convs
- [ ] `inception` — multi-branch
- [ ] `mobilenet` — depthwise-separable convolutions

### 2.5 `rnn/`
- [x] `rnn` — vanilla RNN BPTT, **focus:** *vanishing/exploding gradients*
- [x] `lstm` — gated memory, **variants:** peephole, GRU comparison
- [ ] `gru` — gated recurrent unit
- [ ] `seq2seq` — encoder-decoder, **variants:** with attention

### 2.6 `autoencoder/`
- [ ] `autoencoder` — **variants:** vanilla, denoising, sparse, contractive

---

## 3. `generative-models/`

### 3.1 `gan/`
- [x] `vanilla_gan` — minimax game, **focus:** *mode collapse, training instability*
- [ ] `dcgan` — deep conv GAN
- [ ] `wgan` — Wasserstein, **variants:** WGAN-GP (gradient penalty)
- [ ] `conditional_gan` — class-conditioned
- [ ] `lsgan` — least-squares GAN
- [ ] `infogan` — disentangled latent codes
- [ ] `cyclegan` — unpaired image translation
- [ ] `pix2pix` — paired image translation
- [ ] `stylegan` — style-based generator (conceptual)

### 3.2 `vae/`
- [x] `vae` — ELBO + reparameterization, **variants:** β-VAE, Conditional VAE
- [ ] `vq_vae` — vector-quantized VAE

### 3.3 `diffusion/`
- [ ] `ddpm` — denoising diffusion probabilistic model
- [ ] `ddim` — deterministic sampling
- [ ] `score_based` — score matching / SDE view

### 3.4 `autoregressive/`
- [ ] `pixelcnn` — masked convolutions
- [ ] `pixelrnn` — row/diagonal LSTM

### 3.5 `normalizing-flows/`
- [ ] `realnvp` — affine coupling layers
- [ ] `glow` — invertible 1×1 conv

---

## 4. `nlp/`

### 4.1 `text-representation/`
- [ ] `bow_tfidf` — bag-of-words, TF-IDF
- [ ] `ngram_lm` — n-gram language model + smoothing

### 4.2 `embeddings/`
- [x] `word2vec` — **variants:** CBOW, Skip-gram, negative sampling, hierarchical softmax
- [ ] `glove` — global co-occurrence factorization
- [ ] `fasttext` — subword embeddings

### 4.3 `language-models/`
- [ ] `neural_lm` — RNN/LSTM language model
- [ ] `char_rnn` — character-level generation

### 4.4 `seq2seq/`
- [ ] `seq2seq_attention` — Bahdanau & Luong attention
- [ ] `ner_tagger` — BiLSTM-CRF (conceptual)

---

## 5. `transformers/`

### 5.1 `attention/`
- [x] `attention` — scaled dot-product, **variants:** multi-head, self vs cross, causal mask
- [ ] `positional_encoding` — sinusoidal, learned, RoPE, ALiBi

### 5.2 `architectures/`
- [x] `transformer` — full encoder-decoder ("Attention Is All You Need")
- [ ] `bert` — encoder-only, masked LM pre-training
- [ ] `gpt` — decoder-only, causal LM
- [ ] `t5` — encoder-decoder, text-to-text
- [ ] `vision_transformer` — ViT, patch embeddings

---

## 6. `training-techniques/` (cross-cutting reference)

These are demonstrated *inside* algorithms above. The table below records the
canonical "home" demo and other places the technique reappears.

| Technique | Canonical demo | Also appears in |
|---|---|---|
| Vanishing / exploding gradients | `dl/rnn/rnn` | `dl/rnn/lstm`, `dl/cnn/resnet`, `dl/mlp/mlp` |
| Gradient clipping | `dl/rnn/rnn` | `transformers/architectures/transformer` |
| Weight initialization (Xavier/He) | `dl/mlp/mlp` | every DL model |
| Batch / Layer normalization | `dl/regularization/regularization` | `dl/cnn/resnet`, `transformers/architectures/transformer` |
| Dropout | `dl/regularization/regularization` | `dl/mlp/mlp`, `transformers/architectures/transformer` |
| Skip / residual connections | `dl/cnn/resnet` | `transformers/architectures/transformer` |
| Learning-rate scheduling / warmup | `transformers/architectures/transformer` | `dl/optimizers/optimizers` |
| Reparameterization trick | `generative-models/vae/vae` | diffusion |
| Adversarial / minimax training | `generative-models/gan/vanilla_gan` | all GAN variants |
| Negative sampling | `nlp/embeddings/word2vec` | — |
| Teacher forcing | `nlp/seq2seq/seq2seq_attention` | `dl/rnn/lstm` |

See [`training-techniques/README.md`](training-techniques/README.md) for the
detailed write-up of each technique.

# ML / AI Algorithms — A Dense, Hands-On Tutorial

A from-scratch tour of machine learning and deep learning. Every major
algorithm is implemented **twice** — once in plain **NumPy** (so you see the
math working) and once in idiomatic **PyTorch** (so you see how it's really
done) — and every notebook explains the **concept** and **derives the math**,
like the essence of a good course condensed into runnable code.

> The aim is to be **comprehensive but not shallow**: classic ML → deep
> learning → generative models → NLP → transformers, including the important
> **variants** of each algorithm and the **training techniques** (vanishing
> gradients, normalization, residual connections, …) that make them work.

---

## Start here

- 🗺️ **[`MAP.md`](MAP.md)** — the master catalogue of every algorithm and the
  file structure. *This is the source of truth.*
- 🤖 **[`AGENTS.md`](AGENTS.md)** — format & rules for contributors (human or
  AI). Read this before adding anything.
- 🖥️ **[`docs/gpu-setup.md`](docs/gpu-setup.md)** — CUDA / Apple-MPS / Colab
  setup (everything also runs on CPU).
- 🏋️ **[`training-techniques/README.md`](training-techniques/README.md)** — the
  cross-cutting tricks, and which algorithms demonstrate them.

## Repository structure

```
ml/                  Classic ML: linear models, SVM, trees, kNN,
                     naive Bayes, clustering, dimensionality reduction
dl/                  Basic deep learning: MLP, optimizers, regularization,
                     CNNs, RNN/LSTM/GRU, autoencoders
generative-models/   GANs, VAEs, diffusion, autoregressive, normalizing flows
nlp/                 Text representation, embeddings, language models, seq2seq
transformers/        Attention, the Transformer, BERT, GPT, ViT
training-techniques/ Cross-cutting techniques referenced throughout
common/              Shared template + small utilities
docs/                Setup and reference docs
```

Each leaf holds an algorithm as a pair of files:

```
ml/clustering/kmeans.py      # NumPy + PyTorch implementations + demo
ml/clustering/kmeans.ipynb   # concept + math derivation + the same code
```

## How each algorithm is taught

| Part | Where | What you get |
|---|---|---|
| Intuition & concept | notebook (markdown) | the idea, like lecture slides |
| Math derivation | notebook (LaTeX) | objective + gradient/update, step by step |
| From-scratch code | `.py` + notebook | NumPy implementation, math made explicit |
| Idiomatic code | `.py` + notebook | PyTorch with autograd / `nn.Module` |
| Training technique | notebook | the relevant trick, measured and plotted |
| Variants | both | the major variations of the algorithm |

## Quick start

```bash
# 1. Install dependencies (CPU is fine for everything)
pip install -r requirements.txt

# 2. Run any algorithm module directly
python ml/linear-models/linear_regression.py
python transformers/architectures/transformer.py

# 3. Or open the matching notebook for the full explanation
jupyter lab ml/linear-models/linear_regression.ipynb
```

For GPU acceleration (optional), see [`docs/gpu-setup.md`](docs/gpu-setup.md).

## Status & contributing

The full list — including what's implemented vs. planned — lives in
[`MAP.md`](MAP.md). To add or extend an algorithm, follow the checklist in
[`AGENTS.md`](AGENTS.md §7). The short version: update the MAP, write
`name.py` (NumPy + PyTorch + demo), write `name.ipynb` (concept + math + code),
demonstrate the relevant training technique, run both, commit.

## License

MIT — see [`LICENSE`](LICENSE). Educational use encouraged.

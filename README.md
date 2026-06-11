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
- 🏋️ **[`06.training-techniques/README.md`](06.training-techniques/README.md)** — the
  cross-cutting tricks, and which algorithms demonstrate them.

## Repository structure

A numbered curriculum: **foundations → modern systems → engineering practice.**

```
01.ml/                  Classic ML: linear models, SVM, trees, kNN,
                        naive Bayes, clustering, dimensionality reduction
02.dl/                  Basic deep learning: MLP, optimizers, regularization,
                        CNNs, RNN/LSTM/GRU, autoencoders
03.generative-models/   GANs, VAEs, diffusion, autoregressive, normalizing flows
04.nlp/                 Text representation, embeddings, language models, seq2seq
05.transformers/        Attention, the Transformer, BERT/GPT/T5/ViT, and an
                        exhaustive LLM-architecture catalogue (llm-architectures/)
06.training-techniques/ Cross-cutting techniques referenced throughout
07.frameworks/          RAG, MCP, agents, serving, fine-tuning, eval, vector DBs,
                        observability, guardrails … full-featured, run locally
08.claudecode/          Clean-room Claude-Code-style coding agent (Python + Ruby)
                        + architecture docs (from public sources)
09.huggingface/         The Hugging Face ecosystem: manifest, workflow, internals
10.gpu/                 CUDA / GPU engineering tutorials
11.agent-ai-engineer/   Researched + ranked AI-engineer skill profile (job market)
12.agent-ai-skills/     One rich explainer per ranked skill
common/ tools/ docs/    Infrastructure (shared utils, notebook builder, setup)
.claude/skills/         The `tutorial-architect` authoring skill
```

Sections **01–06** are the from-scratch algorithm curriculum (NumPy + PyTorch +
notebooks). Sections **07–12** cover the modern LLM/agent engineering stack with
full-featured, locally-runnable tutorials and researched reference material. See
[`MANIFEST.md`](MANIFEST.md) for the complete catalogue and build status.

Each leaf holds an algorithm as a pair of files:

```
01.ml/clustering/kmeans.py      # NumPy + PyTorch implementations + demo
01.ml/clustering/kmeans.ipynb   # concept + math derivation + the same code
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
python 01.ml/linear-models/linear_regression.py
python 05.transformers/architectures/transformer.py

# 3. Or open the matching notebook for the full explanation
jupyter lab 01.ml/linear-models/linear_regression.ipynb
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

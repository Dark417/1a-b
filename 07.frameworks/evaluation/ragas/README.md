# RAGAS — reference-free evaluation for RAG pipelines

RAGAS ("RAG Assessment") scores a retrieval-augmented-generation pipeline
**without gold answers** for most of its metrics. Instead of comparing to a
reference, it uses an LLM (and an embedder) to ask structured questions about
the *relationship* between the four objects every RAG turn produces:

```
   question  ──►  retriever  ──►  contexts  ──►  generator  ──►  answer
      │                              │                             │
      └──────────────── (ground_truth, optional) ─────────────────┘
```

> Docs: https://docs.ragas.io/ · Paper: Es et al. 2023, https://arxiv.org/abs/2309.15217

## The four core metrics (and exactly what they measure)

| Metric | Question it answers | Inputs | "From scratch" recipe |
|---|---|---|---|
| **faithfulness** | Is the answer *grounded* in the retrieved context? (no hallucination) | answer, contexts | extract atomic *claims* from the answer; for each, NLI-check entailment against context; score = supported / total |
| **answer_relevancy** | Does the answer actually address the *question*? | answer, question | LLM generates N questions the answer *could* be answering; cosine-sim those to the real question; score = mean similarity |
| **context_precision** | Are the *relevant* contexts ranked at the top? | question, contexts, ground_truth | for each retrieved chunk, is it relevant? compute precision@k weighted by rank (a mean-average-precision flavour) |
| **context_recall** | Did retrieval fetch *everything* the answer needs? | contexts, ground_truth | split ground_truth into sentences; what fraction can be attributed to the retrieved context? |

Two are **reference-free** (faithfulness, answer_relevancy); two need a
`ground_truth` (context_precision, context_recall). A real pipeline reports all
four to triangulate where a RAG system breaks (bad retrieval vs bad generation).

## Architecture in words

RAGAS wraps every metric in a small program:

1. **Prompt templates** ask the judge LLM to do one narrow sub-task (extract
   claims, generate questions, attribute sentences).
2. A **judge LLM** (`LangchainLLMWrapper` / `LlamaIndexLLMWrapper`) runs them.
3. An **embedder** powers similarity metrics (answer_relevancy).
4. Metrics are computed per-row over a `Dataset` and aggregated.

Because the judge is pluggable, we swap in a deterministic **MockLLM** +
**MockEmbedder** (`../_lib.py`) and a **mock NLI** so the whole metric suite
runs offline. The math is identical to the library's.

## Install

```bash
pip install ragas datasets langchain      # optional — files run without it
python evaluation/ragas/01.faithfulness.py # always runs (offline mock)
```

## Feature tour (files)

| File | Metric / feature |
|---|---|
| `01.faithfulness.py` | claim extraction + NLI entailment; faithfulness from scratch |
| `02.answer_relevancy.py` | generate-questions-from-answer + cosine similarity |
| `03.context_precision_recall.py` | rank-aware precision & ground-truth recall |
| `04.ragas_api.py` | the real `ragas.evaluate(...)` call, guarded → graceful skip |
| `app.py` | evaluate a tiny 3-row RAG dataset on all four metrics, print a scorecard |

## Gotchas

- **The judge is a model.** Faithfulness can mark a *correct* answer unfaithful
  if the context phrases the fact differently and the NLI is weak. Validate on a
  human-labelled slice.
- **answer_relevancy rewards focus, not correctness.** A confidently wrong but
  on-topic answer scores high. Pair it with faithfulness.
- **context_precision needs ground_truth** to decide relevance; without it the
  metric is undefined.
- **Cost & latency.** Each metric is 1+ LLM calls per row; faithfulness can be
  several (one per claim). Batch and cache.
- **Determinism.** Set the judge to temperature 0; otherwise scores wobble.

## Comparison

| Tool | Sweet spot |
|---|---|
| **RAGAS** | RAG-specific, reference-free metrics, fast to wire up |
| TruLens | RAG triad + live tracing of a running app |
| DeepEval | general LLM-app metrics + pytest/CI assertions |
| lm-eval-harness | academic benchmarks, not RAG-shaped |

## References
- RAGAS docs — https://docs.ragas.io/
- Es et al., *RAGAS: Automated Evaluation of RAG* (2023) — https://arxiv.org/abs/2309.15217
- Faithfulness via NLI: the metric mirrors entailment scoring (MNLI-style).

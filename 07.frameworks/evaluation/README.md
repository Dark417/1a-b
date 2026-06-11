# evaluation — measuring LLM & RAG quality

How do you know a model, a prompt, or a RAG pipeline got *better*? You measure
it. This area is a hands-on tour of the modern LLM-evaluation stack, with every
example **runnable offline** via a deterministic `MockLLM` / `MockJudge`
(`_lib.py`) so you can study the harness without an API key.

## The mental model: three families of metric

```
                         ┌───────────────────────────────┐
   reference-based       │ you have a gold answer        │
   (deterministic)       │  exact-match, F1, BLEU, ROUGE │  ← lm-eval-harness
                         └───────────────────────────────┘
                         ┌───────────────────────────────┐
   model-graded          │ an LLM judges the output       │
   (LLM-as-judge)        │  G-Eval, faithfulness, ...    │  ← deepeval, ragas, trulens
                         └───────────────────────────────┘
                         ┌───────────────────────────────┐
   adversarial / safety  │ probe for failures             │
   (scanning)            │  bias, injection, robustness   │  ← giskard, promptfoo
                         └───────────────────────────────┘
```

A real evaluation pipeline mixes all three: cheap deterministic metrics for
regression gates, LLM-as-judge for open-ended quality, and adversarial scans
before you ship.

## Frameworks covered

| Folder | Framework | What it teaches |
|---|---|---|
| [`lm-eval-harness/`](lm-eval-harness/) | EleutherAI lm-evaluation-harness | task specs, few-shot, log-likelihood vs generation metrics; a tiny custom task built from scratch |
| [`ragas/`](ragas/) | RAGAS | RAG metrics: faithfulness, answer-relevancy, context-precision/recall — with a MockLLM judge |
| [`deepeval/`](deepeval/) | DeepEval | G-Eval, metric objects, **pytest** integration, assertions in CI |
| [`promptfoo/`](promptfoo/) | promptfoo | declarative YAML test matrices; a Python-driven runner reproducing the assertion engine |
| [`trulens/`](trulens/) | TruLens | feedback functions, the RAG triad, app tracing |
| [`giskard/`](giskard/) | Giskard | scanning an LLM for vulnerabilities (injection, harmful, hallucination) |

## What "from scratch" means here

Where the metric *is* the lesson, we implement it directly so you see the math,
then point at the real library:

- **exact-match / F1** — token sets, the SQuAD recipe.
- **BLEU** — modified n-gram precision + brevity penalty (Papineni et al. 2002).
- **ROUGE-N / ROUGE-L** — recall-oriented n-gram + LCS (Lin 2004).
- **faithfulness** — claim extraction + NLI entailment against context.
- **LLM-as-judge** — a deterministic rubric scorer standing in for GPT-4.

## Install

Each folder has a `requirements.txt`. The from-scratch metrics need **nothing**
beyond the Python standard library. The optional real libraries (`ragas`,
`deepeval`) are guarded — if absent, the file prints a skip note and still
exits 0.

```bash
pip install -r evaluation/<framework>/requirements.txt   # optional
python evaluation/<framework>/01.*.py                    # always runs
```

## Gotchas (true of every LLM eval)

- **LLM-as-judge is itself a model** — it has bias, variance, and position
  effects. Always sanity-check the judge against human labels on a slice.
- **Reference metrics punish paraphrase.** BLEU/ROUGE/EM score *surface* overlap;
  a correct answer in different words scores low. Use them for closed tasks.
- **Determinism.** Set temperature 0 and seed everything, or your eval is noisy.
- **Contamination.** If the benchmark leaked into pretraining, scores lie.
- **Cost.** Model-graded eval calls a model per sample per metric — it adds up.

## References

- EleutherAI lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness
- RAGAS docs — https://docs.ragas.io/
- DeepEval docs — https://docs.confident-ai.com/
- promptfoo docs — https://www.promptfoo.dev/docs/intro/
- TruLens docs — https://www.trulens.org/
- Giskard LLM scan — https://docs.giskard.ai/en/latest/open_source/scan/scan_llm/
- Zheng et al., *Judging LLM-as-a-Judge* (MT-Bench), 2023 — https://arxiv.org/abs/2306.05685

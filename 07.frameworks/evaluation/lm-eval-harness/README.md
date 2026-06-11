# lm-evaluation-harness (EleutherAI)

The de-facto standard for **academic LLM benchmarking**. It is the engine behind
the Hugging Face Open LLM Leaderboard. You point it at a model and a list of
*tasks* (MMLU, HellaSwag, GSM8K, ARC, ...) and it produces comparable numbers.

> Official repo: https://github.com/EleutherAI/lm-evaluation-harness
> Docs / task guide: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md

## Mental model / architecture in words

```
   ┌────────────┐    requests     ┌──────────────┐    metrics    ┌──────────┐
   │   Task     │ ──────────────► │     Model    │ ────────────► │ Aggregate│
   │ (YAML +    │  loglikelihood  │ (LM adapter: │  per-doc      │ exact_   │
   │  dataset)  │  or generate    │  hf, vllm,   │  scores       │ match,…  │
   └────────────┘                 │  openai,…)   │               └──────────┘
        │                         └──────────────┘
   doc_to_text / doc_to_target / few-shot sampling
```

Three abstractions do all the work:

1. **Task** — a dataset + how to turn a row into a prompt (`doc_to_text`), the
   gold target (`doc_to_target`), the *output type* (`multiple_choice`,
   `loglikelihood`, `generate_until`), and which **metrics** to compute.
2. **Model (LM)** — an adapter exposing two primitives: `loglikelihood(context,
   continuation)` and `generate_until(context, stop)`. Built-ins: `hf`,
   `vllm`, `openai-completions`, `local-chat-completions`, ...
3. **Request → metric** — the harness builds *requests* from each doc (with
   few-shot examples prepended), runs them through the model, and scores.

### Two ways tasks are graded

- **Multiple-choice / loglikelihood**: the model never *generates*. The harness
  asks "what is the log-probability of each candidate continuation?" and picks
  the argmax. This is how MMLU/HellaSwag are scored — robust, no parsing.
- **Generation (`generate_until`)**: the model generates text, which is then
  matched against the gold (exact-match, F1, regex). This is GSM8K, math, etc.

## Install & CLI

```bash
pip install lm-eval                     # the package is "lm-eval"
lm_eval --model hf \
        --model_args pretrained=EleutherAI/pythia-160m \
        --tasks lambada_openai,hellaswag \
        --num_fewshot 5 \
        --batch_size auto \
        --output_path results/
```

Real runs download a model + dataset (network + GPU-ish). **This tutorial does
not require that** — we reimplement the harness's core loop offline so you learn
*how it works*, and gate the real CLI behind a try/except.

## Feature tour (files in this folder)

| File | Feature |
|---|---|
| `01.task_anatomy.py` | A Task object from scratch: `doc_to_text`, `doc_to_target`, few-shot prompt assembly |
| `02.loglikelihood_mc.py` | Multiple-choice scoring via log-likelihood (the MMLU mechanism), with a mock LM |
| `03.generation_metrics.py` | `generate_until` + exact-match / F1 from scratch (the GSM8K mechanism) |
| `04.custom_task.py` | Define & run a brand-new custom task end-to-end (offline) + the real-CLI snippet, guarded |
| `app.py` | A mini-harness: register tasks, run a model adapter, aggregate a results table |

## Key knobs you must understand

- `--num_fewshot N` — N in-context examples are sampled and prepended. More shots
  usually help base models; chat models may prefer 0-shot.
- `--apply_chat_template` — wrap prompts in the model's chat format (crucial for
  instruct models; a frequent source of "my scores are terrible" bugs).
- `--limit K` — evaluate only K docs (fast smoke test).
- `--fewshot_as_multiturn` — render shots as a multi-turn chat.
- `bootstrap_iters` — std-error of metrics via bootstrap resampling.

## Gotchas

- **Chat template mismatch** is the #1 footgun: a base-model prompt fed to an
  instruct model (or vice-versa) tanks scores. Use `--apply_chat_template`.
- **Few-shot leakage**: the harness samples shots from the *train/dev* split;
  ensure they don't overlap the test docs.
- **Loglikelihood ≠ generation** — a model can be great at MC and bad at free
  generation. Report the output type alongside the score.
- **Versioning**: task definitions change; always log the harness + task version
  (the harness records this for you in the results JSON).

## Comparison

| Tool | Sweet spot |
|---|---|
| **lm-eval-harness** | reproducible academic benchmarks, leaderboards |
| HELM (Stanford) | broad, scenario-based, heavier |
| OpenAI Evals | OpenAI-centric, registry of evals |
| DeepEval / RAGAS | application-level, LLM-as-judge, RAG |

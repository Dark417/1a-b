# Langfuse — open-source LLM observability & prompt management

[Langfuse](https://langfuse.com) is the most popular **open-source** LLM
engineering platform: tracing, evaluation, prompt management, and a metrics
dashboard. You can use Langfuse Cloud or **self-host the whole stack with Docker
Compose** — the server, web UI, Postgres and ClickHouse are all MIT/open.

> Docs: https://langfuse.com/docs
> Python SDK (v3/v4, OpenTelemetry-based): https://langfuse.com/docs/sdk/python/sdk-v3
> Self-hosting: https://langfuse.com/self-hosting

This tutorial runs **fully offline**. The Langfuse v3+ Python SDK is built on
OpenTelemetry and, when no API keys are configured, it **auto-disables into a
no-op client** — every `@observe`, span and score call still executes (your code
doesn't change), it just doesn't export. That is the ideal teaching setup: the
exact production code path runs without a network. We also ship a tiny `MockLLM`
so there are no model calls.

## Mental model / architecture in words

```
  your app  ──@observe / start_as_current_observation──►  OTel spans
                                                              │ batch export (OTLP)
                                                              ▼
                         ┌──────────────────────────────────────────────┐
                         │  Langfuse server (self-hosted or Cloud)        │
                         │  ingestion → Postgres (metadata) +             │
                         │              ClickHouse (traces/observations)  │
                         │  Web UI: traces, sessions, scores, prompts,    │
                         │          datasets, evals, dashboards           │
                         └──────────────────────────────────────────────┘
```

The core data model (learn these four nouns):

1. **Trace** — one end-to-end request/run. Top-level container, has a name,
   `user_id`, `session_id`, `tags`, `metadata`, and input/output.
2. **Observation** — a step inside a trace. Three flavours:
   - **SPAN** — any unit of work (retrieval, a tool call, a chain step).
   - **GENERATION** — an LLM call: `model`, `input`, `output`, `usage`
     (tokens) and computed **cost**. The headline observation type.
   - **EVENT** — a point-in-time marker (no duration).
3. **Score** — an evaluation attached to a trace/observation: numeric, boolean,
   or categorical. Sources: user feedback, LLM-as-judge, code checks, manual
   annotation. This is how you measure quality over time.
4. **Prompt** — a versioned, named prompt stored in Langfuse (**prompt
   management**): fetch by name, deploy by label (`production`), template with
   `{{variables}}`, and link generations back to the prompt version used.

(If those four sound like the from-scratch tracer in `../from-scratch-tracing/`,
that's the point — Langfuse is that, productionised, with OTel + a real backend.)

## Install

```bash
pip install "langfuse>=3.0.0"          # OpenTelemetry-based SDK
# Optional integrations:
pip install "langfuse[openai]"         # drop-in openai wrapper
```

Configure (only needed to actually export — the tutorial does NOT require it):

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=http://localhost:3000   # your self-hosted server
```

### Self-hosting in one command (optional)
```bash
# https://langfuse.com/self-hosting/local
git clone https://github.com/langfuse/langfuse && cd langfuse
docker compose up           # UI on http://localhost:3000
```

## Feature tour (files in this folder)

| File | Feature |
|---|---|
| `01.observe_decorator.py` | `@observe` — auto-trace functions; nested spans + generations; the v3 SDK |
| `02.traces_spans_generations.py` | Manual API: `start_as_current_observation`, set model/usage/cost, trace metadata |
| `03.scores_and_eval.py` | Scores: user feedback, LLM-as-judge, numeric/categorical/boolean; trace-level eval |
| `04.prompt_management.py` | Versioned prompt templates, labels, `compile({{vars}})`, link-to-generation (offline cache) |
| `app.py` | End-to-end traced RAG pipeline: prompt → retrieve → generate → score |

All run offline (no-op export). To see them in the UI, set the env vars above.

```bash
python 01.observe_decorator.py
python 02.traces_spans_generations.py
python 03.scores_and_eval.py
python 04.prompt_management.py
python app.py
```

## Key features & knobs

- **`@observe(as_type="generation")`** — mark a function as an LLM generation so
  it shows in the generations view with cost/usage.
- **`langfuse.update_current_trace(...)`** — set `user_id`, `session_id`,
  `tags`, `input`, `output`, `metadata` from inside any nested span.
- **OpenAI drop-in**: `from langfuse.openai import openai` auto-traces every call
  (model, usage, cost, latency) with zero extra code.
- **Integrations**: LangChain (`CallbackHandler`), LlamaIndex, LiteLLM, plus any
  OpenTelemetry-instrumented library (it's an OTel backend).
- **Datasets & experiments** — store eval datasets in Langfuse, run your app
  over them, attach scores, compare versions.
- **Prompt management** — `langfuse.get_prompt("name", label="production")`,
  client-side caching, `prompt.compile(**vars)`, and config alongside the text.
- **Sessions** — group multi-turn conversations by `session_id`.

## Gotchas

- **v2 vs v3+ API.** The v2 SDK used `langfuse.trace()/span()/generation()`
  objects you manually closed. v3+ (this tutorial) is **OpenTelemetry-based**:
  use `@observe` or `start_as_current_observation(...)` context managers, and
  call `langfuse.flush()` before exit. Don't mix the two styles.
- **Flush before exit.** Export is batched and async; short scripts must call
  `get_client().flush()` (or rely on the atexit hook) or you lose the last spans.
- **No keys → disabled.** Without keys the client is a no-op (great for tests;
  surprising if you expected data). Check `langfuse.auth_check()`.
- **Cost needs model match.** Cost is computed from a model→price map; an unknown
  `model` name yields zero cost. Set `usage_details`/`cost_details` explicitly if
  your provider isn't recognised.
- **PII.** Inputs/outputs are sent verbatim. Mask sensitive fields before
  logging (the SDK supports a masking function).

## Comparison

| Tool | Open source | Self-host | Prompt mgmt | Eval/datasets | Standard |
|---|---|---|---|---|---|
| **Langfuse** | yes (MIT) | yes (Docker) | yes | yes | OpenTelemetry |
| LangSmith | no | enterprise | yes | yes | proprietary |
| Phoenix (Arize) | yes | yes | partial | strong evals | OpenInference/OTel |
| Helicone | partial | yes | yes | basic | proxy (also OTel) |
| Opik (Comet) | yes | yes | yes | strong evals | OpenTelemetry |

Langfuse's sweet spot: **open-source, self-hostable, strong prompt management**,
and a clean OTel-based SDK.

# From-scratch LLM tracing — build a tiny observability stack

Before reaching for Langfuse, Phoenix, or OpenTelemetry, you should understand
**what they actually do**. Every LLM observability tool is, at its core, a
**tree of timed spans** decorated with attributes (model, prompt, completion,
token counts, cost) plus **scores/evals** attached after the fact. If you can
build that in ~150 lines, every commercial tool becomes "oh, it's that, plus a
UI and a database."

This folder builds exactly that — a working tracer with spans, nested context,
token/cost accounting, latency, and a span-tree printer — using **only the
Python standard library**. It always runs offline, no deps.

## The mental model (the whole field in one picture)

```
Trace  (one user request / one "run")
 └─ Span  "agent.run"                     12.4 ms   ──────────────
     ├─ Span  "retriever.search"           3.1 ms   ───
     │     attrs: {k: 5, hits: 5}
     └─ Span  "llm.generate"  [GENERATION] 8.9 ms      ──────
           attrs: {model, prompt, completion,
                   prompt_tokens, completion_tokens, cost_usd}
           scores: {relevance: 0.8, hallucinated: 0}
```

- **Trace** — one top-level unit of work (a request, an agent run). Has a
  `trace_id`.
- **Span** — a timed operation with a name, start/end, parent, attributes, and
  events. Spans nest to form a tree. This is the OpenTelemetry primitive.
- **Generation** — a *special span* for an LLM call. The thing everyone wants:
  it captures `model`, `input` (prompt/messages), `output` (completion), and
  **usage** (`prompt_tokens`, `completion_tokens`), from which **cost** is
  derived via a price table. Langfuse and Phoenix both model this as "a span
  with LLM-shaped attributes."
- **Score / Eval** — a number (or label) attached to a span/trace *after* it
  ran: a user thumbs-up, an LLM-as-judge grade, a regex check. Observability =
  traces + scores over time.

The two hard parts a real tool solves for you: (1) **propagating the current
span** so a deeply nested function knows its parent without you threading it
through every call — solved with a `contextvars` stack; (2) **exporting**
spans somewhere (a UI / OTLP collector). We do (1) for real and (2) to stdout.

## Architecture in words

```
  @observe / with tracer.span(...)        contextvars stack
        │                                  ┌───────────────┐
        ▼                                  │  _current_span│  ← who is my parent?
   Span(start=t0)  ──push──►   ... work ...│  (a stack)    │
        │                                  └───────────────┘
   Span.end(t1)   ──pop──►  attach to parent.children
        │
        ▼
   Tracer collects root spans ──► print_tree() / to_dict() (JSON export)
                                  └─ token + cost rollups
```

- `Tracer` — owns the price table and the list of completed root spans; renders.
- `Span` — node in the tree; `enter()/exit()` push/pop the contextvars stack so
  `tracer.span(...)` auto-nests. Holds `attributes`, `events`, `scores`.
- `@observe` — a decorator that wraps any function in a span (the Langfuse
  ergonomic). Arguments/return become span I/O.
- `record_generation(...)` — the LLM-specific helper: stores I/O + usage and
  computes cost from the tracer price table.

## Files (run in order)

| File | Teaches |
|---|---|
| `01.minimal_tracer.py` | Span tree + nested context via `contextvars`; latency; tree printer |
| `02.token_cost_latency.py` | Generation spans, token counting (a toy tokenizer), cost from a price table, rollups |
| `03.observe_decorator.py` | The `@observe` decorator ergonomic (auto-span a function) + scores/evals + JSON export |
| `app.py` | A traced "RAG agent": retriever + LLM spans, nested, with a final LLM-judge score |
| `tracer.py` | The reusable library all the above import |

```bash
python 01.minimal_tracer.py     # all exit 0, no deps, no network
python 02.token_cost_latency.py
python 03.observe_decorator.py
python app.py
```

## What real tools add on top of this

| Concern | This tutorial | Langfuse / Phoenix / OTel |
|---|---|---|
| Storage | in-memory list | Postgres / ClickHouse / OTLP collector |
| Transport | stdout | HTTP batch exporter (OTLP), async flush |
| Standard | ad-hoc dict | **OpenTelemetry** span spec + **OpenInference**/GenAI semantic conventions |
| UI | `print_tree()` | waterfall timeline, search, dashboards |
| Sampling | none | head/tail sampling, rate limits |
| Cost | toy table | live, per-model, per-provider price tables |

The semantic conventions matter: OpenTelemetry's **GenAI** conventions and
Arize's **OpenInference** standardize the attribute *names*
(`gen_ai.request.model`, `llm.token_count.completion`, …) so any backend can
read any instrumented app. Our attributes are ad-hoc; the *shape* is identical.

## Gotchas this teaches you to recognize

- **Lost parent / flat traces** — if context isn't propagated (e.g. across
  threads or `async` without copying context), spans show up as roots. Real
  tools hit this too; OTel has explicit context propagation APIs.
- **Double-counting tokens** — count usage on the generation span only, then
  *roll up*; don't add usage at every level.
- **Cost drift** — prices change; keep the price table external and versioned.
- **Sensitive data in spans** — prompts/outputs may contain PII; real tools add
  masking. We note where you'd redact.

## References
- OpenTelemetry tracing spec: https://opentelemetry.io/docs/concepts/signals/traces/
- OpenTelemetry GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenInference (Arize) semantic conventions: https://github.com/Arize-ai/openinference
- Python `contextvars`: https://docs.python.org/3/library/contextvars.html

"""03 — The @observe decorator, scores/evals, and JSON export.

Run: python 03.observe_decorator.py   (no deps, no network, exits 0)

This is the Langfuse ergonomic distilled: decorate any function with @observe and
its call becomes a span — args captured as input, return value as output — and it
auto-nests under whatever decorated function called it. Then we attach *scores*
(evaluations) to spans, the second half of observability.
"""
from __future__ import annotations

import json

from tracer import MockLLM, get_tracer, observe

tracer = get_tracer()
llm = MockLLM("mock-llm")


@observe  # plain decorator -> span named "retrieve"
def retrieve(query: str, k: int = 3) -> list[str]:
    return [f"doc-{i} about {query.split()[0]}" for i in range(k)]


@observe(kind="generation", name="llm.answer")
def answer(query: str, context: list[str]) -> str:
    prompt = f"Q: {query}\nContext: {context}"
    out = llm.generate(prompt)
    # We can reach into the current span to record usage on the generation.
    stack = _current_stack()
    if stack:
        tracer.record_generation(
            stack[-1],
            model=out["model"],
            prompt=prompt,
            completion=out["completion"],
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"],
        )
    return out["completion"]


def _current_stack():
    # Read the tracer's contextvars stack (used to grab the live generation span).
    from tracer import _SPAN_STACK

    return _SPAN_STACK.get()


@observe(name="rag.pipeline")
def rag(query: str) -> str:
    ctx = retrieve(query, k=2)
    return answer(query, ctx)


def llm_judge_relevance(answer_text: str) -> float:
    """A toy 'LLM-as-judge' eval: deterministic, offline. Real tools call a
    model here; the *shape* (produce a 0..1 score) is what matters."""
    return 1.0 if "mock" in answer_text.lower() else 0.0


def main() -> None:
    result = rag("What is the capital of France?")
    print("answer:", result)

    # Attach an evaluation score to the root span after the run.
    root = tracer.roots[-1]
    root.score("relevance", llm_judge_relevance(result), comment="llm-judge (mock)")
    root.score("user_feedback", 1.0, comment="thumbs up")

    print("\n=== Traced @observe pipeline with scores ===")
    tracer.print_tree()

    print("\n=== JSON export of the trace ===")
    print(json.dumps(root.to_dict(), indent=2)[:1100], "...")

    # self-checks
    assert root.name == "rag.pipeline"
    names = [c.name for c in root.children]
    assert "retrieve" in names and "llm.answer" in names, names
    assert any(s.name == "relevance" for s in root.scores)
    gen = [c for c in root.children if c.kind == "generation"][0]
    assert gen.usage.get("total_tokens", 0) > 0
    print("\nOK: @observe auto-spanned + nested the calls; scores attached.")


if __name__ == "__main__":
    main()

"""02 — Manual Langfuse API: spans, generations, trace metadata, usage & cost.

Run: python 02.traces_spans_generations.py   (offline; exits 0)

When you want explicit control (no decorator), the v3+ SDK gives you context
managers:

    with langfuse.start_as_current_span(name=...) as span:
        with langfuse.start_as_current_generation(name=..., model=...) as gen:
            gen.update(input=..., output=..., usage_details=...)

This shows the three observation types (SPAN, GENERATION, EVENT-like updates),
trace-level metadata (user_id/session_id/tags), and usage→cost.
"""
from __future__ import annotations

from _common import MockLLM, get_langfuse, is_exporting

llm = MockLLM("gpt-4o-mini")
lf = get_langfuse()


def start_span(name, **kw):
    """Open a SPAN observation, tolerant of stub/real client differences."""
    try:
        return lf.start_as_current_observation(name=name, as_type="span", **kw)
    except TypeError:
        return lf.start_as_current_span(name=name, **kw)


def start_generation(name, **kw):
    try:
        return lf.start_as_current_observation(name=name, as_type="generation", **kw)
    except TypeError:
        return lf.start_as_current_generation(name=name, **kw)


def main() -> None:
    print("exporting:", is_exporting())

    with start_span("agent.run", input={"q": "capital of France?"}):
        # Trace-level metadata (set from inside any nested span).
        try:
            lf.update_current_trace(
                name="manual-demo",
                user_id="u-42",
                session_id="sess-1",
                tags=["manual", "offline"],
                metadata={"env": "tutorial"},
            )
        except Exception:
            pass

        # A child SPAN: retrieval (no LLM usage).
        with start_span("retriever.search") as r:
            docs = ["Paris is the capital of France."]
            try:
                r.update(output={"hits": len(docs)})
            except Exception:
                pass

        # A GENERATION observation: the LLM call with usage + cost.
        prompt = "Answer: capital of France?"
        out = llm.chat(prompt)
        with start_generation("llm.generate", model=out["model"]) as g:
            try:
                g.update(
                    input=prompt,
                    output=out["output"],
                    usage_details={
                        "input": out["input_tokens"],
                        "output": out["output_tokens"],
                    },
                    model=out["model"],
                )
            except Exception:
                pass

        answer = out["output"]

    try:
        lf.flush()
    except Exception:
        pass

    assert "Paris" in answer
    print("answer:", answer)
    print("OK: SPAN + GENERATION + trace metadata recorded (offline-safe).")


if __name__ == "__main__":
    main()

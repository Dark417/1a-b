"""01 — Langfuse @observe: auto-trace functions into nested spans/generations.

Run: python 01.observe_decorator.py   (offline; exits 0 with or without langfuse)

The @observe decorator wraps a function so each call becomes an *observation* in
the current trace. Nested decorated calls nest automatically (OTel context).
Mark an LLM function with as_type="generation" so it lands in the generations
view with token usage + cost.

Without API keys the v3+ SDK is a no-op exporter: the SAME code runs offline.
Set LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST to see these traces in the UI.
"""
from __future__ import annotations

from _common import MockLLM, get_langfuse, is_exporting, observe

llm = MockLLM("gpt-4o-mini")
langfuse = get_langfuse()


@observe()  # a SPAN observation named after the function
def retrieve(query: str, k: int = 2) -> list[str]:
    return [f"doc{i}:{query.split()[0]}" for i in range(k)]


@observe(as_type="generation")  # a GENERATION observation (LLM call)
def generate(query: str, context: list[str]) -> str:
    prompt = f"Q: {query}\nContext: {context}\nAnswer:"
    out = llm.chat(prompt)
    # Record model + usage on the current generation so cost is computed.
    try:
        langfuse.update_current_generation(
            model=out["model"],
            usage_details={
                "input": out["input_tokens"],
                "output": out["output_tokens"],
            },
            input=prompt,
            output=out["output"],
        )
    except Exception:
        pass  # stub client / older SDK — example still runs
    return out["output"]


@observe()  # the root span -> becomes the trace
def rag(query: str) -> str:
    ctx = retrieve(query, k=2)
    ans = generate(query, ctx)
    # Decorate the trace with metadata visible in the UI.
    try:
        langfuse.update_current_trace(
            name="rag", user_id="demo-user", tags=["tutorial", "offline"]
        )
    except Exception:
        pass
    return ans


def main() -> None:
    print("langfuse exporting to a server:", is_exporting())
    answer = rag("What is the capital of France?")
    print("answer:", answer)

    # Always flush short scripts (export is batched/async).
    try:
        langfuse.flush()
    except Exception:
        pass

    assert "Paris" in answer
    print("OK: @observe traced retrieve + generate under one trace (offline-safe).")


if __name__ == "__main__":
    main()

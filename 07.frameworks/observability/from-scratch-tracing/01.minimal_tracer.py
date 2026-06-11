"""01 — A minimal tracer: spans, nesting, latency, and a tree printer.

Run: python 01.minimal_tracer.py   (no deps, no network, exits 0)

The single idea: a `with tracer.span(name):` block records start/end time and,
crucially, *auto-nests* under whatever span is currently open — because the
tracer keeps a stack of open spans in a contextvar. You never pass parents by
hand. This is exactly OpenTelemetry's "current span" mechanism, in 5 lines.
"""
from __future__ import annotations

import time

from tracer import Tracer


def fake_work(ms: float) -> None:
    """Burn a little wall-clock so latencies are non-zero and visible."""
    time.sleep(ms / 1000.0)


def main() -> None:
    tracer = Tracer()

    # A trace is just the outermost span. Everything opened inside nests under it
    # automatically thanks to the contextvars stack in tracer.span().
    with tracer.span("agent.run") as root:
        root.set(user="demo", question="What is the capital of France?")

        with tracer.span("retriever.search", k=3) as r:
            fake_work(2)
            r.set(hits=3).event("loaded index").event("ranked candidates")

        with tracer.span("planner.decide") as p:
            fake_work(1)
            # Spans nest arbitrarily deep; the parent is found via the stack.
            with tracer.span("planner.tool_choice") as t:
                fake_work(1)
                t.set(tool="search", confidence=0.92)

        with tracer.span("llm.generate", kind="generation") as g:
            fake_work(3)
            g.set(model="mock-llm", output="Paris.")

    print("=== Span tree (waterfall, but as text) ===")
    tracer.print_tree()

    print("\n=== One root span as JSON (what an exporter would ship) ===")
    import json

    print(json.dumps(tracer.roots[0].to_dict(), indent=2)[:900], "...")

    # Sanity asserts so the file is self-checking.
    assert len(tracer.roots) == 1, "should be exactly one trace"
    assert tracer.roots[0].name == "agent.run"
    # planner.tool_choice should be a grandchild, proving auto-nesting works.
    planner = [c for c in tracer.roots[0].children if c.name == "planner.decide"][0]
    assert planner.children and planner.children[0].name == "planner.tool_choice"
    print("\nOK: nesting + latency captured with zero manual parent wiring.")


if __name__ == "__main__":
    main()

"""02 — Generation spans: token counting, cost from a price table, roll-ups.

Run: python 02.token_cost_latency.py   (no deps, no network, exits 0)

LLM observability = ordinary tracing PLUS three numbers per model call:
  - prompt_tokens, completion_tokens   (usage)
  - cost_usd                            (usage x price table)
  - latency_ms                          (span timing)

We use a deterministic MockLLM (a toy whitespace tokenizer) so the numbers are
reproducible offline. Then we *roll up* tokens and cost across the whole trace —
counting usage only on generation spans so nothing is double-counted.
"""
from __future__ import annotations

from tracer import MockLLM, Tracer


def main() -> None:
    # A custom price table (USD / 1k tokens). Real tools keep this external and
    # versioned because provider prices change.
    tracer = Tracer(prices={"mock-llm": (0.0005, 0.0015)})
    llm = MockLLM("mock-llm")

    questions = [
        "Summarise the French Revolution in one sentence.",
        "Name three primary colors.",
        "Explain backpropagation to a five year old please.",
    ]

    with tracer.span("batch.qa") as batch:
        batch.set(n=len(questions))
        for i, q in enumerate(questions):
            with tracer.span(f"item[{i}]") as item:
                # Pretend retrieval (a non-LLM span: no usage, no cost).
                with tracer.span("retriever.search", k=2) as r:
                    r.set(hits=2)
                # The LLM call: open a generation span and record usage+cost.
                with tracer.generation("llm.generate") as g:
                    out = llm.generate(q)
                    tracer.record_generation(
                        g,
                        model=out["model"],
                        prompt=q,
                        completion=out["completion"],
                        prompt_tokens=out["prompt_tokens"],
                        completion_tokens=out["completion_tokens"],
                    )
                item.set(answer=out["completion"])

    print("=== Trace with per-generation token + cost ===")
    tracer.print_tree()

    print("\n=== Roll-up across the whole trace ===")
    summary = tracer.summary()
    print(summary)

    # Verify the roll-up equals the manual sum of the three generations.
    gens = []

    def walk(sp):
        if sp.kind == "generation":
            gens.append(sp)
        for c in sp.children:
            walk(c)

    for root in tracer.roots:
        walk(root)
    manual_cost = sum(g.cost_usd for g in gens)
    manual_tok = sum(g.usage["total_tokens"] for g in gens)
    assert len(gens) == 3, "three LLM calls expected"
    assert abs(summary["cost_usd"] - round(manual_cost, 6)) < 1e-9
    assert summary["tokens"]["total_tokens"] == manual_tok
    print(
        f"\nOK: {len(gens)} generations, {manual_tok} tokens, "
        f"${manual_cost:.6f} — roll-up matches manual sum (no double counting)."
    )


if __name__ == "__main__":
    main()

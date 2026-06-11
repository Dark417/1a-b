"""08.structured_and_retries.py — structured output + node-level retry policy.

Two production concerns:

  * **Structured output** — keep state in a typed shape. A node parses free text
    into a Pydantic model and stores validated fields. With a real LLM you'd use
    `model.with_structured_output(Schema)`; offline we parse deterministically,
    but the *state contract* (validated Pydantic) is identical.
  * **Retries** — attach a `RetryPolicy` to a node so transient failures are
    retried with backoff. We make a node fail the first N times to prove the
    runtime retries it, then succeeds.

Docs:
  * Structured output: https://langchain-ai.github.io/langgraph/how-tos/#structured-output
  * RetryPolicy: https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.RetryPolicy
"""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field


# --- structured output schema ---------------------------------------------- #
class Invoice(BaseModel):
    vendor: str
    amount: float = Field(ge=0)
    currency: str


class State(TypedDict):
    raw: str
    invoice: dict
    attempts: int
    note: str


def parse_invoice(state: State) -> dict:
    """Deterministic 'extraction' → validated Pydantic model → dict in state."""
    text = state["raw"]
    vendor = re.search(r"Vendor:\s*(.+)", text).group(1).strip()
    amount = float(re.search(r"Amount:\s*([\d.]+)", text).group(1))
    currency = re.search(r"Currency:\s*(\w+)", text).group(1)
    invoice = Invoice(vendor=vendor, amount=amount, currency=currency)  # validates!
    return {"invoice": invoice.model_dump()}


# --- a flaky node to exercise RetryPolicy ---------------------------------- #
_FAILS = {"n": 0}


def flaky_enrich(state: State) -> dict:
    _FAILS["n"] += 1
    if _FAILS["n"] < 3:  # fail twice, succeed on the 3rd attempt
        raise ConnectionError(f"transient failure (attempt {_FAILS['n']})")
    return {"attempts": _FAILS["n"], "note": "enriched after retries"}


def build():
    g = StateGraph(State)
    g.add_node("parse_invoice", parse_invoice)
    # Retry up to 5 times with fast backoff, on any Exception.
    g.add_node(
        "flaky_enrich",
        flaky_enrich,
        retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.01, retry_on=Exception),
    )
    g.add_edge(START, "parse_invoice")
    g.add_edge("parse_invoice", "flaky_enrich")
    g.add_edge("flaky_enrich", END)
    return g.compile()


def main() -> None:
    app = build()
    raw = "Vendor: Acme Corp\nAmount: 199.95\nCurrency: USD"
    result = app.invoke({"raw": raw, "attempts": 0})

    print("structured invoice (validated Pydantic):")
    for k, v in result["invoice"].items():
        print(f"   {k:9}: {v}")
    assert result["invoice"]["amount"] == 199.95

    print(f"\nflaky node succeeded after {result['attempts']} attempts ->", result["note"])
    assert result["attempts"] == 3  # 2 failures + 1 success

    print("\nOK: structured output validated; RetryPolicy recovered a flaky node.")


if __name__ == "__main__":
    main()

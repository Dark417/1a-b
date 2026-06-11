"""06.streaming.py — stream state updates, full values, and message tokens.

`.invoke` returns only the final state. `.stream` yields *as the graph runs*,
controlled by `stream_mode`:

  * `"updates"` — the per-node delta (what each node returned). Great for logs.
  * `"values"`  — the full state after each super-step.
  * `"messages"`— (chat graphs) streams LLM **tokens** as (token, metadata),
    so you can render output incrementally.
  * a list, e.g. `["updates","messages"]`, multiplexes several modes.

We demonstrate `updates` and `values` on a 3-node graph; the token-streaming
note explains the `messages` mode (which needs a streaming-capable model).

Docs: https://langchain-ai.github.io/langgraph/concepts/streaming/
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    value: int
    log: Annotated[list[str], operator.add]


def add_ten(state: State) -> dict:
    return {"value": state["value"] + 10, "log": ["add_ten"]}


def double(state: State) -> dict:
    return {"value": state["value"] * 2, "log": ["double"]}


def negate(state: State) -> dict:
    return {"value": -state["value"], "log": ["negate"]}


def build():
    g = StateGraph(State)
    for name, fn in [("add_ten", add_ten), ("double", double), ("negate", negate)]:
        g.add_node(name, fn)
    g.add_edge(START, "add_ten")
    g.add_edge("add_ten", "double")
    g.add_edge("double", "negate")
    g.add_edge("negate", END)
    return g.compile()


def main() -> None:
    app = build()

    print("stream_mode='updates' (per-node deltas):")
    for chunk in app.stream({"value": 5, "log": []}, stream_mode="updates"):
        for node, delta in chunk.items():
            print(f"   {node:8} -> {delta}")

    print("\nstream_mode='values' (full state after each step):")
    for state in app.stream({"value": 5, "log": []}, stream_mode="values"):
        print(f"   value={state['value']:<5} log={state['log']}")

    print("\nstream_mode=['updates','values'] (multiplexed):")
    for mode, payload in app.stream(
        {"value": 1, "log": []}, stream_mode=["updates", "values"]
    ):
        print(f"   [{mode}] {payload}")

    print(
        "\nNote: stream_mode='messages' streams LLM *tokens* as (chunk, meta) for"
        "\n      chat graphs with a streaming model (e.g. ChatOpenAI). Offline our"
        "\n      ScriptedChatModel emits whole messages, so we demo state streaming."
    )
    print("\nOK: streamed updates, values, and a multiplexed mode.")


if __name__ == "__main__":
    main()

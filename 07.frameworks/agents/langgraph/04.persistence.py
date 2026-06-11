"""04.persistence.py — checkpointing, threads, resume, and time-travel.

Compile a graph with a **checkpointer** and every super-step's state is
snapshotted, keyed by a `thread_id` you pass in `config`. This unlocks:

  * **Memory across calls** — a second `.invoke` on the same thread continues
    from the saved state (the reducer keeps appending messages).
  * **Resume after a crash** — re-running picks up where it left off.
  * **Time-travel** — `get_state_history(config)` lists every checkpoint; you
    can re-run from any past one (e.g. to branch an alternative).

We use `MemorySaver` (in-process). Swap for `SqliteSaver`/`PostgresSaver` to
persist to disk/DB — identical API.

Docs: https://langchain-ai.github.io/langgraph/concepts/persistence/
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    turns: Annotated[list[str], operator.add]
    counter: int


def step(state: State) -> dict:
    n = state.get("counter", 0) + 1
    return {"counter": n, "turns": [f"turn #{n}"]}


def build():
    g = StateGraph(State)
    g.add_node("step", step)
    g.add_edge(START, "step")
    g.add_edge("step", END)
    # The checkpointer is what makes the graph durable.
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    app = build()
    cfg = {"configurable": {"thread_id": "user-42"}}

    # Three separate invocations on the SAME thread accumulate state.
    for _ in range(3):
        app.invoke({"turns": []}, cfg)
    snap = app.get_state(cfg)
    print("after 3 invokes on thread 'user-42':")
    print("   counter =", snap.values["counter"])
    print("   turns   =", snap.values["turns"])
    assert snap.values["counter"] == 3

    # A DIFFERENT thread is independent — no shared memory.
    other = {"configurable": {"thread_id": "user-99"}}
    app.invoke({"turns": []}, other)
    assert app.get_state(other).values["counter"] == 1
    print("\nthread 'user-99' counter =", app.get_state(other).values["counter"], "(independent)")

    # Time-travel: list the checkpoint history for user-42.
    history = list(app.get_state_history(cfg))
    print(f"\ntime-travel: {len(history)} checkpoints recorded for user-42")
    # Re-run from the OLDEST checkpoint to branch an alternate timeline.
    oldest = history[-1]
    branched = app.invoke({"turns": []}, oldest.config)
    print("   re-running from the oldest checkpoint gives counter =", branched["counter"])

    print("\nOK: state persisted per-thread, isolated, and time-travelled.")


if __name__ == "__main__":
    main()

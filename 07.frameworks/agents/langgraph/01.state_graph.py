"""01.state_graph.py — the minimal StateGraph: typed state, nodes, edges.

The whole of LangGraph rests on three ideas shown here:

  * **State** is a `TypedDict`. Each key is a *channel*. A channel can have a
    **reducer** (via `Annotated[..., reducer]`) that merges concurrent writes;
    without one, a write *replaces* the value.
  * **Nodes** are plain functions `state -> partial_state`. The returned dict is
    merged into state via the reducers.
  * **Edges** wire nodes together; `START` and `END` are sentinels.

We build a tiny 3-node pipeline (normalize → count words → summarize) to make
the data flow concrete, and show a reducer accumulating a log channel.

Docs: https://langchain-ai.github.io/langgraph/concepts/low_level/
Runs fully offline (no LLM needed for this one).
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


# --- State: two normal channels + one *reduced* channel (a running log) ----- #
class State(TypedDict):
    text: str                              # replaced by whoever writes it
    word_count: int                        # replaced
    summary: str                           # replaced
    log: Annotated[list[str], operator.add]  # REDUCED: writes are concatenated


# --- Nodes: each returns ONLY the channels it changes ----------------------- #
def normalize(state: State) -> dict:
    cleaned = " ".join(state["text"].split()).strip()
    return {"text": cleaned, "log": ["normalize: collapsed whitespace"]}


def count_words(state: State) -> dict:
    n = len(state["text"].split())
    return {"word_count": n, "log": [f"count_words: {n} words"]}


def summarize(state: State) -> dict:
    words = state["text"].split()
    head = " ".join(words[:6])
    summary = f"{head}{'…' if len(words) > 6 else ''} ({state['word_count']} words)"
    return {"summary": summary, "log": ["summarize: built summary"]}


def build_graph():
    g = StateGraph(State)
    g.add_node("normalize", normalize)
    g.add_node("count_words", count_words)
    g.add_node("summarize", summarize)
    g.add_edge(START, "normalize")
    g.add_edge("normalize", "count_words")
    g.add_edge("count_words", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


def main() -> None:
    app = build_graph()
    result = app.invoke(
        {"text": "  LangGraph   models  agents   as   stateful graphs  ", "log": []}
    )
    print("final text   :", repr(result["text"]))
    print("word_count   :", result["word_count"])
    print("summary      :", result["summary"])
    print("log (reduced):")
    for line in result["log"]:
        print("   -", line)

    # The reducer is why `log` accumulated across 3 nodes instead of being
    # overwritten. `word_count`/`summary` have no reducer, so they were replaced.
    assert result["word_count"] == 6
    assert len(result["log"]) == 3
    print("\nOK: state flowed through 3 nodes; reducer accumulated the log.")


if __name__ == "__main__":
    main()

"""05.human_in_the_loop.py — pause mid-graph with interrupt(), then resume.

A durable graph (one with a checkpointer) can call `interrupt(payload)` inside a
node. This **suspends** the run, surfaces `payload` to the caller, and persists
state. The human inspects it, then the caller **resumes** by invoking the graph
with `Command(resume=<their answer>)`; execution re-enters the interrupted node
and `interrupt()` now *returns* that answer.

This is the backbone of approval gates, edits, and "are you sure?" prompts.

Pattern shown: a node proposes a risky action; we interrupt for approval; on
"approve" we execute, on "reject" we skip.

Docs: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    action: str
    decision: str
    result: str


def propose(state: State) -> dict:
    return {"action": f"DELETE all files in /tmp/{state['action']}"}


def approval_gate(state: State) -> dict:
    # Execution stops here on the first pass; `decision` is supplied on resume.
    decision = interrupt({"review": state["action"], "options": ["approve", "reject"]})
    return {"decision": decision}


def execute(state: State) -> dict:
    if state["decision"] == "approve":
        return {"result": f"executed: {state['action']}"}
    return {"result": "skipped (human rejected)"}


def build():
    g = StateGraph(State)
    g.add_node("propose", propose)
    g.add_node("approval_gate", approval_gate)
    g.add_node("execute", execute)
    g.add_edge(START, "propose")
    g.add_edge("propose", "approval_gate")
    g.add_edge("approval_gate", "execute")
    g.add_edge("execute", END)
    return g.compile(checkpointer=MemorySaver())


def run_with_decision(human_says: str, thread: str) -> str:
    app = build()
    cfg = {"configurable": {"thread_id": thread}}

    # First invoke runs until interrupt() and returns with an __interrupt__ key.
    first = app.invoke({"action": "cache"}, cfg)
    assert "__interrupt__" in first, "expected to pause at the approval gate"
    prompt = first["__interrupt__"][0].value
    print(f"[{thread}] paused for review: {prompt['review']}  -> human: {human_says!r}")

    # Resume by supplying the human's decision.
    final = app.invoke(Command(resume=human_says), cfg)
    print(f"[{thread}] resumed → {final['result']}")
    return final["result"]


def main() -> None:
    approved = run_with_decision("approve", "t-approve")
    rejected = run_with_decision("reject", "t-reject")
    assert approved.startswith("executed")
    assert rejected.startswith("skipped")
    print("\nOK: interrupt() paused the graph; Command(resume=...) continued it.")


if __name__ == "__main__":
    main()

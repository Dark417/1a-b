"""07.multi_agent.py — a supervisor routing between specialist worker agents.

Multi-agent in LangGraph = several agent nodes sharing one state, with a
**supervisor** node whose router decides which worker runs next (or finish).
This is the "supervisor" topology (others: network/swarm, hierarchical).

Topology:

        START → supervisor ──┬─ "math"     → math_agent ──┐
                             ├─ "research" → research_agent┤→ back to supervisor
                             └─ "done"     → END           ┘

The supervisor reads the latest worker output and the task, and emits the next
route. Workers are themselves mini ReAct loops (here, scripted) that write a
result back into shared state. We cap iterations so the demo always terminates.

Docs: https://langchain-ai.github.io/langgraph/concepts/multi_agent/
Runs offline via the ScriptedChatModel.
"""
from __future__ import annotations

import importlib.util
import operator
import pathlib
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

_spec = importlib.util.spec_from_file_location(
    "mock_llm", pathlib.Path(__file__).with_name("00.mock_llm.py")
)
mock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock)

TOOLS = {t.name: t for t in mock.ALL_TOOLS}


class State(TypedDict):
    task: str
    findings: Annotated[list[str], operator.add]
    next: str
    steps: int


# --- supervisor: a deterministic plan over two workers, then finish --------- #
PLAN = ["research", "math", "done"]


def supervisor(state: State) -> dict:
    i = state.get("steps", 0)
    nxt = PLAN[min(i, len(PLAN) - 1)]
    return {"next": nxt, "steps": i + 1}


def research_agent(state: State) -> dict:
    fact = TOOLS["web_search"].invoke({"query": "langgraph"})
    return {"findings": [f"[research] {fact}"]}


def math_agent(state: State) -> dict:
    answer = TOOLS["calculator"].invoke({"expression": "21*2"})
    return {"findings": [f"[math] 21*2 = {answer}"]}


def route(state: State) -> str:
    return state["next"]


def build():
    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    g.add_node("research", research_agent)
    g.add_node("math", math_agent)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor", route, {"research": "research", "math": "math", "done": END}
    )
    # Each worker reports back to the supervisor (the collaboration cycle).
    g.add_edge("research", "supervisor")
    g.add_edge("math", "supervisor")
    return g.compile()


def main() -> None:
    app = build()
    result = app.invoke(
        {"task": "Summarize LangGraph and compute 21*2", "findings": [], "steps": 0},
        {"recursion_limit": 20},
    )
    print("task:", result["task"])
    print("collaboration result:")
    for f in result["findings"]:
        print("   -", f)
    assert any("research" in f for f in result["findings"])
    assert any("42" in f for f in result["findings"])
    print("\nOK: supervisor routed work to two specialist agents over shared state.")


if __name__ == "__main__":
    main()

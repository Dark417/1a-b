"""app.py — end-to-end research assistant graph tying every feature together.

A small but complete agent that, given a question:

  1. **plan**      — decompose into sub-steps (structured state).
  2. **act**       — a ReAct tool loop (web_search + calculator) over shared state.
  3. **reflect**   — check whether findings are sufficient; loop back if not
                     (conditional edge / the agent cycle).
  4. **approve**   — human-in-the-loop gate before publishing (interrupt()).
  5. **report**    — write the final note (tool), return structured output.

It uses: StateGraph + reducers, conditional edges, ToolNode-style execution,
persistence (MemorySaver + thread_id), interrupt()/Command resume, and streaming
of progress. Fully offline via ScriptedChatModel; swap in ChatOpenAI for live.

Docs: https://langchain-ai.github.io/langgraph/tutorials/
"""
from __future__ import annotations

import importlib.util
import operator
import pathlib
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

_spec = importlib.util.spec_from_file_location(
    "mock_llm", pathlib.Path(__file__).with_name("00.mock_llm.py")
)
mock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock)
TOOLS = {t.name: t for t in mock.ALL_TOOLS}


class State(TypedDict):
    question: str
    plan: list[str]
    messages: Annotated[list[BaseMessage], operator.add]
    findings: Annotated[list[str], operator.add]
    loops: int
    approved: str
    report: str


def make_model() -> mock.ScriptedChatModel:
    return mock.ScriptedChatModel(
        responses=[
            AIMessage(
                content="Searching for background.",
                tool_calls=[mock.tool_call("web_search", {"query": "langgraph"}, "s1")],
            ),
            AIMessage(
                content="Now a quick calculation.",
                tool_calls=[mock.tool_call("calculator", {"expression": "2024-2017"}, "s2")],
            ),
            AIMessage(content="I have enough to answer."),
        ]
    )


def plan_node(state: State) -> dict:
    plan = ["search background", "compute a figure", "summarize"]
    return {"plan": plan, "messages": [HumanMessage(content=state["question"])]}


def act_node(state: State, *, model) -> dict:
    reply = model.invoke(state["messages"])
    updates: dict = {"messages": [reply]}
    if reply.tool_calls:
        tool_msgs, finds = [], []
        for c in reply.tool_calls:
            res = TOOLS[c["name"]].invoke(c["args"])
            tool_msgs.append(ToolMessage(content=str(res), tool_call_id=c["id"]))
            finds.append(f"{c['name']}({c['args']}) -> {res}")
        updates["messages"] += tool_msgs
        updates["findings"] = finds
    return updates


def reflect_node(state: State) -> dict:
    return {"loops": state.get("loops", 0) + 1}


def route_reflect(state: State) -> str:
    """Loop back to act until the model stops calling tools (or we hit a cap)."""
    last = state["messages"][-1]
    enough = isinstance(last, AIMessage) and not last.tool_calls
    if enough or state.get("loops", 0) >= 4:
        return "approve"
    return "act"


def approve_node(state: State) -> dict:
    decision = interrupt({"draft_findings": state["findings"], "ok_to_publish?": True})
    return {"approved": decision}


def report_node(state: State) -> dict:
    if state["approved"] != "approve":
        return {"report": "(publishing cancelled by reviewer)"}
    body = "Findings:\n" + "\n".join(f" - {f}" for f in state["findings"])
    TOOLS["write_note"].invoke({"name": "report.md", "content": body})
    return {"report": body}


def build():
    model = make_model()
    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("act", lambda s: act_node(s, model=model))
    g.add_node("reflect", reflect_node)
    g.add_node("approve", approve_node)
    g.add_node("report", report_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "reflect")
    g.add_conditional_edges("reflect", route_reflect, {"act": "act", "approve": "approve"})
    g.add_edge("approve", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    app = build()
    cfg = {"configurable": {"thread_id": "research-1"}}
    question = "What is LangGraph and how many years since the Transformer paper?"

    print("=== streaming planning + tool loop ===")
    for chunk in app.stream({"question": question, "loops": 0}, cfg, stream_mode="updates"):
        for node, delta in chunk.items():
            if node == "__interrupt__":
                continue
            keys = ", ".join(delta.keys()) if isinstance(delta, dict) else delta
            print(f"   [{node}] updated: {keys}")

    state = app.get_state(cfg)
    if "__interrupt__" in (state.tasks[0].interrupts and {"__interrupt__": 1} or {}) or state.next:
        pass  # (the graph paused at the approval gate)

    print("\n=== human-in-the-loop: approving publication ===")
    final = app.invoke(Command(resume="approve"), cfg)
    print(final["report"])

    assert "report.md" in mock._FILES
    assert final["approved"] == "approve"
    print("\nOK: plan→act(tool loop)→reflect→approve(HITL)→report ran end-to-end.")


if __name__ == "__main__":
    main()

"""02.conditional_edges.py — branching, routing, and the agent loop as a cycle.

A plain edge always fires. A **conditional edge** runs a *router* function that
inspects state and returns the name of the next node (or `END`). This is how
LangGraph expresses:

  * if/else branching, and
  * the **agent loop** itself: a cycle `model -> (tools? -> model) -> end`.

We hand-build a ReAct-style loop *without* the prebuilt helper so the mechanics
are visible: a `model` node decides to call a tool or finish; a router sends
flow to `tools` (then back to `model`) or to `END`.

Docs: https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges
Runs fully offline via the ScriptedChatModel from 00.mock_llm.py.
"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Import the digit-prefixed sibling module by path.
_spec = importlib.util.spec_from_file_location(
    "mock_llm", pathlib.Path(__file__).with_name("00.mock_llm.py")
)
mock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# Tool registry by name for manual execution.
TOOLS = {t.name: t for t in mock.ALL_TOOLS}


def make_model() -> mock.ScriptedChatModel:
    # Script: call calculator, then (after seeing the result) answer.
    return mock.ScriptedChatModel(
        responses=[
            AIMessage(
                content="I should compute this.",
                tool_calls=[mock.tool_call("calculator", {"expression": "12*9"}, "c1")],
            ),
            AIMessage(content="12 * 9 = 108."),
        ]
    )


def model_node(state: State, *, model) -> dict:
    reply = model.invoke(state["messages"])
    return {"messages": [reply]}


def tools_node(state: State) -> dict:
    last = state["messages"][-1]
    out: list[BaseMessage] = []
    for call in last.tool_calls:
        result = TOOLS[call["name"]].invoke(call["args"])
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": out}


def route_after_model(state: State) -> str:
    """Router: if the model asked for tools, go run them; else stop."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def build_graph():
    model = make_model()
    g = StateGraph(State)
    g.add_node("model", lambda s: model_node(s, model=model))
    g.add_node("tools", tools_node)
    g.add_edge(START, "model")
    # Conditional: model -> tools | END
    g.add_conditional_edges("model", route_after_model, {"tools": "tools", END: END})
    # Unconditional: after tools, always re-enter the model (the cycle).
    g.add_edge("tools", "model")
    return g.compile()


def main() -> None:
    app = build_graph()
    state = app.invoke({"messages": [HumanMessage(content="What is 12*9?")]})
    print("conversation:")
    for m in state["messages"]:
        kind = type(m).__name__
        extra = f" tool_calls={[c['name'] for c in m.tool_calls]}" if getattr(m, "tool_calls", None) else ""
        print(f"  {kind:13} | {str(m.content)[:48]!r}{extra}")
    assert "108" in state["messages"][-1].content
    print("\nOK: conditional edge formed an agent loop (model→tools→model→end).")


if __name__ == "__main__":
    main()

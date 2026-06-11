"""03.tool_calling.py — tools the easy way: ToolNode + prebuilt ReAct agent.

File 02 wired the agent loop by hand. In practice you use two prebuilt pieces:

  * **`ToolNode`** — a ready-made node that reads the last AIMessage's
    `tool_calls`, runs the matching tools (in parallel), and appends
    `ToolMessage`s. Drop it in instead of hand-writing `tools_node`.
  * **`create_react_agent(model, tools)`** — compiles an entire ReAct agent
    (model ↔ tools loop, with the right state schema and routing) in one call.

We show both, driven offline by the ScriptedChatModel. A real deployment passes
`ChatOpenAI(...).bind_tools(tools)` instead.

Docs:
  * ToolNode: https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode
  * create_react_agent: https://langchain-ai.github.io/langgraph/reference/agents/
"""
from __future__ import annotations

import importlib.util
import pathlib
import warnings

from langchain_core.messages import AIMessage, HumanMessage

warnings.filterwarnings("ignore")  # silence the create_react_agent move notice

_spec = importlib.util.spec_from_file_location(
    "mock_llm", pathlib.Path(__file__).with_name("00.mock_llm.py")
)
mock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mock)

from typing import Annotated, TypedDict  # noqa: E402

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode, create_react_agent  # noqa: E402


class _TNState(TypedDict):
    messages: Annotated[list, add_messages]


def demo_tool_node() -> None:
    """ToolNode executes tool_calls from the last message automatically.

    ToolNode is meant to live *inside* a compiled graph (it reads injected
    config/store), so we wrap it in a one-node graph rather than calling it
    standalone.
    """
    g = StateGraph(_TNState)
    g.add_node("tools", ToolNode(mock.ALL_TOOLS))
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    app = g.compile()
    msg = AIMessage(
        content="",
        tool_calls=[
            mock.tool_call("calculator", {"expression": "2+2"}, "a"),
            mock.tool_call("web_search", {"query": "capital of france"}, "b"),
        ],
    )
    out = app.invoke({"messages": [msg]})
    produced = [m for m in out["messages"] if getattr(m, "tool_call_id", None)]
    print("ToolNode produced", len(produced), "ToolMessages:")
    for tm in produced:
        print("   -", str(tm.content)[:50])


def demo_prebuilt_agent() -> None:
    """create_react_agent compiles the whole loop in one line."""
    model = mock.ScriptedChatModel(
        responses=[
            AIMessage(
                content="Let me look that up and compute.",
                tool_calls=[mock.tool_call("web_search", {"query": "transformer"}, "t1")],
            ),
            AIMessage(
                content="",
                tool_calls=[mock.tool_call("calculator", {"expression": "2017+0"}, "t2")],
            ),
            AIMessage(content="The Transformer (2017) replaced recurrence with attention."),
        ]
    )
    agent = create_react_agent(model, mock.ALL_TOOLS)
    result = agent.invoke(
        {"messages": [HumanMessage(content="Tell me about the transformer.")]}
    )
    print("\nprebuilt ReAct agent transcript:")
    for m in result["messages"]:
        tc = [c["name"] for c in getattr(m, "tool_calls", [])] or ""
        print(f"  {type(m).__name__:13} | {str(m.content)[:42]!r} {tc}")
    assert "attention" in result["messages"][-1].content.lower()


def main() -> None:
    demo_tool_node()
    demo_prebuilt_agent()
    print("\nOK: ToolNode + create_react_agent ran the tool loop offline.")


if __name__ == "__main__":
    main()

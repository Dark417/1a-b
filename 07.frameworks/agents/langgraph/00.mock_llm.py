"""00.mock_llm.py — the deterministic, offline LLM + tools every example reuses.

LangGraph is provider-agnostic: a "model" is any `langchain_core` chat model.
To run the *real* LangGraph runtime with **no API key and no network**, we
implement `ScriptedChatModel`, a genuine `BaseChatModel` subclass whose replies
are a fixed script (a list of `AIMessage`s, some carrying `tool_calls`). The
graph executes for real; only the token source is mocked.

Swap it for a live model by importing `ChatOpenAI`/`ChatAnthropic` and binding
tools — the rest of every file is unchanged.

Docs:
  * Custom chat models: https://python.langchain.com/docs/how_to/custom_chat_model/
  * Tools: https://python.langchain.com/docs/concepts/tools/
"""
from __future__ import annotations

import random
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

random.seed(0)


# --------------------------------------------------------------------------- #
# Tools — small, deterministic, offline. Real `@tool`-decorated callables.
# --------------------------------------------------------------------------- #
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '2 + 3 * 4'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "error: unsupported characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 (sandboxed)
    except Exception as exc:  # pragma: no cover - defensive
        return f"error: {exc}"


# A tiny offline "knowledge base" so web_search is deterministic.
_FACTS = {
    "langgraph": "LangGraph models agents as stateful graphs with persistence and HITL.",
    "python": "Python is a high-level, dynamically typed programming language (1991).",
    "transformer": "The Transformer (Vaswani et al., 2017) uses self-attention; no recurrence.",
    "capital of france": "Paris is the capital of France.",
}


@tool
def web_search(query: str) -> str:
    """Look up a fact in a deterministic offline knowledge base."""
    q = query.lower().strip()
    for key, val in _FACTS.items():
        if key in q:
            return val
    return f"No offline result for {query!r}; (live search would go here)."


_FILES: dict[str, str] = {}


@tool
def write_note(name: str, content: str) -> str:
    """Persist a short note to an in-memory store; returns a confirmation."""
    _FILES[name] = content
    return f"wrote {len(content)} chars to {name}"


@tool
def read_note(name: str) -> str:
    """Read a previously written note back from the in-memory store."""
    return _FILES.get(name, f"error: no note named {name!r}")


ALL_TOOLS = [calculator, web_search, write_note, read_note]


# --------------------------------------------------------------------------- #
# ScriptedChatModel — a real BaseChatModel whose outputs are pre-scripted.
# --------------------------------------------------------------------------- #
class ScriptedChatModel(BaseChatModel):
    """Deterministic chat model that replays a fixed list of AIMessages.

    Each call to the model pops the next scripted `AIMessage`. Messages that
    carry `tool_calls` drive LangGraph's tool loop; a plain-text message ends it.
    The script is *shorter-safe*: once exhausted it repeats the last message so
    a runaway loop terminates instead of crashing.
    """

    responses: list[AIMessage] = []
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-offline"

    # create_react_agent / .bind_tools() call this; our script is fixed, so we
    # simply return self (tool schemas are ignored by the script).
    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self.i, len(self.responses) - 1)
        self.i += 1
        msg = self.responses[idx]
        # Return a *copy* so re-runs (e.g. interrupt resume) don't mutate ids.
        out = AIMessage(content=msg.content, tool_calls=list(msg.tool_calls or []))
        return ChatResult(generations=[ChatGeneration(message=out)])


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Helper to build a LangChain tool-call dict for a scripted AIMessage."""
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


if __name__ == "__main__":
    print("calculator('6*7') ->", calculator.invoke({"expression": "6*7"}))
    print("web_search('python') ->", web_search.invoke({"query": "python"}))
    m = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[tool_call("calculator", {"expression": "6*7"}, "c1")]),
            AIMessage(content="The answer is 42."),
        ]
    )
    print("scripted reply 1:", m.invoke("hi").tool_calls)
    print("scripted reply 2:", m.invoke("hi").content)
    print("\nOK: offline mock LLM + tools ready.")

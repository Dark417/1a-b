"""llm.py — backend abstraction for the mini coding agent.

A *backend* turns a list of conversation messages (plus tool schemas) into the
next assistant *step*: either some text + a set of tool calls, or a final
answer. This mirrors the real Claude Code architecture, where the agent loop
calls the Messages API with `tools=[...]`, inspects `stop_reason`, executes any
`tool_use` blocks, and feeds `tool_result` blocks back (see
`docs/02.architecture.md` and the Anthropic tool-use docs cited there).

Three backends are provided:

* ``MockBackend`` — deterministic, offline, *default*. It runs a small scripted
  planner so the demo works with **no API key and no network**. The script is
  keyed off the user's task and the observations seen so far, so it behaves like
  a tiny real agent (read a dir, write a file, run it, finish) without an LLM.
* ``AnthropicBackend`` — optional. Uses the real ``anthropic`` SDK + Messages
  API tool-use loop if the package is installed and ``ANTHROPIC_API_KEY`` is
  set. Model defaults to ``claude-opus-4-8`` with adaptive thinking + streaming,
  per Anthropic's current guidance.
* ``OllamaBackend`` — optional. Talks to a local Ollama server for a fully
  local, open-model path. Tool calls are parsed from a JSON convention because
  not every local model supports native tool calling.

Only ``MockBackend`` is needed for the offline demo; the other two are clearly
marked optional and degrade gracefully when their dependency is missing.

This is original, clean-room educational code — NOT Anthropic's proprietary
Claude Code source. See ``docs/07.references.md`` for the public sources used.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Data types shared by every backend
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single request from the model to run a tool.

    ``id`` correlates the call with its result (exactly like the Anthropic API's
    ``tool_use_id``). ``name`` selects a tool from the registry and ``args`` is
    the parsed JSON input object.
    """

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Step:
    """One assistant turn produced by a backend.

    * ``text`` — assistant prose to stream to the user (may be empty).
    * ``tool_calls`` — tools the agent wants to run this turn. If non-empty the
      agent loop executes them and calls the backend again with the results.
    * ``done`` — when True and there are no tool calls, the turn is final and the
      loop ends (the analog of ``stop_reason == "end_turn"``).
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False


class Backend:
    """Interface every backend implements.

    ``step`` receives the full message history (the API is stateless, so the
    whole transcript is resent each turn) and the available tool schemas, and
    returns the next :class:`Step`. ``stream`` optionally yields text fragments
    for incremental output; the default just emits the whole ``Step.text`` once.
    """

    name = "backend"

    def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Step:
        raise NotImplementedError

    def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterable[str]:
        """Yield text chunks for the *next* step, for streaming-style output.

        Backends that have a real token stream override this. The mock splits its
        text on word boundaries so the REPL still shows incremental output.
        """
        step = self.step(messages, tools)
        self._last_step = step
        for chunk in _word_chunks(step.text):
            yield chunk

    # After ``stream`` runs, the agent reads the resolved step from here.
    _last_step: Step | None = None

    def last_step(self) -> Step:
        assert self._last_step is not None, "call stream() before last_step()"
        return self._last_step


def _word_chunks(text: str) -> Iterable[str]:
    """Split text into small chunks to simulate token streaming."""
    if not text:
        return
    parts = re.split(r"(\s+)", text)
    for p in parts:
        if p:
            yield p


# ---------------------------------------------------------------------------
# MockBackend — deterministic scripted planner (offline default)
# ---------------------------------------------------------------------------

# A "plan" is a list of *planners*: functions that look at the conversation so
# far and decide the next step. This keeps the mock honest — it reacts to tool
# observations like a real agent rather than blindly replaying a fixed list.

Planner = Callable[["MockState"], Step | None]


@dataclass
class MockState:
    """Distilled view of the conversation the mock reasons over."""

    task: str
    observations: list[str]            # tool result texts, in order
    tool_calls_made: list[str]         # names of tools already invoked
    counter: int                       # monotonically increasing id source


class MockBackend(Backend):
    """Deterministic, offline backend driving a scripted agent demo.

    The default script implements the canonical "create a file and run it"
    task: list the directory, write a small program, run it, then summarize.
    You can pass your own ``script`` (a list of :class:`Step` templates or
    planner callables) to drive other scripted demos.
    """

    name = "mock"

    def __init__(self, script: list[Step | Planner] | None = None) -> None:
        self.script = script
        self._idx = 0
        self._counter = 0

    # -- public API ---------------------------------------------------------

    def step(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Step:
        state = self._distill(messages)
        if self.script is not None:
            return self._scripted_step(state)
        return self._default_plan(state)

    # -- helpers ------------------------------------------------------------

    def _next_id(self) -> str:
        self._counter += 1
        return f"call_{self._counter:03d}"

    def _distill(self, messages: list[dict[str, Any]]) -> MockState:
        task = ""
        observations: list[str] = []
        tool_calls_made: list[str] = []
        for m in messages:
            content = m.get("content")
            if m["role"] == "user":
                # First textual user message is the task.
                if isinstance(content, str) and not task:
                    task = content
                elif isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_result":
                            observations.append(str(block.get("content", "")))
            elif m["role"] == "assistant" and isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_use":
                        tool_calls_made.append(block["name"])
        return MockState(task, observations, tool_calls_made, self._counter)

    def _scripted_step(self, state: MockState) -> Step:
        if self._idx >= len(self.script):
            return Step(text="(scripted plan complete)", done=True)
        item = self.script[self._idx]
        self._idx += 1
        if callable(item):
            result = item(state)
            return result if result is not None else Step(done=True)
        # A Step template: assign fresh ids to its tool calls.
        fresh = Step(
            text=item.text,
            done=item.done,
            tool_calls=[ToolCall(self._next_id(), c.name, c.args) for c in item.tool_calls],
        )
        return fresh

    def _default_plan(self, state: MockState) -> Step:
        """The built-in 'write a file and run it' agent script.

        Each branch is gated on what's already been done, so the planner is
        re-entrant and robust to the exact message shapes.
        """
        made = state.tool_calls_made

        # 1. Orient: look at the working directory.
        if "list_dir" not in made:
            return Step(
                text="I'll start by looking at the working directory to orient myself.",
                tool_calls=[ToolCall(self._next_id(), "list_dir", {"path": "."})],
            )

        # 2. Act: write a small Python program.
        if "write_file" not in made:
            program = (
                "import sys\n"
                'print("hello from the mini coding agent")\n'
                "print('2 + 2 =', 2 + 2)\n"
                "sys.exit(0)\n"
            )
            return Step(
                text="The directory is empty. I'll create a small program `hello.py`.",
                tool_calls=[
                    ToolCall(
                        self._next_id(),
                        "write_file",
                        {"path": "hello.py", "content": program},
                    )
                ],
            )

        # 3. Verify: run it.
        if "bash" not in made:
            return Step(
                text="Now I'll run the program to verify it works.",
                tool_calls=[
                    ToolCall(self._next_id(), "bash", {"command": "python hello.py"})
                ],
            )

        # 4. Finish: summarize, no more tool calls.
        last = state.observations[-1] if state.observations else ""
        return Step(
            text=(
                "Done. I created `hello.py` and ran it successfully. "
                f"Its output was:\n{last.strip()}\n"
                "The task is complete."
            ),
            done=True,
        )


# ---------------------------------------------------------------------------
# AnthropicBackend — optional real Claude backend
# ---------------------------------------------------------------------------

class AnthropicBackend(Backend):
    """Real Claude backend via the official ``anthropic`` SDK (optional).

    Used only when ``anthropic`` is importable and ``ANTHROPIC_API_KEY`` is set.
    Implements one turn of the Messages API tool-use loop: it sends the history
    + tool schemas, streams text, and converts ``tool_use`` blocks into
    :class:`ToolCall` objects. Defaults follow Anthropic's current guidance —
    model ``claude-opus-4-8``, adaptive thinking, and streaming for long output.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-4-8", system: str = "") -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover - optional path
            raise RuntimeError(
                "AnthropicBackend requires `pip install anthropic`"
            ) from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("AnthropicBackend requires ANTHROPIC_API_KEY")
        import anthropic

        self._client = anthropic.Anthropic()
        self.model = model
        self.system = system

    def step(self, messages, tools):  # pragma: no cover - needs network/key
        # Translate our generic tool schemas to the API's input_schema shape.
        api_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t["parameters"],
            }
            for t in tools
        ]
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        with self._client.messages.stream(
            model=self.model,
            max_tokens=8000,
            system=self.system or None,
            thinking={"type": "adaptive"},
            tools=api_tools,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    text_parts.append(event.delta.text)
            final = stream.get_final_message()
        for block in final.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCall(block.id, block.name, dict(block.input)))
        done = final.stop_reason == "end_turn"
        return Step(text="".join(text_parts), tool_calls=tool_calls, done=done)


# ---------------------------------------------------------------------------
# OllamaBackend — optional local open-model backend
# ---------------------------------------------------------------------------

OLLAMA_TOOL_PROTOCOL = """\
You are a coding agent. You have these tools (call by emitting a JSON object on \
its own line):
{tool_list}

To call tools, reply with ONLY a JSON object:
  {{"tool_calls": [{{"name": "<tool>", "args": {{...}}}}]}}
When the task is finished, reply with ONLY:
  {{"done": true, "text": "<final summary>"}}
"""


class OllamaBackend(Backend):
    """Local open-model backend via Ollama's HTTP API (optional).

    Many small local models lack native tool calling, so this backend uses a
    JSON convention (see ``OLLAMA_TOOL_PROTOCOL``) and parses the model's reply.
    It degrades clearly if the server is unreachable.
    """

    name = "ollama"

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")

    def step(self, messages, tools):  # pragma: no cover - needs local server
        import urllib.request

        tool_list = "\n".join(f"- {t['name']}: {t.get('description','')}" for t in tools)
        system = OLLAMA_TOOL_PROTOCOL.format(tool_list=tool_list)
        prompt = self._flatten(messages)
        payload = json.dumps(
            {"model": self.model, "system": system, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return self._parse(data.get("response", ""))

    @staticmethod
    def _flatten(messages: list[dict[str, Any]]) -> str:
        lines = []
        for m in messages:
            c = m["content"]
            if isinstance(c, list):
                c = json.dumps(c)
            lines.append(f"{m['role'].upper()}: {c}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def _parse(self, response: str) -> Step:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return Step(text=response, done=True)
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return Step(text=response, done=True)
        if obj.get("done"):
            return Step(text=obj.get("text", ""), done=True)
        calls = [
            ToolCall(f"call_{i}", c["name"], c.get("args", {}))
            for i, c in enumerate(obj.get("tool_calls", []))
        ]
        return Step(text=obj.get("text", ""), tool_calls=calls)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def auto_backend(system: str = "") -> Backend:
    """Pick the best available backend without failing.

    Order: real Anthropic (if key + SDK present) → Ollama (if reachable) →
    Mock. The demo forces Mock explicitly so it never depends on this.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:  # pragma: no cover - optional
            return AnthropicBackend(system=system)
        except Exception:
            pass
    return MockBackend()

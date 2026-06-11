"""agent.py — the agent loop / REPL.

This is the heart of the mini coding agent and the educational analog of Claude
Code's main loop (``docs/02.architecture.md``):

    gather context -> ask the model -> stream its text
        -> if it requested tools:
               gate each call (permissions)
               run the allowed ones
               feed results back
           and repeat
        -> else: the turn is done

It wires together the four subsystems:

* :class:`~llm.Backend`        — produces the next step (mock / real Claude / Ollama)
* :class:`~tools.ToolRegistry` — executes tool calls
* :class:`~permissions.PermissionGate` — allow/ask/deny before each call
* :class:`~context.ContextManager`     — history + token-budget compaction

It also implements **streaming-style incremental output** and **parallel
read-only tool execution** (read-only calls in the same turn run concurrently,
mutating ones run serially — mirroring how Claude Code schedules parallel-safe
tools). All original, clean-room code.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, TextIO

from context import ContextManager
from llm import Backend, MockBackend, Step, ToolCall
from permissions import PermissionGate
from tools import ToolRegistry, ToolResult


DEFAULT_SYSTEM = (
    "You are a concise coding agent. Use the available tools to inspect and "
    "modify the workspace. Plan briefly, act with tools, verify your work, then "
    "stop. Prefer dedicated tools (read_file/write_file/edit_file) over bash for "
    "file operations."
)


@dataclass
class TurnLog:
    """Structured record of one agent turn, useful for tests and transcripts."""

    text: str
    tool_calls: list[ToolCall]
    results: list[tuple[str, ToolResult]]   # (tool_name, result)
    done: bool


class Agent:
    """Drives the gather→plan→act→observe loop to completion."""

    def __init__(
        self,
        backend: Backend | None = None,
        tools: ToolRegistry | None = None,
        gate: PermissionGate | None = None,
        context: ContextManager | None = None,
        system: str = DEFAULT_SYSTEM,
        out: TextIO | None = None,
        max_turns: int = 20,
        stream: bool = True,
    ) -> None:
        self.backend = backend or MockBackend()
        self.tools = tools or ToolRegistry(".")
        self.gate = gate or PermissionGate()
        self.context = context or ContextManager(system=system)
        self.out = out or sys.stdout
        self.max_turns = max_turns
        self.stream = stream
        self.turns: list[TurnLog] = []

    # -- output helper ------------------------------------------------------

    def _emit(self, text: str, end: str = "") -> None:
        self.out.write(text + end)
        self.out.flush()

    # -- public entry point -------------------------------------------------

    def run(self, task: str) -> str:
        """Run the loop on ``task`` and return the final assistant text."""
        self.context.add_user(task)
        final_text = ""

        for _turn in range(self.max_turns):
            step = self._ask_backend()
            final_text = step.text

            # Record the assistant turn (text + any tool_use blocks).
            self.context.add_assistant(step.text, step.tool_calls)

            if not step.tool_calls:
                # No tools requested → the turn (and run) is finished.
                self.turns.append(TurnLog(step.text, [], [], True))
                break

            # Execute the requested tools, gated by permissions.
            results = self._run_tools(step.tool_calls)
            self.turns.append(TurnLog(step.text, step.tool_calls, results, step.done))

            # Feed results back as tool_result blocks for the next turn.
            self.context.add_tool_results([
                (call.id, res.content, res.is_error)
                for call, (_, res) in zip(step.tool_calls, results)
            ])

            # Manage the token budget between turns.
            if self.context.maybe_compact():
                self._emit("\n[context compacted]\n")

        return final_text

    # -- one model turn -----------------------------------------------------

    def _ask_backend(self) -> Step:
        messages = self.context.to_api_messages()
        schemas = self.tools.schemas()
        self._emit("\nassistant> ")
        if self.stream:
            for chunk in self.backend.stream(messages, schemas):
                self._emit(chunk)
            step = self.backend.last_step()
        else:
            step = self.backend.step(messages, schemas)
            self._emit(step.text)
        self._emit("\n")
        return step

    # -- tool execution -----------------------------------------------------

    def _run_tools(self, calls: list[ToolCall]) -> list[tuple[str, ToolResult]]:
        """Gate then run each call. Read-only calls run in parallel."""
        # Partition into parallel-safe (read-only) and serial (mutating) calls,
        # preserving original order for the returned list.
        results: dict[int, tuple[str, ToolResult]] = {}
        parallel: list[tuple[int, ToolCall]] = []
        serial: list[tuple[int, ToolCall]] = []
        for i, call in enumerate(calls):
            tool = self.tools.get(call.name)
            read_only = bool(tool and tool.read_only)
            (parallel if read_only else serial).append((i, call))

        # Run read-only calls concurrently.
        if parallel:
            with ThreadPoolExecutor(max_workers=min(4, len(parallel))) as pool:
                futures = {pool.submit(self._gate_and_run, c): i for i, c in parallel}
                for fut, i in futures.items():
                    results[i] = fut.result()

        # Run mutating calls one at a time.
        for i, call in serial:
            results[i] = self._gate_and_run(call)

        return [results[i] for i in range(len(calls))]

    def _gate_and_run(self, call: ToolCall) -> tuple[str, ToolResult]:
        tool = self.tools.get(call.name)
        read_only = bool(tool and tool.read_only)
        allowed, reason = self.gate.check(call.name, call.args, read_only)
        self._emit(f"  · {call.name}({_fmt_args(call.args)}) — {reason}\n")
        if not allowed:
            return call.name, ToolResult(f"Permission denied: {reason}", is_error=True)
        result = self.tools.call(call.name, call.args)
        # Echo a short preview of the result for the transcript.
        preview = result.content.splitlines()[0] if result.content else ""
        self._emit(f"    {preview[:100]}\n")
        return call.name, result


def _fmt_args(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        s = str(v).replace("\n", "\\n")
        parts.append(f"{k}={s[:40]}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def repl(agent: Agent) -> None:  # pragma: no cover - interactive
    """A minimal interactive loop: read a task, run it, repeat."""
    print("mini coding agent — type a task, or 'quit' to exit.")
    while True:
        try:
            task = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if task.lower() in {"quit", "exit"}:
            break
        if task:
            agent.run(task)


if __name__ == "__main__":  # pragma: no cover
    import tempfile

    from permissions import PermissionConfig, Mode, auto_approve

    with tempfile.TemporaryDirectory() as d:
        a = Agent(
            backend=MockBackend(),
            tools=ToolRegistry(d),
            gate=PermissionGate(PermissionConfig(mode=Mode.DEFAULT), prompter=auto_approve),
            context=ContextManager(system=DEFAULT_SYSTEM),
        )
        repl(a)

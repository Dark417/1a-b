"""context.py — conversation history + token budget / compaction.

The Messages API is stateless: every turn resends the whole transcript. Long
agentic sessions therefore grow until they threaten the context window, so real
systems *compact* — summarize older turns and keep recent ones verbatim (see
``docs/02.architecture.md`` and Anthropic's compaction docs).

This module provides:

* :class:`ContextManager` — appends user / assistant / tool messages in the
  exact block shapes the API expects (``tool_use`` / ``tool_result``), estimates
  a token budget, and **compacts** when the estimate crosses a threshold.
* A simple, dependency-free token estimator (~4 chars/token). Real code should
  use the ``count_tokens`` endpoint; we keep it offline and approximate.

Compaction here replaces a run of old messages with a single summary message,
preserving the system prompt and the most recent turns — enough to teach the
mechanism without an LLM in the loop (the summary is built deterministically).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """Very rough token estimate (~4 characters per token).

    Deliberately offline and approximate. The real agent would call
    ``client.messages.count_tokens(...)``; here we only need a monotonic signal
    to trigger compaction in the demo.
    """
    return max(1, len(text) // 4)


def _message_tokens(message: dict[str, Any]) -> int:
    content = message.get("content", "")
    if isinstance(content, str):
        return estimate_tokens(content)
    total = 0
    for block in content:
        if block.get("type") == "text":
            total += estimate_tokens(block.get("text", ""))
        elif block.get("type") == "tool_use":
            total += estimate_tokens(str(block.get("input", "")))
        elif block.get("type") == "tool_result":
            total += estimate_tokens(str(block.get("content", "")))
    return total


@dataclass
class ContextManager:
    """Holds the transcript and compacts it when over budget.

    ``budget`` is the soft token ceiling; once the running estimate exceeds
    ``trigger_ratio * budget`` the manager summarizes everything except the most
    recent ``keep_recent`` messages.
    """

    system: str = ""
    budget: int = 4000
    trigger_ratio: float = 0.75
    keep_recent: int = 4
    messages: list[dict[str, Any]] = field(default_factory=list)
    compactions: int = 0

    # -- appending ----------------------------------------------------------

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str, tool_calls: list[Any] | None = None) -> None:
        """Add an assistant turn with optional ``tool_use`` blocks.

        ``tool_calls`` are ``llm.ToolCall`` objects; they're stored as API-shaped
        ``tool_use`` blocks so the next turn's ``tool_result`` blocks line up.
        """
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        for call in tool_calls or []:
            blocks.append({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.args,
            })
        # If there were no blocks at all, store an empty text block to keep shape.
        self.messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        """Add a user turn carrying ``tool_result`` blocks.

        ``results`` is a list of ``(tool_use_id, content, is_error)``.
        """
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": tid,
                "content": content,
                "is_error": is_error,
            }
            for tid, content, is_error in results
        ]
        self.messages.append({"role": "user", "content": blocks})

    # -- budget / compaction -----------------------------------------------

    def total_tokens(self) -> int:
        return estimate_tokens(self.system) + sum(_message_tokens(m) for m in self.messages)

    def maybe_compact(self) -> bool:
        """Compact if over the trigger threshold. Returns True if it compacted."""
        if self.total_tokens() <= self.trigger_ratio * self.budget:
            return False
        if len(self.messages) <= self.keep_recent + 1:
            return False
        self._compact()
        return True

    def _compact(self) -> None:
        head = self.messages[: -self.keep_recent]
        tail = self.messages[-self.keep_recent :]
        summary = self._summarize(head)
        # Replace the head with one synthetic user message holding the summary.
        self.messages = [{"role": "user", "content": summary}] + tail
        self.compactions += 1

    @staticmethod
    def _summarize(messages: list[dict[str, Any]]) -> str:
        """Deterministic summary of compacted messages (no LLM needed).

        A real implementation asks the model for a summary; for an offline demo
        we extract the salient actions: which tools ran and the first line of
        each user message. This keeps the mechanism honest and inspectable.
        """
        actions: list[str] = []
        for m in messages:
            content = m.get("content")
            if m["role"] == "user" and isinstance(content, str):
                actions.append(f"- user said: {content.splitlines()[0][:80]}")
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_use":
                        actions.append(f"- ran tool: {block['name']}({block.get('input', {})})")
                    elif block.get("type") == "tool_result":
                        first = str(block.get("content", "")).splitlines()[:1]
                        if first:
                            actions.append(f"  -> {first[0][:80]}")
        body = "\n".join(actions) if actions else "(no prior actions)"
        return f"[COMPACTED SUMMARY of earlier conversation]\n{body}"

    # -- export -------------------------------------------------------------

    def to_api_messages(self) -> list[dict[str, Any]]:
        """The message list to send to a backend this turn."""
        return list(self.messages)

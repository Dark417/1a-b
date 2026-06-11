"""permissions.py — the permission gate.

Before the agent runs any tool, the request passes through a *permission gate*.
This is a faithful, educational model of Claude Code's permission system
(``docs/04.permissions.md``):

* **Three decisions:** ``allow`` (run it), ``deny`` (refuse), ``ask`` (prompt
  the user). ``deny`` always wins over ``allow``, matching the real precedence.
* **Rules** are ``Tool(pattern)`` strings, e.g. ``"bash(git *)"`` or
  ``"read_file(*)"``. ``*`` is a glob wildcard over the rule's "signature".
* **Permission modes** mirror Claude Code: ``default`` (ask for risky tools),
  ``acceptEdits`` (auto-allow file writes/edits), ``plan`` (read-only — block
  all mutating tools), and ``bypass`` (allow everything — for CI/sandboxes).
* **Read-only tools** are auto-allowed: searching and reading never need a gate.
* **Auto-approve** mode answers every ``ask`` with "yes" so non-interactive
  demos run end-to-end with no human in the loop.

The gate is deliberately independent of the tool implementations: it sees only
the tool name + arguments, exactly like the real harness intercepting tool_use
blocks before dispatch.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class Mode(str, Enum):
    DEFAULT = "default"        # ask before risky (mutating) tools
    ACCEPT_EDITS = "acceptEdits"  # auto-allow file writes/edits, still ask for bash
    PLAN = "plan"              # read-only: block every mutating tool
    BYPASS = "bypass"          # allow everything (use only in sandboxes/CI)


# Tools that never mutate state — always safe to run without a prompt.
READ_ONLY_TOOLS = {"read_file", "glob", "grep", "list_dir"}
# Tools that write to the filesystem (governed by acceptEdits / plan mode).
EDIT_TOOLS = {"write_file", "edit_file"}


def tool_signature(name: str, args: dict[str, Any]) -> str:
    """Build the string a rule pattern matches against.

    For ``bash`` we expose the command (so ``bash(git *)`` is meaningful); for
    file tools we expose the path; otherwise just the tool name. This mirrors
    Claude Code rules like ``Bash(npm run test *)`` and ``Read(./.env)``.
    """
    if name == "bash":
        return f"bash({args.get('command', '')})"
    if "path" in args:
        return f"{name}({args['path']})"
    if "pattern" in args:
        return f"{name}({args['pattern']})"
    return f"{name}()"


@dataclass
class PermissionConfig:
    """An allow/deny/ask rule set + a mode, like a slice of ``settings.json``."""

    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=list)
    mode: Mode = Mode.DEFAULT


def _matches(signature: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(signature, p) for p in patterns)


# A prompter answers an "ask": given a human-readable question, return True/False.
Prompter = Callable[[str], bool]


def auto_approve(_question: str) -> bool:
    """Non-interactive prompter that approves everything (for demos/CI)."""
    return True


def auto_deny(_question: str) -> bool:
    return False


def cli_prompter(question: str) -> bool:  # pragma: no cover - interactive
    """Interactive prompter reading y/n from stdin."""
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        return False


class PermissionGate:
    """Decides whether a tool call may run, and resolves ``ask`` via a prompter."""

    def __init__(self, config: PermissionConfig | None = None, prompter: Prompter | None = None) -> None:
        self.config = config or PermissionConfig()
        self.prompter = prompter or cli_prompter

    def decide(self, name: str, args: dict[str, Any], read_only: bool = False) -> Decision:
        """Return the raw decision (without prompting) for ``name(args)``."""
        sig = tool_signature(name, args)

        # 1. Explicit deny always wins.
        if _matches(sig, self.config.deny):
            return Decision.DENY

        # 2. Mode-level policy.
        mode = self.config.mode
        if mode is Mode.BYPASS:
            return Decision.ALLOW
        if mode is Mode.PLAN and not (read_only or name in READ_ONLY_TOOLS):
            # Plan mode: only read-only exploration is permitted.
            return Decision.DENY

        # 3. Read-only tools never need a gate.
        if read_only or name in READ_ONLY_TOOLS:
            return Decision.ALLOW

        # 4. Explicit allow.
        if _matches(sig, self.config.allow):
            return Decision.ALLOW

        # 5. acceptEdits auto-allows file mutations.
        if mode is Mode.ACCEPT_EDITS and name in EDIT_TOOLS:
            return Decision.ALLOW

        # 6. Explicit ask, else fall through to default (ask for mutating tools).
        if _matches(sig, self.config.ask):
            return Decision.ASK
        return Decision.ASK

    def check(self, name: str, args: dict[str, Any], read_only: bool = False) -> tuple[bool, str]:
        """Resolve a decision into ``(allowed, reason)``, prompting if needed."""
        decision = self.decide(name, args, read_only)
        sig = tool_signature(name, args)
        if decision is Decision.ALLOW:
            return True, f"allowed: {sig}"
        if decision is Decision.DENY:
            return False, f"denied by policy: {sig}"
        # ASK -> consult the prompter.
        approved = self.prompter(f"Allow {sig}?")
        return approved, ("approved by user" if approved else "rejected by user") + f": {sig}"

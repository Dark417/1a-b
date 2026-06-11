"""tools.py — the tool registry for the mini coding agent.

Claude Code's power comes from a small, well-designed set of tools the model can
call: Read, Write, Edit, Bash, Glob, Grep, plus task/agent and todo tools. This
module is an original, educational re-implementation of that idea: each tool is

* a plain Python function that does the work and returns a string result,
* paired with a JSON-Schema-ish ``parameters`` block (the same shape passed to
  the Anthropic API's ``input_schema``), and
* registered in a :class:`ToolRegistry` that the agent loop dispatches against.

Design notes that mirror the real system (cited in ``docs/03.tools.md``):

* **Dedicated tools over raw bash.** ``read_file`` / ``write_file`` /
  ``edit_file`` give the harness typed, gateable, auditable hooks instead of an
  opaque shell string. ``edit_file`` enforces a *staleness*/uniqueness check:
  the target string must appear exactly once, so an edit can't silently corrupt
  a file — exactly Claude Code's Edit semantics.
* **Read-only tools are parallel-safe.** ``glob``, ``grep``, ``read_file`` and
  ``list_dir`` set ``read_only=True`` so a scheduler may run them concurrently.
* **Bash has a timeout** and runs in a subprocess.

Everything operates relative to a configurable *root* so demos run safely inside
a temp dir. No proprietary source is used.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class ToolResult:
    """Result of running a tool. ``is_error`` maps to the API's tool_result flag."""

    content: str
    is_error: bool = False


@dataclass
class Tool:
    """A callable tool plus the metadata the model and harness need."""

    name: str
    description: str
    parameters: dict[str, Any]            # JSON-Schema for the inputs
    func: Callable[..., ToolResult]
    read_only: bool = False               # parallel-safe + no permission gate

    def schema(self) -> dict[str, Any]:
        """The schema advertised to the backend (API ``tools[]`` entry shape)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "read_only": self.read_only,
        }


class ToolRegistry:
    """Holds the tools, builds their schemas, and dispatches calls.

    The registry is rooted at a directory; all path arguments resolve relative
    to it and are confined inside it (a basic sandbox — see
    ``docs/04.permissions.md``).
    """

    def __init__(self, root: str | os.PathLike[str] = ".") -> None:
        self.root = Path(root).resolve()
        self._tools: dict[str, Tool] = {}
        self._register_builtins()

    # -- registration -------------------------------------------------------

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Error: unknown tool {name!r}", is_error=True)
        try:
            return tool.func(**args)
        except TypeError as e:
            return ToolResult(f"Error: bad arguments for {name}: {e}", is_error=True)
        except Exception as e:  # tools should not crash the agent loop
            return ToolResult(f"Error: {type(e).__name__}: {e}", is_error=True)

    # -- path safety --------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Resolve ``path`` under the root, rejecting escapes via ``..``."""
        p = (self.root / path).resolve()
        if self.root != p and self.root not in p.parents:
            raise ValueError(f"path {path!r} escapes the sandbox root")
        return p

    # -- built-in tools -----------------------------------------------------

    def _register_builtins(self) -> None:
        self.register(Tool(
            name="read_file",
            description="Read a UTF-8 text file and return its contents with line numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root."},
                },
                "required": ["path"],
            },
            func=self._read_file,
            read_only=True,
        ))
        self.register(Tool(
            name="write_file",
            description="Write (create or overwrite) a UTF-8 text file. Creates parent dirs.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            func=self._write_file,
        ))
        self.register(Tool(
            name="edit_file",
            description=(
                "Replace an exact substring in a file. `old` must occur exactly once "
                "(prevents ambiguous/corrupting edits)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "Exact text to replace; must be unique."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old", "new"],
            },
            func=self._edit_file,
        ))
        self.register(Tool(
            name="bash",
            description="Run a shell command in the workspace root with a timeout. Returns stdout+stderr.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds (default 30)."},
                },
                "required": ["command"],
            },
            func=self._bash,
        ))
        self.register(Tool(
            name="glob",
            description="Find files matching a glob pattern (e.g. '**/*.py'). Returns matching paths.",
            parameters={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
            func=self._glob,
            read_only=True,
        ))
        self.register(Tool(
            name="grep",
            description="Search file contents for a regex. Returns matching 'path:lineno: line' rows.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string", "description": "Optional path filter (default '**/*')."},
                },
                "required": ["pattern"],
            },
            func=self._grep,
            read_only=True,
        ))
        self.register(Tool(
            name="list_dir",
            description="List the entries of a directory (default the workspace root).",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
            func=self._list_dir,
            read_only=True,
        ))

    # -- implementations ----------------------------------------------------

    def _read_file(self, path: str) -> ToolResult:
        p = self._resolve(path)
        if not p.is_file():
            return ToolResult(f"Error: no such file {path!r}", is_error=True)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        numbered = "\n".join(f"{i:>4}\t{line}" for i, line in enumerate(lines, 1))
        return ToolResult(numbered or "(empty file)")

    def _write_file(self, path: str, content: str) -> ToolResult:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        n = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return ToolResult(f"Wrote {len(content)} bytes ({n} lines) to {path}.")

    def _edit_file(self, path: str, old: str, new: str) -> ToolResult:
        p = self._resolve(path)
        if not p.is_file():
            return ToolResult(f"Error: no such file {path!r}", is_error=True)
        text = p.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return ToolResult(f"Error: `old` not found in {path}.", is_error=True)
        if count > 1:
            return ToolResult(
                f"Error: `old` occurs {count} times in {path}; must be unique.",
                is_error=True,
            )
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return ToolResult(f"Edited {path} (1 replacement).")

    def _bash(self, command: str, timeout: int = 30) -> ToolResult:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.root, capture_output=True,
                text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"Error: command timed out after {timeout}s.", is_error=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        out = out.strip() or "(no output)"
        prefix = "" if proc.returncode == 0 else f"[exit {proc.returncode}]\n"
        return ToolResult(prefix + out, is_error=proc.returncode != 0)

    def _glob(self, pattern: str) -> ToolResult:
        matches = sorted(
            str(p.relative_to(self.root))
            for p in self.root.glob(pattern)
            if p.is_file()
        )
        return ToolResult("\n".join(matches) if matches else "(no matches)")

    def _grep(self, pattern: str, glob: str = "**/*") -> ToolResult:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(f"Error: bad regex: {e}", is_error=True)
        rows: list[str] = []
        for p in sorted(self.root.glob(glob)):
            if not p.is_file():
                continue
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        rows.append(f"{p.relative_to(self.root)}:{i}: {line.rstrip()}")
            except OSError:
                continue
        return ToolResult("\n".join(rows) if rows else "(no matches)")

    def _list_dir(self, path: str = ".") -> ToolResult:
        p = self._resolve(path)
        if not p.is_dir():
            return ToolResult(f"Error: not a directory {path!r}", is_error=True)
        entries = sorted(
            f"{e.name}/" if e.is_dir() else e.name for e in p.iterdir()
        )
        return ToolResult("\n".join(entries) if entries else "(empty directory)")


# Convenience for `python -c` style checks.
if __name__ == "__main__":  # pragma: no cover
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        reg = ToolRegistry(d)
        print(reg.call("write_file", {"path": "a.txt", "content": "hi\nthere\n"}).content)
        print(reg.call("read_file", {"path": "a.txt"}).content)
        print(reg.call("edit_file", {"path": "a.txt", "old": "hi", "new": "hello"}).content)
        print(reg.call("grep", {"pattern": "hello"}).content)
        print(reg.call("bash", {"command": "echo ran"}).content)

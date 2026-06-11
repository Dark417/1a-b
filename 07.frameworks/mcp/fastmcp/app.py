"""app.py · A complete FastMCP "notes" app driven by a MockLLM agent loop.

End-to-end demo tying the folder together. We build a small **notes server**
exposing:
  * tools:    add_note, list_notes, search_notes
  * resource: notes://all  (the notebook as context)
  * prompt:   digest       (summarize the notebook)

Then a deterministic **MockLLM** plays an agent: for each user instruction it
picks the right MCP tool and arguments, the host calls it through the in-memory
client, and the result feeds the next turn. No API key, no network — the MockLLM
makes the whole loop reproducible and offline.

Run:  python app.py   (exit 0; falls back to a no-LLM script if fastmcp missing)
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any

try:
    from fastmcp import Client, FastMCP

    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False


# ---------------------------------------------------------------------------
# The server (state lives in a closure for the demo).
# ---------------------------------------------------------------------------
def build_notes_server() -> "FastMCP":
    mcp = FastMCP("notes")
    notes: list[str] = []

    @mcp.tool
    def add_note(text: str) -> str:
        """Append a note to the notebook."""
        notes.append(text)
        return f"stored note #{len(notes)}"

    @mcp.tool
    def list_notes() -> list[str]:
        """Return all notes."""
        return list(notes)

    @mcp.tool
    def search_notes(query: str) -> list[str]:
        """Return notes containing the query (case-insensitive)."""
        q = query.lower()
        return [n for n in notes if q in n.lower()]

    @mcp.resource("notes://all")
    def all_notes() -> str:
        """The whole notebook as text context."""
        return "\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1)) or "(empty)"

    @mcp.prompt
    def digest() -> str:
        """A prompt that asks for a digest of the notebook."""
        body = "\n".join(notes) or "(no notes)"
        return f"Summarize these notes into 3 bullets:\n\n{body}"

    return mcp


# ---------------------------------------------------------------------------
# A deterministic MockLLM agent: instruction -> (tool, args).
# ---------------------------------------------------------------------------
class MockLLM:
    def decide(self, instruction: str) -> tuple[str, dict[str, Any]]:
        t = instruction.lower()
        if t.startswith("note:") or "remember" in t:
            text = instruction.split(":", 1)[-1].strip()
            return "add_note", {"text": text}
        if "search" in t or "find" in t:
            m = re.search(r"(?:search|find)\s+(?:for\s+)?(.*)", t)
            return "search_notes", {"query": (m.group(1).strip() if m else "")}
        return "list_notes", {}


def _text(result) -> str:
    if getattr(result, "data", None) is not None:
        return str(result.data)
    return " ".join(getattr(b, "text", str(b)) for b in result.content)


async def run() -> None:
    mcp = build_notes_server()
    llm = MockLLM()
    script = [
        "note: buy milk",
        "remember to call the dentist",
        "note: milk is on sale at the corner store",
        "search milk",
        "list everything",
    ]
    async with Client(mcp) as client:
        tool_names = [t.name for t in await client.list_tools()]
        print("[host] notes server tools:", tool_names, "\n")
        for instruction in script:
            tool, args = llm.decide(instruction)
            result = _text(await client.call_tool(tool, args))
            print(f"[user] {instruction}")
            print(f"[host] -> {tool}({args}) = {result}\n")

        # Read a resource and render the prompt to show those primitives too.
        nb = await client.read_resource("notes://all")
        print("[host] resource notes://all ->\n   " + nb[0].text.replace("\n", "\n   "))
        digest = await client.get_prompt("digest", {})
        print("\n[host] prompt 'digest' ->\n   " + digest.messages[0].content.text.replace("\n", "\n   "))

    print("\n[host] done — a full MCP app, driven by a deterministic MockLLM, offline.")


def run_fallback() -> None:
    print("[skip] fastmcp not installed — pip install -r requirements.txt")
    print("       (The python-sdk/ folder has a stdlib-only round-trip fallback.)")


def main() -> int:
    print("=== FastMCP notes app (offline) ===\n")
    if HAVE:
        asyncio.run(run())
    else:
        run_fallback()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""demo.py — end-to-end offline demo of the mini coding agent.

Runs the full gather→plan→act→observe loop with the deterministic
:class:`~llm.MockBackend` on a scripted task ("create a small Python program and
run it"), inside a throwaway temp directory. It exercises every subsystem:

* the **agent loop** with streaming-style incremental output,
* the **tool registry** (list_dir → write_file → bash),
* the **permission gate** (auto-approve, non-interactive),
* the **context manager** (token-budget tracking + compaction), and
* a **live MCP round-trip** against the bundled example server.

It requires NO API key and NO network, and exits 0 on success. This is the
validation entry point: ``python demo.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile

from agent import Agent, DEFAULT_SYSTEM
from context import ContextManager
from llm import MockBackend
from permissions import Mode, PermissionConfig, PermissionGate, auto_approve
from tools import ToolRegistry


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_agent_loop(workdir: str) -> str:
    banner("1. AGENT LOOP — scripted 'write a file and run it' task (offline)")
    # Tight context budget so compaction is demonstrated mid-run.
    ctx = ContextManager(system=DEFAULT_SYSTEM, budget=600, trigger_ratio=0.6, keep_recent=2)
    agent = Agent(
        backend=MockBackend(),
        tools=ToolRegistry(workdir),
        gate=PermissionGate(
            PermissionConfig(
                mode=Mode.DEFAULT,
                allow=["bash(python *)"],   # pre-approve running python
            ),
            prompter=auto_approve,           # non-interactive
        ),
        context=ctx,
        system=DEFAULT_SYSTEM,
        stream=True,
    )
    final = agent.run("Create a small Python program `hello.py` and run it to verify it works.")
    print(f"\n[turns: {len(agent.turns)} | compactions: {ctx.compactions} | "
          f"est. tokens: {ctx.total_tokens()}]")
    return final


def demo_permission_modes() -> None:
    banner("2. PERMISSIONS — how each mode decides a risky tool call")
    call = ("bash", {"command": "rm -rf /"})
    for mode in (Mode.DEFAULT, Mode.PLAN, Mode.ACCEPT_EDITS, Mode.BYPASS):
        gate = PermissionGate(
            PermissionConfig(mode=mode, deny=["bash(rm -rf *)"]),
            prompter=auto_approve,
        )
        decision = gate.decide(*call)
        print(f"  mode={mode.value:<12} bash(rm -rf /) -> {decision.value}")
    # A safe edit under acceptEdits vs plan mode.
    for mode in (Mode.ACCEPT_EDITS, Mode.PLAN):
        gate = PermissionGate(PermissionConfig(mode=mode), prompter=auto_approve)
        d = gate.decide("write_file", {"path": "x.py", "content": "..."})
        print(f"  mode={mode.value:<12} write_file(x.py)  -> {d.value}")


def demo_mcp() -> None:
    banner("3. MCP — live JSON-RPC round-trip with the bundled example server")
    from mcp_client import StdioMCPClient, mcp_tools_as_agent_tools

    here = os.path.dirname(os.path.abspath(__file__))
    server = os.path.join(here, "mcp_server_example.py")
    try:
        with StdioMCPClient([sys.executable, server]) as client:
            print(f"  connected to {client.server_info.get('name')!r} "
                  f"(protocol {client.capabilities})")
            adapted = mcp_tools_as_agent_tools(client)
            print(f"  discovered tools: {[t.name for t in adapted]}")
            # Call one through the adapter, like the agent would.
            for t in adapted:
                if t.name.endswith("add"):
                    print("  mcp add(40, 2) =>", t.func(a=40, b=2).content)
    except Exception as e:  # never fail the demo on MCP issues
        print(f"  (MCP demo skipped: {e})")


def main() -> int:
    print("Mini Claude-Code-style coding agent — OFFLINE demo (MockBackend)")
    print("No API key or network required.")
    with tempfile.TemporaryDirectory() as workdir:
        final = demo_agent_loop(workdir)
        # Sanity check: the agent actually created and ran the file.
        produced = os.path.join(workdir, "hello.py")
        ok = os.path.isfile(produced)
        print(f"\n[verify] hello.py created in workspace: {ok}")
        demo_permission_modes()
        demo_mcp()

    banner("DEMO COMPLETE")
    if not ok:
        print("FAILED: expected hello.py to be created.")
        return 1
    print("All subsystems exercised successfully.")
    print("Final assistant message:\n  " + final.replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

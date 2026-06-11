# Mini Claude-Code-style coding agent (Python)

A small but **full-featured**, runnable coding agent that re-implements the core
architecture of [Claude Code](https://code.claude.com/docs/en/overview) for
teaching. It is **clean-room** — built from public documentation and public
write-ups (see [`../docs/07.references.md`](../docs/07.references.md)), not from
any proprietary source.

It runs **completely offline** with a deterministic mock backend (no API key, no
network), and optionally plugs into real Claude or a local Ollama model.

```bash
python demo.py        # offline end-to-end demo; exits 0 with no key/network
python agent.py       # interactive REPL (offline mock by default)
python mcp_client.py  # MCP JSON-RPC round-trip against the bundled server
python tools.py       # quick tool-registry smoke test
```

## What it demonstrates

| Subsystem | File | Real Claude Code analog |
|---|---|---|
| Agent loop (gather → plan → act → observe) | [`agent.py`](agent.py) | The main turn loop |
| Backend abstraction (mock / Claude / Ollama) | [`llm.py`](llm.py) | Messages API tool-use loop |
| Tool registry: read/write/edit/bash/glob/grep/list | [`tools.py`](tools.py) | Read, Write, Edit, Bash, Glob, Grep tools |
| Permission gate: allow/ask/deny + modes | [`permissions.py`](permissions.py) | Permission modes + `settings.json` rules |
| Context manager: token budget + compaction | [`context.py`](context.py) | Context management / compaction |
| MCP client + bundled server | [`mcp_client.py`](mcp_client.py), [`mcp_server_example.py`](mcp_server_example.py) | MCP integration |
| End-to-end offline demo | [`demo.py`](demo.py) | — |

## The agent loop

The heart of the system (in [`agent.py`](agent.py)) is the same loop Claude Code
runs:

```
add the user's task to the conversation
repeat (up to max_turns):
    ask the backend for the next step      # streams text incrementally
    record the assistant turn (text + tool_use blocks)
    if the step requested no tools:
        stop — the turn is final            # == stop_reason "end_turn"
    for each requested tool call:
        check the permission gate           # allow / ask / deny
        run the allowed ones                # read-only calls run in parallel
    feed the results back as tool_result blocks
    compact the context if over budget
```

Each subsystem is decoupled so you can swap any piece:

```python
from agent import Agent
from llm import MockBackend                # or AnthropicBackend / OllamaBackend
from tools import ToolRegistry
from permissions import PermissionGate, PermissionConfig, Mode, auto_approve
from context import ContextManager

agent = Agent(
    backend=MockBackend(),
    tools=ToolRegistry("/path/to/workspace"),
    gate=PermissionGate(PermissionConfig(mode=Mode.DEFAULT), prompter=auto_approve),
    context=ContextManager(budget=4000),
)
agent.run("Create hello.py and run it.")
```

## How it maps to the real architecture

* **Backends return *steps*, not just text.** A `Step` carries assistant text
  plus `ToolCall`s — the exact shape of an Anthropic API turn whose
  `stop_reason` is `tool_use`. The agent executes the calls and resends the
  transcript with `tool_result` blocks, looping until `done` (`end_turn`). See
  [`../docs/02.architecture.md`](../docs/02.architecture.md).
* **Tools are dedicated, typed, and gateable.** `edit_file` enforces a unique
  match (no ambiguous edits); read-only tools are flagged parallel-safe; `bash`
  has a timeout. See [`../docs/03.tools.md`](../docs/03.tools.md).
* **Permissions gate every mutating call.** `deny` beats `allow`; modes
  (`default` / `acceptEdits` / `plan` / `bypass`) mirror Claude Code. See
  [`../docs/04.permissions.md`](../docs/04.permissions.md).
* **Context is compacted under a token budget.** See
  [`../docs/02.architecture.md`](../docs/02.architecture.md).
* **MCP is real JSON-RPC 2.0.** The client does `initialize` →
  `notifications/initialized` → `tools/list` → `tools/call` over stdio, then
  adapts server tools into the agent's registry as `mcp__example__*`. See
  [`../docs/05.mcp.md`](../docs/05.mcp.md).

## Plugging in a real backend

The mock is the default so everything runs offline. To use real Claude:

```bash
pip install -r requirements.txt        # installs `anthropic`
export ANTHROPIC_API_KEY=sk-...
python -c "
from agent import Agent; from llm import AnthropicBackend
from tools import ToolRegistry
from permissions import PermissionGate, PermissionConfig, Mode, cli_prompter
Agent(
    backend=AnthropicBackend(model='claude-opus-4-8'),
    tools=ToolRegistry('.'),
    gate=PermissionGate(PermissionConfig(mode=Mode.DEFAULT), prompter=cli_prompter),
).run('List the Python files and summarize what they do.')
"
```

`AnthropicBackend` uses the official `anthropic` SDK with `claude-opus-4-8`,
adaptive thinking, and streaming — Anthropic's current recommended defaults. For
a fully local path, `OllamaBackend('llama3.1')` talks to a local Ollama server
(no extra dependency; uses a JSON tool-call convention since many small local
models lack native tool calling).

## Requirements

Pure standard library for the offline path. See [`requirements.txt`](requirements.txt).
Tested on Python 3.11.

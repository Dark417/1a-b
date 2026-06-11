# Mini Claude-Code-style coding agent (Ruby)

The same full-featured mini coding agent as [`../python/`](../python/),
reimplemented idiomatically in Ruby. It re-creates the core architecture of
[Claude Code](https://code.claude.com/docs/en/overview) for teaching, and is
**clean-room** — built from public documentation and write-ups (see
[`../docs/07.references.md`](../docs/07.references.md)), not proprietary source.

Runs **completely offline** with a deterministic mock backend (no API key, no
network, **no gems** — pure stdlib).

```bash
ruby demo.rb           # offline end-to-end demo; exits 0 with no key/network
ruby agent.rb          # interactive REPL (offline mock by default)
ruby mcp_client.rb     # MCP JSON-RPC round-trip against the bundled server
ruby tools.rb          # quick tool-registry smoke test
```

Tested on Ruby 3.3.

## Files

| File | Role | Claude Code analog |
|---|---|---|
| [`agent.rb`](agent.rb) | Agent loop (gather → plan → act → observe) + REPL | The main turn loop |
| [`llm.rb`](llm.rb) | Backend abstraction (mock / Claude / Ollama) | Messages API tool-use loop |
| [`tools.rb`](tools.rb) | Tool registry: read/write/edit/bash/glob/grep/list | Read, Write, Edit, Bash, Glob, Grep |
| [`permissions.rb`](permissions.rb) | Permission gate: allow/ask/deny + modes | Permission modes + `settings.json` |
| [`context.rb`](context.rb) | History + token-budget compaction | Context management / compaction |
| [`mcp_client.rb`](mcp_client.rb) / [`mcp_server_example.rb`](mcp_server_example.rb) | MCP client + bundled server | MCP integration |
| [`demo.rb`](demo.rb) | End-to-end offline demo | — |

## The agent loop

Same loop as the real product (in [`agent.rb`](agent.rb)):

```
add the user's task to the conversation
repeat (up to max_turns):
    ask the backend for the next step       # streams text incrementally
    record the assistant turn (text + tool_use blocks)
    if no tools were requested:
        stop — the turn is final             # == stop_reason "end_turn"
    for each requested tool call:
        check the permission gate            # allow / ask / deny
        run the allowed ones                 # read-only calls run on threads
    feed the results back as tool_result blocks
    compact the context if over budget
```

Wiring is decoupled so any piece is swappable:

```ruby
require_relative "agent"

agent = MiniAgent::Agent.new(
  backend: MiniAgent::MockBackend.new,                 # or AnthropicBackend / OllamaBackend
  tools:   MiniAgent::ToolRegistry.new("/path/to/workspace"),
  gate:    MiniAgent::PermissionGate.new(
             MiniAgent::PermissionConfig.new(mode: MiniAgent::Mode::DEFAULT),
             MiniAgent::AUTO_APPROVE),
  context: MiniAgent::ContextManager.new(budget: 4000)
)
agent.run("Create hello.rb and run it.")
```

## How it maps to the real architecture

Identical to the Python notes — see [`../python/README.md`](../python/README.md)
and the docs:

* Backends return **steps** (text + `ToolCall`s), the shape of an API turn whose
  `stop_reason` is `tool_use`; the loop runs the calls and resends the
  transcript with `tool_result` blocks until `done`
  ([`../docs/02.architecture.md`](../docs/02.architecture.md)).
* Tools are **dedicated, typed, gateable**; `edit_file` enforces a unique match;
  read-only tools are parallel-safe; `bash` has a timeout
  ([`../docs/03.tools.md`](../docs/03.tools.md)).
* Permissions **gate every mutating call**; `deny` beats `allow`; modes
  `default` / `acceptEdits` / `plan` / `bypass`
  ([`../docs/04.permissions.md`](../docs/04.permissions.md)).
* Context is **compacted** under a token budget
  ([`../docs/02.architecture.md`](../docs/02.architecture.md)).
* MCP is **real JSON-RPC 2.0** over stdio: `initialize` →
  `notifications/initialized` → `tools/list` → `tools/call`
  ([`../docs/05.mcp.md`](../docs/05.mcp.md)).

## Plugging in a real backend

The mock is the default so everything runs offline with no gems. For real
Claude, install the `anthropic` gem (uncomment it in [`Gemfile`](Gemfile)) and
set `ANTHROPIC_API_KEY`, then use `MiniAgent::AnthropicBackend.new(model:
"claude-opus-4-8")` — it uses `claude-opus-4-8`, adaptive thinking, and
streaming (Anthropic's current recommended defaults). For a fully local path,
`MiniAgent::OllamaBackend.new(model: "llama3.1")` talks to a local Ollama server
over `Net::HTTP` (no gem; JSON tool-call convention since many small local
models lack native tool calling).

## Requirements

Pure Ruby standard library for the offline path. See [`Gemfile`](Gemfile).

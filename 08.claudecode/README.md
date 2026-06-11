# 08.claudecode — a clean-room, educational Claude-Code-style coding agent

> **Ethics / legal.** This section is built **only** from *public* documentation
> and *public* reverse-engineering write-ups (all cited in
> [`docs/07.references.md`](docs/07.references.md)). **No proprietary or leaked
> Anthropic source is used or redistributed.** The code is original, clean-room,
> and clearly labeled as an educational re-implementation of the *architecture*.

This section explains how [Claude Code](https://code.claude.com/docs/en/overview)
works and backs the explanation with a **small but full-featured coding agent**
you can read and run — implemented twice, in **Python** and **Ruby**.

## Layout

```
08.claudecode/
├── docs/                     architecture explainers (Markdown, cited)
│   ├── 01.overview.md          what Claude Code is; product surface; where it runs
│   ├── 02.architecture.md      the agent loop, context mgmt, streaming, compaction
│   ├── 03.tools.md             Read/Write/Edit/Bash/Glob/Grep/Agent/TodoWrite
│   ├── 04.permissions.md       allow/ask/deny, modes, sandboxing, hooks, settings.json
│   ├── 05.mcp.md               Model Context Protocol (servers/clients, transports)
│   ├── 06.skills-subagents.md  skills, sub-agents, slash commands
│   └── 07.references.md        full citation list
├── python/                   runnable mini coding-agent (stdlib-only offline path)
│   ├── agent.py llm.py tools.py permissions.py context.py
│   ├── mcp_client.py mcp_server_example.py demo.py
│   ├── README.md requirements.txt
└── ruby/                     the same agent, idiomatic Ruby (stdlib-only offline path)
    ├── agent.rb llm.rb tools.rb permissions.rb context.rb
    ├── mcp_client.rb mcp_server_example.rb demo.rb
    ├── README.md Gemfile
```

## Run it (offline — no API key, no network)

```bash
cd python && python demo.py     # exits 0; full agent loop + permissions + MCP
cd ruby   && ruby demo.rb       # exits 0; same, pure Ruby stdlib
```

Both demos run the canonical "create a small program and run it" task end-to-end
with a deterministic mock backend, show how each permission mode decides a risky
call, and perform a **live MCP JSON-RPC round-trip** against a bundled example
server — all offline.

## What the mini-agent re-implements

| Real Claude Code | Here |
|---|---|
| The agent loop (gather → plan → act → observe) | `agent.py` / `agent.rb` |
| Messages API tool-use turn (text + `tool_use` → `tool_result`) | `llm.py` / `llm.rb` |
| Read/Write/Edit/Bash/Glob/Grep tools, parallel-safe scheduling | `tools.py` / `tools.rb` |
| Permission modes + allow/ask/deny rules (`settings.json`) | `permissions.py` / `permissions.rb` |
| Context management + compaction under a token budget | `context.py` / `context.rb` |
| MCP integration (JSON-RPC 2.0 over stdio) | `mcp_client.py` + server (and Ruby twins) |

The mock backend is the **default** so everything runs with no key or network.
Optional `AnthropicBackend` (real Claude, `claude-opus-4-8`, adaptive thinking,
streaming) and `OllamaBackend` (local open models) are provided and degrade
gracefully when their dependency/key is absent.

Start with [`docs/01.overview.md`](docs/01.overview.md), then read the code.

# FastMCP — the batteries-included MCP framework

> **FastMCP** is the fast, Pythonic way to build **MCP servers and clients**. You
> write decorated functions; FastMCP infers schemas from type hints and handles
> the protocol, transports, and lifecycle. The *original* FastMCP (v1) was
> contributed into the official `mcp` SDK as `mcp.server.fastmcp.FastMCP`;
> **FastMCP v2/v3** (the standalone `fastmcp` package) is a **superset** that adds
> an in-memory test client, authentication, server composition & proxying,
> OpenAPI/FastAPI→MCP generation, and a CLI.
>
> Docs: <https://gofastmcp.com> · Repo: <https://github.com/jlowin/fastmcp>

---

## 1. The mental model

If you read `../python-sdk/`, you know MCP is JSON-RPC 2.0 with a lifecycle and
three primitives (tools, resources, prompts). FastMCP's pitch: **never touch the
protocol.** Decorate a function, get a tool. The framework owns schema inference,
content-block packing, the lifecycle handshake, transports, and error mapping.

```
your typed Python fn  ──@mcp.tool──►  FastMCP  ──►  JSON-RPC server
                                         ▲                 │
   in-memory / stdio / HTTP transport ───┘                 ▼
                                                     any MCP host
```

The single biggest convenience for *testing* is the **in-memory transport**:
`Client(server_object)` connects a client straight to a server object in the same
process — no subprocess, no sockets — so your tests run a real MCP round-trip
instantly and **offline**. Every file in this folder uses it.

## 2. FastMCP (standalone) vs. `mcp.server.fastmcp`

| | `mcp.server.fastmcp.FastMCP` (in `mcp`) | `fastmcp` v2/v3 (standalone) |
|---|---|---|
| Build tools/resources/prompts | yes | yes |
| Client | use SDK's `ClientSession` | built-in `Client` + **in-memory** transport |
| Auth (OAuth, bearer, JWT) | minimal | first-class |
| Server **composition** (`mount`, `import`) | no | yes |
| **Proxy** another MCP server | no | yes |
| Generate MCP from **OpenAPI / FastAPI** | no | yes |
| CLI (`fastmcp run/dev/install`) | partial (`mcp` cmd) | full |
| Testing ergonomics | spawn subprocess | in-memory client |

Rule of thumb: prototyping a single local server → either works; building a real,
authenticated, composed, or API-wrapping server → reach for standalone `fastmcp`.

## 3. Install

```bash
pip install -r requirements.txt      # fastmcp>=2.0
# recommended (Astral uv):
uv pip install fastmcp

# scaffold & run servers from the CLI:
fastmcp run app.py:mcp               # run a server object named `mcp`
fastmcp dev app.py:mcp               # run + open the MCP Inspector (needs Node)
fastmcp install claude-desktop app.py:mcp   # register into Claude Desktop config
```

> Everything here also runs with **no install**: each file skips gracefully if
> `fastmcp` is missing and exits 0.

## 4. The files in this tutorial

| File | Teaches |
|---|---|
| [`01.decorators.py`](01.decorators.py) | `@tool` / `@resource` / `@prompt`, schema inference from type hints, templated resources, structured outputs. |
| [`02.in_memory_client.py`](02.in_memory_client.py) | The in-memory `Client(server)` — a real MCP round-trip with zero transport, calling every primitive. |
| [`03.context_and_errors.py`](03.context_and_errors.py) | The `Context` object (logging, progress, state) and clean error handling with `ToolError`. |
| [`04.composition.py`](04.composition.py) | Server **composition**: build small servers and `import_server`/`mount` them into one, with prefixes. |
| [`app.py`](app.py) | A complete "notes" server (tools + resource + prompt) driven by a deterministic **MockLLM** agent loop over the in-memory client. |

Each `.py` runs standalone (`python 0X_*.py`, exit 0).

## 5. Feature tour (snippets)

**A tool** — types become the schema, the docstring becomes the description:
```python
from fastmcp import FastMCP
mcp = FastMCP("demo")

@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
```

**A templated resource** — the `{}` segment is a parameter:
```python
@mcp.resource("user://{uid}/profile")
def profile(uid: str) -> dict:
    return {"id": uid, "name": f"user-{uid}"}
```

**A prompt** — a reusable message template:
```python
@mcp.prompt
def review(code: str) -> str:
    return f"Review this code:\n\n{code}"
```

**Test it instantly, in-memory:**
```python
from fastmcp import Client
async with Client(mcp) as c:                 # no subprocess, no socket
    print((await c.call_tool("add", {"a": 2, "b": 3})).data)   # 5
```

**`Context`** — logging/progress/state from inside a handler:
```python
from fastmcp import Context

@mcp.tool
async def crunch(n: int, ctx: Context) -> int:
    await ctx.info(f"crunching {n}")
    await ctx.report_progress(progress=n, total=n)
    return n * n
```

**Composition** — assemble big servers from small ones:
```python
main = FastMCP("main")
await main.import_server(math_server, prefix="math")   # math_add, math_sub, ...
```

## 6. Comparison to alternatives

* **vs. raw `mcp` SDK** — FastMCP *is* the high-level half of that SDK, plus a
  client and production features. Choose standalone for auth/composition/proxy.
* **vs. a plain function-calling registry** — same as MCP generally: FastMCP
  gives you an out-of-process, language-agnostic, discoverable contract that any
  MCP host can consume, instead of an app-specific in-process tool list.
* **vs. LangChain tools / OpenAI function calling** — those are framework- or
  vendor-specific; an MCP server written with FastMCP works with Claude Desktop,
  Claude Code, IDEs, and any other MCP host unchanged.

## 7. Gotchas

* **stdout is sacred on stdio.** Never `print()` to stdout from a stdio server —
  use `ctx.info(...)` or log to stderr. (The in-memory transport sidesteps this,
  which is one reason it's so handy for tests.)
* **Annotate everything.** Missing type hints produce a loose schema; the model
  then mis-calls the tool. Give every param a type and the function a docstring.
* **`@mcp.tool` vs `@mcp.tool()`** — v2/v3 accept both the bare decorator and the
  called form; pick one style and be consistent.
* **`call_tool` returns a result object.** Read `.data` (structured) or iterate
  `.content` blocks for text; don't assume it's a bare string.
* **Async handlers must not block.** Use `await` for I/O; a blocking call stalls
  the event loop and every other request on that server.
* **Composition prefixes.** When you `import_server(..., prefix="math")`, tools
  surface as `math_add` etc. — mind name collisions across mounted servers.

## 8. References

* FastMCP docs — <https://gofastmcp.com>
* FastMCP repo — <https://github.com/jlowin/fastmcp>
* MCP spec — <https://modelcontextprotocol.io>
* Official SDK's FastMCP — `../python-sdk/02.fastmcp_server.py`

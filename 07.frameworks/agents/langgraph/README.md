# LangGraph — agents as explicit state graphs

**LangGraph** models an agent as a **directed graph** of nodes (Python
functions) that read and write a shared, typed **state**. Edges — plain or
**conditional** — decide what runs next. On top of this tiny core it ships the
production machinery real agents need: **persistence/checkpointing**,
**human-in-the-loop interrupts**, **streaming**, **time-travel**, **subgraphs**,
and a prebuilt **ReAct** agent. It is built by the LangChain team and is the
orchestration layer behind many production agents.

- Docs: https://langchain-ai.github.io/langgraph/
- Concepts: https://langchain-ai.github.io/langgraph/concepts/
- Repo: https://github.com/langchain-ai/langgraph

> This folder runs **fully offline**. Every example uses a deterministic
> `ScriptedChatModel` (a real `langchain_core.BaseChatModel` subclass whose
> replies are scripted) so the *real* LangGraph runtime executes — no API key,
> no network. Where a package is missing the file prints a note and runs a
> pure-Python fallback. See `00.mock_llm.py`.

---

## When to use it

| Use LangGraph when… | Reach for something else when… |
|---|---|
| You need an explicit, inspectable **control flow** (cycles, branches, retries). | A single linear "prompt → tool → answer" is enough (use the prebuilt agent or plain LangChain). |
| You need **durable** runs: pause, persist, resume, recover from a crash. | The task is stateless and short-lived. |
| You need **human-in-the-loop** approval mid-run. | Fully autonomous batch jobs. |
| You want **multi-agent** topologies (supervisor, swarm) with shared state. | A two-line role-play (CrewAct/AutoGen may be terser). |
| You want **streaming** of tokens *and* intermediate state. | — |

LangGraph is the "assembly language" of agent frameworks: more verbose than
CrewAI/AutoGen, but nothing is hidden — you own the graph.

---

## Architecture in words

```
            ┌─────────────────────────── State (TypedDict) ───────────────────────────┐
            │   a single typed dict threaded through every node; channels are merged   │
            │   by REDUCERS (e.g. messages use add_messages → append instead of clobber)│
            └──────────────────────────────────────────────────────────────────────────┘
                              ▲ read/write                       ▲ read/write
   START ──▶ ┌──────────┐  conditional   ┌──────────┐  normal   ┌──────────┐ ──▶ END
            │  node A   │ ───edge?──────▶ │  node B  │ ─edge────▶│  node C  │
            │ (a func:  │                 │ (a func) │           │ (a func) │
            │  state→Δstate)              └──────────┘           └──────────┘
            └──────────┘
                 │
                 └── checkpointer (MemorySaver / SqliteSaver) snapshots state after every
                     super-step → enables resume, time-travel, and human-in-the-loop interrupts
```

Mental model — five pieces:

1. **State** — a `TypedDict` (or Pydantic model). Each key is a *channel*.
   A channel may have a **reducer** that says how concurrent writes merge.
   The canonical one is `add_messages` for a running chat transcript.
2. **Nodes** — functions `(state) -> partial_state`. Whatever dict they return
   is merged into state via the channel reducers.
3. **Edges** — `add_edge(A, B)` (always go A→B) or
   `add_conditional_edges(A, router_fn, {...})` (router returns the next node
   name). Special nodes `START` and `END`.
4. **Compile** — `graph.compile(checkpointer=…)` turns the builder into a
   runnable `Pregel` app you `.invoke` / `.stream`.
5. **Runtime** — executes in **super-steps** (BSP): all ready nodes run, writes
   are applied, repeat. A `checkpointer` snapshots state after each super-step,
   keyed by a `thread_id` in the `config`. That snapshot is what makes
   persistence, resume, time-travel, and `interrupt()` possible.

---

## Install

```bash
pip install -r requirements.txt   # langgraph + langchain-core
```

Everything here runs with just those two pure-Python packages. A live model is
optional: set `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` and swap `ScriptedChatModel`
for `ChatOpenAI`/`ChatAnthropic` (commented in each file).

---

## Feature tour (one runnable file each)

| File | Feature | Key API |
|---|---|---|
| `00.mock_llm.py` | the offline `ScriptedChatModel` + tools every file imports | `BaseChatModel`, `@tool` |
| `01.state_graph.py` | minimal `StateGraph`, nodes, edges, `TypedDict` state | `StateGraph`, `START`, `END` |
| `02.conditional_edges.py` | branching / routing; the **agent loop** as a cycle | `add_conditional_edges` |
| `03.tool_calling.py` | bind tools, `ToolNode`, prebuilt `create_react_agent` | `ToolNode`, `create_react_agent` |
| `04.persistence.py` | checkpointing, `thread_id`, resume, `get_state_history` (time-travel) | `MemorySaver`, `config` |
| `05.human_in_the_loop.py` | `interrupt()` mid-graph, approve/edit, resume with `Command` | `interrupt`, `Command` |
| `06.streaming.py` | stream **state** (`updates`/`values`) and **tokens** (`messages`) | `.stream(stream_mode=…)` |
| `07.multi_agent.py` | supervisor routing between worker agents over shared state | conditional edges + subgraphs |
| `08.structured_and_retries.py` | structured output + node-level retry/error handling | `RetryPolicy`, pydantic |
| `app.py` | end-to-end **research assistant**: plan → tools → reflect → HITL → report | everything above |

Run any of them:

```bash
python 01.state_graph.py
python app.py
```

---

## Gotchas

- **Reducers, not assignment.** Returning `{"messages": [m]}` from a node
  *appends* (because of `add_messages`); for non-reduced channels a returned
  value **replaces**. Forgetting a reducer is the #1 "my list got clobbered" bug.
- **State is the contract.** Every node must return keys that exist in the state
  schema; unknown keys raise. Return *only* what changed.
- **Checkpointer needs a `thread_id`.** Persistence/HITL/streaming-resume all
  require `config={"configurable": {"thread_id": "..."}}`. No thread_id → no
  memory across `.invoke` calls.
- **`interrupt()` raises and resumes.** The node *re-runs from the top* on
  resume, so put `interrupt()` early or make the node idempotent.
- **`create_react_agent` moved.** In LangGraph ≥1.0 it's also exported from
  `langchain.agents.create_agent`; the `langgraph.prebuilt` path still works but
  warns. We use the prebuilt path and silence the deprecation note.
- **Recursion limit.** A mis-wired cycle loops forever; LangGraph stops at
  `recursion_limit` (default 25) and raises — bump it in `config` when intended.

---

## Comparison to alternatives

| | LangGraph | CrewAI | AutoGen |
|---|---|---|---|
| Core abstraction | state graph | roles + tasks | conversing agents |
| Control flow | explicit (you draw it) | sequential/hierarchical process | emergent from chat |
| Persistence/HITL | first-class | limited | via teams/state |
| Best for | durable, branchy workflows | role-based pipelines | open-ended collaboration |

LangGraph trades brevity for control. If you can draw the flowchart, LangGraph
makes it executable, durable, and resumable.

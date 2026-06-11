"""LangChain RAG 08 — agentic RAG (retrieval as a tool).

In *agentic* RAG the LLM is not forced to retrieve once and answer. Instead the
retriever is exposed as a **tool**, and an agent decides *whether*, *when*, and
*with what query* to search — enabling multi-hop questions ("biggest planet, and
how many moons does the red planet have?") by issuing several searches.

LangChain builds this with ``create_retriever_tool`` + an agent (or, more
commonly today, a LangGraph ``ReAct`` agent). Here we implement a tiny,
transparent ReAct-style loop over the harness so the control flow is visible and
runs offline.

Docs: https://python.langchain.com/docs/how_to/qa_sources/  (tools)
      https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/
"""

from __future__ import annotations

import re
import sys

sys.path.insert(0, ".")
from _rag_common import (  # noqa: E402
    LocalEmbeddings,
    MiniVectorStore,
    MockLLM,
    banner,
    note,
    sample_documents,
    seed_everything,
)

QUESTION = "What is the largest planet, and how many moons does the red planet have?"


def make_retriever_tool(store):
    """Return a callable tool the agent can invoke with a search string."""
    def retrieve(query: str):
        return [d for d, _ in store.similarity(query, k=1)]
    return retrieve


def plan_subqueries(question):
    """Deterministic 'agent planner': split a compound question into searches."""
    parts = re.split(r",| and ", question)
    subs = []
    for p in parts:
        p = p.strip().rstrip("?")
        if "largest" in p or "biggest" in p:
            subs.append("largest planet in the Solar System")
        elif "red planet" in p or "moons" in p:
            subs.append("how many moons does Mars have")
    return subs or [question]


def main():
    seed_everything()
    banner("LangChain 08 — agentic RAG (retrieval-as-a-tool)")
    print(f"Q: {QUESTION}\n")

    try:
        from langchain_core.tools import tool  # noqa: F401

        note("langchain_core.tools available (live tool decorator)")
    except Exception as e:  # pragma: no cover
        note(f"langchain_core.tools unavailable ({e}); plain tool callable")

    store = MiniVectorStore(LocalEmbeddings())
    store.add(sample_documents())
    retrieve = make_retriever_tool(store)
    llm = MockLLM()

    # Agent loop: plan -> for each sub-query call the retrieval tool -> gather.
    subqueries = plan_subqueries(QUESTION)
    print("Agent decided to issue these searches:")
    gathered = []
    for sq in subqueries:
        hits = retrieve(sq)
        print(f"   TOOL search({sq!r}) -> {hits[0]['id']}")
        gathered.extend(hits)

    # Synthesise a final answer from everything gathered.
    context = "\n".join(d["text"] for d in gathered)
    print("\nFinal synthesis:")
    for sq in subqueries:
        print(f"   {llm.answer(sq, context)}")

    print("\nOK")


if __name__ == "__main__":
    main()

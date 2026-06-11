"""LangChain RAG 07 — conversational / memory RAG with query rewriting.

In a chat, follow-ups are elliptical: "and its moons?" only makes sense given
the prior turn. The standard fix is **history-aware retrieval**: before
retrieving, rewrite the follow-up into a *standalone* question using the chat
history, then run normal RAG. LangChain provides
``create_history_aware_retriever`` for this; we also keep a running message
history (memory).

Docs: https://python.langchain.com/docs/tutorials/qa_chat_history/

Offline we implement the rewrite deterministically (resolve pronouns from the
last user/AI turn) so the lesson runs without an LLM.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _rag_common import (  # noqa: E402
    LocalEmbeddings,
    MiniVectorStore,
    MockLLM,
    banner,
    format_context,
    note,
    sample_documents,
    seed_everything,
)

TURNS = [
    "Tell me about Mars.",
    "How many moons does it have?",   # 'it' -> Mars
    "And what about its color?",       # 'its' -> Mars
]


def rewrite_standalone(history, follow_up):
    """Deterministic pronoun resolution using the last named entity in history."""
    subject = None
    for role, text in reversed(history):
        for ent in ("Mars", "Jupiter", "Earth", "Sun", "Moon"):
            if ent.lower() in text.lower():
                subject = ent
                break
        if subject:
            break
    if subject is None:
        return follow_up
    out = follow_up
    for pron in (" it ", " its ", " it?", " its?"):
        out = out.replace(pron, f" {subject}{'?' if pron.endswith('?') else ' '}")
    if out.lower().startswith("and "):
        out = out[4:]
    return out.strip()


def main():
    seed_everything()
    banner("LangChain 07 — conversational memory RAG")

    try:
        from langchain_core.messages import AIMessage, HumanMessage  # noqa: F401

        note("langchain_core messages available (live memory types)")
    except Exception as e:  # pragma: no cover
        note(f"langchain_core messages unavailable ({e}); plain memory")

    store = MiniVectorStore(LocalEmbeddings())
    store.add(sample_documents())
    llm = MockLLM()

    history = []  # list of (role, text)
    for follow_up in TURNS:
        standalone = rewrite_standalone(history, follow_up)
        hits = [d for d, _ in store.similarity(standalone, k=2)]
        ctx = format_context(hits, with_sources=False)
        answer = llm.answer(standalone, ctx)

        print(f"\nUser     : {follow_up}")
        if standalone != follow_up:
            print(f"Rewritten: {standalone}")
        print(f"Assistant: {answer}")

        history.append(("user", follow_up))
        history.append(("assistant", answer))

    print(f"\nConversation memory now holds {len(history)} messages.")
    print("\nOK")


if __name__ == "__main__":
    main()

"""LangChain RAG 04 — the LCEL RAG chain + prompt templates + streaming.

This is the heart of LangChain RAG. We compose a single ``Runnable`` with the
pipe operator:

    {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt | llm | StrOutputParser()

``RunnablePassthrough`` forwards the question untouched into the prompt's
``{question}`` slot, while the same question flows into the retriever to fetch
``{context}``. Because the whole thing is a ``Runnable`` it supports
``.invoke`` / ``.batch`` / ``.stream`` with no extra code — we demonstrate
streaming at the end.

Docs: https://python.langchain.com/docs/tutorials/rag/
      https://python.langchain.com/docs/concepts/lcel/
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from _lc_adapters import get_lc_pieces  # noqa: E402
from _rag_common import (  # noqa: E402
    LocalEmbeddings,
    MiniVectorStore,
    MockLLM,
    banner,
    build_prompt,
    format_context,
    note,
    sample_documents,
    seed_everything,
)

QUESTIONS = [
    "What does the Sun fuse in its core?",
    "What is the largest planet in the Solar System?",
]


def build_lcel_chain():
    """Return (chain, 'live') or raise to signal fallback."""
    llm, embeddings, live = get_lc_pieces()
    if not live:
        raise RuntimeError("langchain_core not importable")

    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    docs = [Document(page_content=d["text"], metadata={"source": d["source"]})
            for d in sample_documents()]
    retriever = FAISS.from_documents(docs, embeddings).as_retriever(
        search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY the context.\n\nContext:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    )

    def format_docs(docs):
        return "\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def fallback_chain():
    """A plain-Python RAG callable when langchain is unavailable."""
    raw = sample_documents()
    store = MiniVectorStore(LocalEmbeddings())
    store.add(raw)
    llm = MockLLM()

    def chain(question):
        hits = [d for d, _ in store.similarity(question, k=3)]
        ctx = format_context(hits, with_sources=False)
        return llm.answer(question, ctx)

    return chain


def main():
    seed_everything()
    banner("LangChain 04 — LCEL RAG chain + streaming")

    try:
        chain, _ = build_lcel_chain()
        live = True
        note("built LIVE LCEL chain: retriever | prompt | llm | parser")
    except Exception as e:  # pragma: no cover
        note(f"LCEL unavailable ({e}); plain-python fallback chain")
        chain = fallback_chain()
        live = False

    # .invoke for each question
    for q in QUESTIONS:
        ans = chain.invoke(q) if live else chain(q)
        print(f"\nQ: {q}\nA: {ans}")

    # .batch (live only; mock loops)
    print("\n[batch] running both questions at once:")
    if live:
        for q, a in zip(QUESTIONS, chain.batch(QUESTIONS)):
            print(f"   {q} -> {a[:60]}")
    else:
        for q in QUESTIONS:
            print(f"   {q} -> {chain(q)[:60]}")

    # .stream — token streaming
    print("\n[stream] streaming the first answer token-by-token:")
    if live:
        for tok in chain.stream(QUESTIONS[0]):
            print(tok, end="", flush=True)
        print()
    else:
        for tok in MockLLM().stream(build_prompt(QUESTIONS[0], sample_documents()[:1])):
            print(tok, end="", flush=True)
        print()

    print("\nOK")


if __name__ == "__main__":
    main()

"""LangChain adapters around the offline harness.

These wrap the deterministic ``MockLLM`` and ``LocalEmbeddings`` from
``_rag_common`` in LangChain's ``LLM`` and ``Embeddings`` interfaces so the
tutorials can build *real* LangChain objects (FAISS stores, LCEL chains,
retrievers) while never touching the network.

If ``langchain_core`` cannot be imported, ``get_lc_pieces`` returns
``(None, None, False)`` and callers degrade to the plain harness objects.
"""

from __future__ import annotations

from typing import List

from _rag_common import LocalEmbeddings, MockLLM


def get_lc_pieces():
    """Return (llm, embeddings, live) where live=True iff langchain_core loaded."""
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_core.language_models.llms import LLM
    except Exception:
        return MockLLM(), LocalEmbeddings(), False

    _mock = MockLLM()
    _emb = LocalEmbeddings()

    class HarnessLLM(LLM):
        """Wrap MockLLM as a LangChain text LLM."""

        @property
        def _llm_type(self) -> str:
            return "harness-mock"

        def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
            return _mock.generate(prompt)

    class HarnessEmbeddings(Embeddings):
        """Wrap LocalEmbeddings as a LangChain Embeddings."""

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return _emb.embed_batch(texts)

        def embed_query(self, text: str) -> List[float]:
            return _emb.embed(text)

    return HarnessLLM(), HarnessEmbeddings(), True

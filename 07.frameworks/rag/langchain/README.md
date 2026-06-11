# LangChain — RAG, end to end

LangChain is the most widely used Python framework for composing LLM
applications. For **retrieval-augmented generation (RAG)** it gives you a
batteries-included toolbox: document loaders, text splitters, an `Embeddings`
interface, ~80 vector-store integrations, a rich family of retrievers
(similarity, MMR, BM25, hybrid/ensemble, multi-query, contextual compression /
re-ranking, parent-document, self-query), prompt templates, and the **LCEL**
(LangChain Expression Language) `Runnable` graph that wires it all into a single
streaming, batchable, async-able pipeline.

> Official docs: <https://python.langchain.com/docs/tutorials/rag/> ·
> RAG concepts: <https://python.langchain.com/docs/concepts/rag/> ·
> Retrievers: <https://python.langchain.com/docs/concepts/retrievers/> ·
> LCEL: <https://python.langchain.com/docs/concepts/lcel/>

## When to use it

- You want a **standard interface** over many model/store providers so you can
  swap FAISS↔Chroma or OpenAI↔Ollama by changing one line.
- You want composable, streaming pipelines (LCEL) with built-in `.invoke`,
  `.batch`, `.stream`, `.ainvoke`.
- You need the **breadth** of retriever strategies (hybrid, multi-query,
  compression/rerank, self-query) without writing them yourself.
- For graph-shaped, stateful agentic RAG, LangChain pairs with **LangGraph**
  (see `07.frameworks/agents/langgraph`).

Reach for **LlamaIndex** instead if your center of gravity is *indexing/ingestion*
and query engines over structured docs; reach for **Haystack** if you want a
typed, declarative *pipeline graph*; reach for **DSPy** if you want to
*optimize* prompts/weights rather than hand-wire them.

## Architecture in words

The mental model is a **graph of `Runnable`s**. Every LangChain primitive
(prompt, model, retriever, parser, even a plain function via `RunnableLambda`)
implements the `Runnable` interface — `invoke / batch / stream / ainvoke`. You
compose them with the pipe operator `|`, exactly like a Unix pipeline:

```
question ─► retriever ─► format_docs ─► prompt ─► llm ─► output_parser ─► answer
                ▲                                   ▲
          vector store                         chat template
                ▲
          embeddings  ◄── splitter ◄── loader ◄── raw documents
```

The classic RAG chain is built with `RunnableParallel` so the user's question
flows to *two* places at once — into the retriever (to fetch context) and
straight through (to fill the `{question}` slot):

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
chain.invoke("What does the Sun fuse?")
```

Because everything is a `Runnable`, the *same* chain streams tokens
(`chain.stream(...)`), runs a list of questions in parallel (`chain.batch(...)`),
and goes async (`await chain.ainvoke(...)`) with no code change.

Key abstractions:

| Abstraction | Interface | Examples |
|---|---|---|
| **Document loader** | `BaseLoader.load()` | `TextLoader`, `WebBaseLoader`, `PyPDFLoader` |
| **Text splitter** | `TextSplitter.split_documents()` | `RecursiveCharacterTextSplitter`, `TokenTextSplitter` |
| **Embeddings** | `Embeddings.embed_documents/embed_query` | `OpenAIEmbeddings`, `HuggingFaceEmbeddings` |
| **Vector store** | `VectorStore.add_documents/similarity_search` | `FAISS`, `Chroma`, `Qdrant` |
| **Retriever** | `BaseRetriever.invoke(query) -> List[Document]` | similarity, MMR, BM25, ensemble, multi-query |
| **Prompt** | `PromptTemplate` / `ChatPromptTemplate` | format strings + message roles |
| **Model** | `BaseChatModel` / `LLM` | `ChatOpenAI`, `ChatOllama` |
| **Output parser** | `BaseOutputParser` | `StrOutputParser`, `PydanticOutputParser` |

## Install

```bash
pip install -r requirements.txt
# core pieces (pure-python, no GPU):
#   langchain-core langchain-community langchain-text-splitters
#   faiss-cpu chromadb rank-bm25 scikit-learn
```

For a real local model you can add an Ollama backend (`pip install langchain-ollama`,
then `ollama pull llama3.2`) or HuggingFace embeddings
(`pip install langchain-huggingface sentence-transformers`). **None of this is
required** — every file here falls back to the deterministic `MockLLM` and a
local embedding from [`_rag_common.py`](_rag_common.py).

## The example files

Each file runs standalone (`python NN.name.py`, exit 0) with no API key, and
prints whether it ran a **live** LangChain object or the **mock** fallback.

| File | Feature |
|---|---|
| [`01.loaders_and_splitting.py`](01.loaders_and_splitting.py) | Document loaders + chunking strategies (recursive, token, semantic) |
| [`02.embeddings_and_vectorstore.py`](02.embeddings_and_vectorstore.py) | `Embeddings` interface + FAISS/Chroma vector stores |
| [`03.retrievers.py`](03.retrievers.py) | similarity, **MMR**, **BM25**, **hybrid/ensemble** retrievers |
| [`04.rag_chain_lcel.py`](04.rag_chain_lcel.py) | the LCEL RAG chain, prompt templates, **streaming** |
| [`05.citations_sources.py`](05.citations_sources.py) | returning **sources/citations** alongside the answer |
| [`06.multi_query_and_compression.py`](06.multi_query_and_compression.py) | **multi-query** expansion + contextual-compression **reranking** |
| [`07.conversational_memory.py`](07.conversational_memory.py) | **history-aware** conversational RAG (query rewriting + memory) |
| [`08.agentic_rag.py`](08.agentic_rag.py) | **agentic RAG**: retrieval as a tool the agent decides to call |
| [`09.evaluation.py`](09.evaluation.py) | **evaluation hooks**: faithfulness / context-recall / answer-match |
| [`app.py`](app.py) | end-to-end CLI RAG app tying every feature together |

## Full feature tour (snippets)

**Loaders + splitting** — chunk so each piece fits the model and is semantically
coherent. `RecursiveCharacterTextSplitter` tries paragraph→line→word boundaries:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```

**Vector store + retriever** — index, then expose as a retriever:

```python
from langchain_community.vectorstores import FAISS
store = FAISS.from_documents(chunks, embeddings)
retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 4})
```

**Hybrid retrieval** — fuse dense + sparse with `EnsembleRetriever` (reciprocal
rank fusion):

```python
from langchain.retrievers import EnsembleRetriever
hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
```

**Multi-query** — let the LLM rewrite one question into several to improve recall:

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
mq = MultiQueryRetriever.from_llm(retriever=dense, llm=llm)
```

**Conversational RAG** — rewrite a follow-up ("and its moons?") into a
standalone query using chat history, then retrieve.

**Agentic RAG** — wrap the retriever as a `Tool` and let an agent decide whether
and how to search, enabling multi-hop questions.

## Gotchas

- **Package split.** Since v0.1, LangChain is split into `langchain-core`
  (interfaces + LCEL), `langchain-community` (integrations), and provider
  packages (`langchain-openai`, `langchain-ollama`, …). Import from the most
  specific package. The `langchain` meta-package is optional glue.
- **Chunking dominates quality.** Too-large chunks bury the answer; too-small
  chunks lose context. Tune `chunk_size`/`chunk_overlap` for your data.
- **MMR vs similarity.** Pure similarity can return near-duplicate chunks; MMR
  (`search_type="mmr"`) trades a little relevance for diversity/coverage.
- **`format_docs` matters.** The model only sees the *string* you build from the
  retrieved `Document`s — always include the text and (if you want citations)
  the metadata.
- **Versions move fast.** APIs that moved out of `langchain.*` into
  `langchain_community.*` are common; pin versions (see `requirements.txt`).

## References

- RAG tutorial — <https://python.langchain.com/docs/tutorials/rag/>
- Retrievers concept — <https://python.langchain.com/docs/concepts/retrievers/>
- Text splitters — <https://python.langchain.com/docs/concepts/text_splitters/>
- Vector stores — <https://python.langchain.com/docs/concepts/vectorstores/>
- LCEL — <https://python.langchain.com/docs/concepts/lcel/>
- Conversational RAG — <https://python.langchain.com/docs/tutorials/qa_chat_history/>
- Agentic RAG — <https://python.langchain.com/docs/tutorials/qa_chat_history/#agents>

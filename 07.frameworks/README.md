# 07.frameworks — the LLM / agent engineering stack

Full-featured, **locally-runnable** tutorials for the modern AI-engineering
toolchain. Each framework lives in its own folder with a real explainer
`README.md`, numbered runnable example files (one per major feature), and an
end-to-end `app.py`. Everything defaults to **local/open models** (Ollama,
llama.cpp, Hugging Face, sentence-transformers, local vector stores); any
API-key or GPU step is marked optional and falls back to a mock so the tutorial
still runs.

See [`../MANIFEST.md`](../MANIFEST.md) for the full framework list + status.

| Area | What it covers | Representative frameworks |
|---|---|---|
| [`rag/`](rag/) | retrieval-augmented generation | LangChain, LlamaIndex, Haystack, DSPy, txtai |
| [`agents/`](agents/) | agent/multi-agent orchestration | LangGraph, CrewAI, AutoGen, Semantic Kernel, smolagents, Letta |
| [`mcp/`](mcp/) | Model Context Protocol | python-sdk, FastMCP, server/client examples |
| [`serving-inference/`](serving-inference/) | run/serve LLMs | Ollama, llama.cpp, vLLM, TGI, SGLang, LiteLLM |
| [`fine-tuning/`](fine-tuning/) | adapt models | PEFT/LoRA, QLoRA, TRL, Unsloth, Axolotl, torchtune |
| [`evaluation/`](evaluation/) | measure quality | lm-eval-harness, RAGAS, DeepEval, promptfoo, TruLens |
| [`vector-databases/`](vector-databases/) | similarity search | FAISS, Chroma, Qdrant, Milvus, Weaviate, pgvector, LanceDB |
| [`orchestration/`](orchestration/) | pipelines/graphs | LangGraph, Flowise, Langflow, Burr |
| [`observability/`](observability/) | trace/monitor | Langfuse, Phoenix, Helicone, OpenLLMetry |
| [`prompt-engineering/`](prompt-engineering/) | structured prompting | DSPy, guidance, outlines, instructor, LMQL |
| [`data-processing/`](data-processing/) | ingest/chunk | unstructured, docling, markitdown, chunking |
| [`embeddings/`](embeddings/) | text→vectors | sentence-transformers, BGE, Nomic, ColBERT |
| [`guardrails/`](guardrails/) | safety/validation | Guardrails AI, NeMo Guardrails, Llama Guard, Presidio |

## Conventions (per `.claude/skills/tutorial-architect`)
- `<framework>/README.md` — what it is, when to use, **architecture in words**,
  install, and a **full** feature tour (not a quickstart) with comparisons.
- `<framework>/NN.<feature>.py` — one runnable, fully-featured file per feature.
- `<framework>/app.py` — an end-to-end app exercising the whole framework.
- `<framework>/requirements.txt` — pinned deps for that tutorial.
- A shared local-LLM helper pattern (Ollama/HF, with a deterministic mock
  fallback) keeps examples runnable offline.

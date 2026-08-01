# MANIFEST — Modern AI Engineering Stack (sections 05.llm → 12)

Companion to [`MAP.md`](MAP.md) (which covers the from-scratch algorithm
curriculum, sections 01–06). This file is the source of truth for the **LLM
architecture catalogue** and **sections 07–12**: the modern LLM/agent
engineering stack, Claude-Code-style agents, Hugging Face, GPU/CUDA, and the
AI-engineer skill profile.

**Standard:** authored per [`.claude/skills/tutorial-architect/SKILL.md`](.claude/skills/tutorial-architect/SKILL.md)
— concept → architecture/math → **full-featured, locally-runnable code** →
pitfalls; exhaustive coverage; real citations; clean-room (no leaked source).

**Status:** `[ ]` planned · `[~]` in progress · `[x]` done.
**Runnable note:** examples target **local/open models** (Ollama, llama.cpp,
Hugging Face, sentence-transformers, local vector stores). Steps needing an API
key or GPU are marked optional and degrade to a mock/fallback so tutorials still
run.

---

## 05.transformers / `llm-architectures/` — exhaustive LLM catalogue

A `00.catalogue.md` table (name · year · org · params · context · key
innovations · license) plus per-family explainer `.md` files and **runnable
PyTorch** for each novel mechanism.

### Architecture innovations (runnable components)
- [ ] `rope.py` — Rotary Position Embeddings (+ NTK / linear scaling, YaRN idea)
- [ ] `alibi.py` — Attention with Linear Biases
- [ ] `gqa_mqa.py` — Grouped-Query / Multi-Query Attention
- [ ] `mla.py` — Multi-head Latent Attention (DeepSeek-V2/V3)
- [ ] `swiglu.py` — SwiGLU / GeGLU gated FFN
- [ ] `rmsnorm.py` — RMSNorm (+ pre/post-norm, DeepNorm)
- [ ] `moe.py` — Mixture-of-Experts routing (top-k, switch, aux-loss, shared experts)
- [ ] `kv_cache.py` — KV cache + paged attention idea
- [ ] `flash_attention.py` — tiling / online-softmax explainer (+ small ref impl)
- [ ] `sliding_window.py` — sliding-window / sparse attention (Mistral, Longformer)
- [ ] `speculative_decoding.py` — draft-and-verify decoding
- [ ] `ssm_mamba.py` — state-space models / Mamba selective scan
- [ ] `rwkv.py` — RWKV linear-attention RNN-transformer hybrid
- [ ] `mtp.py` — multi-token prediction (DeepSeek-V3)
- [ ] `quantization.py` — GPTQ/AWQ/bitsandbytes int8/int4, GGUF (concept + demo)

### Model families (explainer `.md`, classic → SOTA → variants)
- [ ] Encoder: BERT, RoBERTa, ALBERT, DistilBERT, ELECTRA, DeBERTa
- [ ] Encoder-decoder: T5/Flan-T5, BART, mT5, UL2
- [ ] GPT line: GPT-1/2/3, GPT-3.5, GPT-4/4o, o1/o3 (reasoning)
- [ ] LLaMA line: LLaMA-1/2/3/3.1, Code Llama
- [ ] Mistral line: Mistral-7B, Mixtral 8x7B/8x22B (MoE), Mistral-Nemo
- [ ] Qwen line: Qwen1.5/2/2.5, Qwen-MoE
- [ ] DeepSeek line: V2/V3 (MLA + MoE + MTP), R1 (RL reasoning)
- [ ] Google: Gemma 1/2, Gemini (public arch notes), PaLM, Chinchilla (scaling)
- [ ] Others: Falcon, MPT, GPT-NeoX, Pythia, BLOOM, OPT, Phi-1/2/3, Yi,
      Command-R(+), DBRX, Grok (public notes), StableLM, OLMo (open)
- [ ] Multimodal: CLIP, LLaVA, Flamingo, Qwen-VL, GPT-4V (notes), Whisper
- [ ] Non-transformer: Mamba/Mamba-2, RWKV, Jamba (hybrid), RetNet
- [ ] Training/alignment: scaling laws, RLHF, DPO, instruction tuning, MoE training

---

## 07.frameworks/ — the LLM/agent engineering stack (exhaustive)

> **Build status (in progress).** A parallel authoring wave was interrupted by an
> account usage limit (resets 4:30pm UTC). Current coverage:
> - **Substantial:** `rag/`, `agents/`, `mcp/`, `evaluation/`, `observability/`,
>   `fine-tuning/`, `embeddings/`, `vector-databases/`.
> - **Remaining (to author on resume):** `serving-inference/`, `data-processing/`,
>   `orchestration/`, `prompt-engineering/`, `guardrails/`.
> A consolidated `python <file>.py` validation sweep across all committed
> framework examples is also pending (to confirm the offline-runnable contract).

Each framework: its own folder with a full-feature `README.md` + numbered
runnable example files + an end-to-end `app.py`. Areas and frameworks:

### `rag/`
- [ ] langchain · llamaindex · haystack · dspy · txtai · embedchain · canopy · llmware
### `agents/`
- [ ] langgraph · crewai · autogen · semantic-kernel · openai-agents-sdk ·
      smolagents · letta-memgpt · agno-phidata · llama-stack · pydantic-ai
### `mcp/`
- [ ] python-sdk · fastmcp · server-examples · client-examples · typescript-sdk (doc)
### `serving-inference/`
- [ ] ollama · llama-cpp · vllm · tgi · sglang · tensorrt-llm · lmdeploy · litellm
### `fine-tuning/`
- [ ] peft-lora · qlora · trl · unsloth · axolotl · torchtune · llama-factory
### `evaluation/`
- [ ] lm-eval-harness · ragas · deepeval · promptfoo · trulens · openai-evals · giskard
### `vector-databases/`
- [ ] faiss · chroma · qdrant · milvus · weaviate · pgvector · lancedb · pinecone(doc)
### `orchestration/`
- [ ] langgraph-flows · flowise · langflow · haystack-pipelines · burr · controlflow
### `observability/`
- [ ] langfuse · langsmith · phoenix-arize · helicone · openllmetry · opik
### `prompt-engineering/`
- [ ] dspy-optimizers · guidance · outlines · instructor · lmql · sglang-frontend
### `data-processing/`
- [ ] unstructured · docling · llama-parse · markitdown · chunking-strategies
### `embeddings/`
- [ ] sentence-transformers · flagembedding-bge · instructor-embeddings · nomic · colbert
### `guardrails/`
- [ ] guardrails-ai · nemo-guardrails · llama-guard · presidio-pii · prompt-injection-defense

---

## 08.claudecode/ — clean-room Claude-Code-style coding agent

> Built from **public documentation + public reverse-engineering write-ups**
> (cited). **No leaked/proprietary source is used or redistributed.** The code
> here is original, educational, and clearly labeled as a re-implementation.

- [ ] `docs/01.overview.md` — what Claude Code is; CLI/agent product surface
- [ ] `docs/02.architecture.md` — agent loop, tool system, context mgmt, streaming
- [ ] `docs/03.tools.md` — Read/Write/Edit/Bash/Grep/Glob/Agent tool design
- [ ] `docs/04.permissions.md` — permission modes, sandboxing, hooks
- [ ] `docs/05.mcp.md` — MCP integration; `06.skills-subagents.md`; `07.references.md` (citations)
- [ ] `python/` — runnable mini-agent: REPL loop, Anthropic/Ollama backends,
      tool registry (fs/bash/search), permission gate, streaming, MCP client,
      todo/skills — with local fallback so it runs without an API key
- [ ] `ruby/` — the same mini-agent reimplemented in idiomatic Ruby

---

## 09.huggingface/ — the Hugging Face ecosystem

- [ ] `00.manifest.md` — what the Hub has (models/datasets/spaces) + how to use
- [ ] `01.architecture.md` — Hub + libraries map; how pieces fit; cache/workflow
- [ ] `transformers/`, `datasets/`, `tokenizers/`, `accelerate/`, `peft/`, `trl/`,
      `diffusers/`, `hub/`, `gradio-spaces/`, `optimum/` — runnable examples each
- [ ] `02.workflow.md` — end-to-end: load → fine-tune → evaluate → push → serve

---

## 10.gpu/ — CUDA / GPU engineering

- [ ] `01.gpu-architecture.md` — SMs, warps, memory hierarchy, occupancy
- [ ] `02.cuda-programming.md` — kernels, threads/blocks/grids, memory model
- [ ] `03.numba-cuda/` — runnable GPU kernels in Python (Numba) + CPU fallback
- [ ] `04.triton/` — Triton kernels (vector add, softmax, matmul, fused attention)
- [ ] `05.pytorch-performance.md` — profiling, AMP, fusion, CUDA graphs, compile
- [ ] `06.optimization-patterns.md` — coalescing, tiling, shared memory, reductions
- [ ] `07.multi-gpu.md` — data/tensor/pipeline parallelism, NCCL, FSDP/DeepSpeed
- [ ] `08.inference-optimization.md` — KV cache, paged attention, quantization, batching

---

## 11.agent-ai-engineer/ — researched, ranked skill profile

- [ ] `01.research-method.md` — sources, sample of job descriptions, method
- [ ] `02.job-descriptions.md` — condensed corpus (12–100 roles, cited, not verbatim)
- [ ] `03.ranked-skills.md` — the ranked, numbered skills (most→least demanded)
- [ ] `04.skill-frequency.md` — frequency/importance table + how ranking was derived
- [ ] `05.ai-engineer-agent.md` — a drafted "AI engineer" agent persona using the skills
- [ ] `agent-ai-engineer.skill.md` — the numbered skill list (drives section 12)

---

## 12.agent-ai-skills/ — one rich explainer per ranked skill

- [ ] `NN.<skill>.md` for each numbered skill from section 11 — deep explainer:
      what it is, why it matters, how to learn it, tools, a worked example,
      interview signals, and links into this repo. (Seeded now; enriched later.)

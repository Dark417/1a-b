# serving-inference — running and serving LLMs

How do you actually *run* a large language model and put it behind an API? This
area covers the **inference / serving** layer of the stack: the engines and
client libraries that turn model weights into tokens, fast and at scale.

There are three jobs in this layer, and most tools specialise in one or two:

1. **Local single-user runtimes** — easiest way to run a model on your laptop.
   `ollama` (a friendly daemon + model registry over llama.cpp) and
   `llama-cpp` (the C/C++ engine itself, GGUF quantised weights, CPU/Metal/CUDA).
2. **High-throughput GPU servers** — production engines that squeeze a GPU with
   *paged attention* and *continuous batching*: `vllm`, `tgi`
   (Text Generation Inference), `sglang`. These serve many concurrent users.
3. **Provider abstraction** — `litellm` gives you **one** OpenAI-shaped API in
   front of 100+ backends (OpenAI, Anthropic, Ollama, vLLM, Bedrock, …) plus
   routing, retries, fallbacks, cost tracking.

## The mental model: tokens-per-second is a *systems* problem

A decoder LLM generates one token at a time; each step reads the whole **KV
cache** (the attention keys/values of every prior token). Naive serving wastes
GPU memory (pre-allocating max-length cache per request) and GPU time (the GPU
idles while one slow request finishes). The modern engines fix exactly this:

- **Paged attention** (vLLM) — store the KV cache in fixed-size *blocks* like OS
  virtual-memory pages, so memory isn't fragmented and prompts can share blocks.
  ([vLLM paper / docs](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html))
- **Continuous (in-flight) batching** — instead of waiting for a whole batch to
  finish, slot new requests into the batch as soon as any sequence completes.
  TGI, vLLM, SGLang all do this.
  ([HF TGI](https://huggingface.co/docs/text-generation-inference))
- **RadixAttention / prefix caching** (SGLang) — reuse the KV cache of shared
  prompt prefixes across requests via a radix tree.
  ([SGLang docs](https://docs.sglang.ai/))
- **GGUF + quantisation** (llama.cpp/Ollama) — pack weights to 2–8 bits in a
  single mmap-able file so a 7B model fits in a few GB of RAM and runs on CPU.
  ([GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md))

## Which one should I use?

| Tool | Layer | Hardware | Best for | Concurrency |
|---|---|---|---|---|
| **Ollama** | local runtime | CPU / Metal / CUDA | laptops, prototypes, dev | low (1–few) |
| **llama.cpp** | engine | CPU / Metal / CUDA | embedded, edge, max control | low–medium |
| **vLLM** | server | GPU (CUDA/ROCm) | production throughput, OpenAI API | very high |
| **TGI** | server | GPU | HF-native production serving | very high |
| **SGLang** | server | GPU | agentic/structured + prefix reuse | very high |
| **LiteLLM** | client/proxy | any (calls others) | provider abstraction, gateway | n/a (proxy) |

Rule of thumb: **prototype on Ollama**, **serve on vLLM/TGI/SGLang**, and put
**LiteLLM** in front so your application code never hard-codes a provider.

## What runs here, and how it stays offline

Per the repo contract, **every `.py` exits 0 on a plain CPU machine**:

- `ollama` — talks to a local Ollama daemon if `http://localhost:11434` answers;
  otherwise prints a clear note and runs a deterministic **mock** of the REST API.
- `litellm` — examples run **live** using LiteLLM's built-in `mock_response`, so
  no API key or network is needed.
- `llama-cpp`, `vllm`, `tgi`, `sglang` — write **correct, idiomatic** code, then
  guard execution behind `torch.cuda.is_available()` / a connection probe and
  fall back to a CPU/mock smoke path with a printed note.

## Sub-folders

| Folder | Engine | Live here? |
|---|---|---|
| [`ollama/`](ollama/) | Ollama REST + python client, Modelfiles | live if daemon present, else mock |
| [`llama-cpp/`](llama-cpp/) | llama-cpp-python, GGUF, grammars | live if `llama_cpp` + GGUF, else guarded |
| [`vllm/`](vllm/) | OfflineLLM + OpenAI server, paged attn | live on GPU, else mock smoke |
| [`tgi/`](tgi/) | Text Generation Inference client + Docker | live if server reachable, else mock |
| [`sglang/`](sglang/) | SGLang frontend DSL + server | live on GPU server, else mock |
| [`litellm/`](litellm/) | unified API, routing, fallbacks, proxy | **live (mock provider)** |

## References
- vLLM docs — https://docs.vllm.ai/
- llama.cpp — https://github.com/ggml-org/llama.cpp
- llama-cpp-python — https://llama-cpp-python.readthedocs.io/
- Ollama API — https://github.com/ollama/ollama/blob/main/docs/api.md
- TGI — https://huggingface.co/docs/text-generation-inference
- SGLang — https://docs.sglang.ai/
- LiteLLM — https://docs.litellm.ai/

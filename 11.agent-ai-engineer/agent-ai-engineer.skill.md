# AI-Engineer Skill List (canonical, machine-readable)

Ranked, numbered skill profile for the **AI Engineer** archetype, derived in
[`03.ranked-skills.md`](03.ranked-skills.md). Format is intentionally simple and
stable — `NN. <skill name> — one-line scope` — because **this file seeds
[section 12](../12.agent-ai-skills/)**: one explainer file (`NN.<slug>.md`) per
line. Keep names crisp and do not renumber without updating section 12.

Tiers: 01–10 Core · 11–22 Important · 23–32 Differentiating.

```
01. Python & Software Engineering — idiomatic, tested, debuggable production Python and SWE craft
02. ML & Deep Learning Foundations — core math, training dynamics, and the neural-net toolbox
03. LLM & Transformer Architecture — attention, tokenization, context, decoding, scaling laws, model landscape
04. LLM APIs & Prompt Engineering — provider APIs, tool calling, structured output, and measured prompt design
05. Retrieval-Augmented Generation (RAG) — chunking, embeddings, retrieval, reranking, grounded context assembly
06. Agent Design & Orchestration — agent loops, tool use, memory, multi-agent topologies, MCP
07. Evaluation & LLM Testing — eval suites, LLM-as-judge, RAG/agent metrics, regression gating
08. PyTorch & DL Frameworks — building, training, and fine-tuning models in PyTorch (and JAX/TF)
09. Vector Databases & Embeddings — embedding models, ANN indexes, hybrid search, vector stores
10. Cloud Platforms & Deployment — AWS/GCP/Azure, containers, IaC, CI/CD, scaling, cost control
11. MLOps & LLMOps — versioning, serving, monitoring, rollback, and the model lifecycle
12. Fine-Tuning & Model Adaptation — full/PEFT (LoRA/QLoRA), SFT, instruction tuning, data curation
13. AI System Design — architecting end-to-end LLM systems for latency, cost, quality, and reliability
14. Data Engineering for AI — ingestion, cleaning, chunking, labeling, and pipelines at scale
15. Communication & Stakeholder Translation — explaining AI clearly and turning needs into plans
16. Inference Optimization & Serving — KV-cache, batching, quantization, vLLM/TGI/SGLang/TensorRT-LLM
17. Product Sense & Shipping — scoping, prioritizing, iterating, and actually delivering AI products
18. NLP Foundations — tokenization, embeddings, classic NLP tasks and evaluation metrics
19. Safety, Alignment & Guardrails — red-teaming, injection defense, PII, content filtering, RLHF/DPO concepts
20. Second Programming Language — Go, C++, TypeScript, Scala, or Rust beyond Python
21. Containers & Kubernetes — Docker, K8s, GPU scheduling, and workload orchestration
22. LLM Observability & Monitoring — tracing, token/cost dashboards, drift, and quality SLOs
23. GPU Programming & CUDA — kernels (CUDA/Triton), memory hierarchy, profiling, occupancy
24. Distributed Training — data/tensor/pipeline parallelism, FSDP, DeepSpeed, Megatron, NCCL
25. Multimodal AI — vision-language, speech, and image/video generation models
26. Hugging Face Ecosystem — Transformers, Datasets, PEFT, TRL, Accelerate, Hub, Spaces
27. RLHF & Preference Optimization — reward models, PPO, DPO, preference data, RL for reasoning
28. Research Literacy & Publications — reading, reproducing, and extending papers; tracking SOTA
29. Orchestration & Workflow Frameworks — multi-step LLM pipelines (LangGraph, DSPy, Haystack, Burr)
30. Domain & Industry Expertise — vertical depth and cross-domain context-switching
31. Classic ML & Data Science — tabular ML, statistics, experiment design, A/B testing
32. MCP & Agent Protocols — Model Context Protocol and agent interop (A2A, ACP)
```

---

## Suggested section-12 filenames

Stable slugs for [`../12.agent-ai-skills/`](../12.agent-ai-skills/) (one rich
explainer per skill):

```
01.python-software-engineering.md
02.ml-deep-learning-foundations.md
03.llm-transformer-architecture.md
04.llm-apis-prompt-engineering.md
05.retrieval-augmented-generation.md
06.agent-design-orchestration.md
07.evaluation-llm-testing.md
08.pytorch-dl-frameworks.md
09.vector-databases-embeddings.md
10.cloud-platforms-deployment.md
11.mlops-llmops.md
12.fine-tuning-model-adaptation.md
13.ai-system-design.md
14.data-engineering-for-ai.md
15.communication-stakeholder-translation.md
16.inference-optimization-serving.md
17.product-sense-shipping.md
18.nlp-foundations.md
19.safety-alignment-guardrails.md
20.second-programming-language.md
21.containers-kubernetes.md
22.llm-observability-monitoring.md
23.gpu-programming-cuda.md
24.distributed-training.md
25.multimodal-ai.md
26.hugging-face-ecosystem.md
27.rlhf-preference-optimization.md
28.research-literacy-publications.md
29.orchestration-workflow-frameworks.md
30.domain-industry-expertise.md
31.classic-ml-data-science.md
32.mcp-agent-protocols.md
```

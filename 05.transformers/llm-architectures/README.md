# LLM Architectures — Modern Building Blocks

Runnable, from-scratch PyTorch components for the architectural innovations that
turned the 2017 Transformer into today's frontier LLMs. Each file is
self-contained: a teaching docstring (intuition → math → what model introduced
it → references), a clean implementation with the math in comments, and a fast
`demo()` you can run on CPU.

```bash
cd 05.transformers/llm-architectures
python 01.rope.py        # every NN.name.py runs standalone, exits 0
```

Read the `attention/` and `architectures/` sections first for the baseline
(scaled dot-product attention, multi-head attention, the encoder-decoder). This
folder assumes you know vanilla attention and asks: *what changed to get to
LLaMA, Mistral, DeepSeek-V3, Mamba?*

---

## How the modern LLM block fits together (architecture-in-words)

A modern decoder-only LLM is a stack of identical blocks. Trace one token through
a contemporary block (LLaMA / DeepSeek-style), and almost every piece below is one
of these files:

```
                     ┌─────────────────────────── residual highway ──────────────┐
  token embeddings   │                                                            │
        │            │   ┌── RMSNorm (06) ──┐                                      │
        ▼            ▼   ▼                  │                                      ▼
   x ──────────────► + ◄── Attention ───────┘                              ──────► + ──► ...
                          │   - RoPE (01) rotates q,k   (or ALiBi 02)
                          │   - GQA / MQA (03) or MLA (04) shrink the KV cache
                          │   - causal mask; sliding-window (09) for long ctx
                          │   - FlashAttention (10) computes it IO-efficiently
                          │   - KV cache (08) makes decoding O(L) not O(L²)
                          ▼
                      ┌── RMSNorm (06) ──┐
                      ▼                  │
   ... ────────────► + ◄── FFN ──────────┘ ──► next block
                          │   - SwiGLU / GeGLU (05) gated FFN, OR
                          │   - MoE (07): route each token to top-k experts
```

After `N` such blocks: a final RMSNorm, then the output head. Training may add
**multi-token prediction** (14) heads; **quantization** (15) compresses the
trained weights for deployment; **speculative decoding** (11) speeds up
generation. **State-space models** (12, Mamba) and **RWKV** (13) replace the
attention sub-block entirely with an O(L) recurrence — the main non-transformer
alternatives.

The through-line of almost every innovation here is one of two pressures:

1. **Inference cost** — the KV cache and the O(L²) attention dominate serving.
   → GQA/MQA, MLA, sliding-window, FlashAttention, paged KV cache, quantization,
   speculative decoding, and SSM/RWKV all attack this.
2. **Quality per FLOP / parameter** — get more capability for the same compute.
   → RoPE (better positions), RMSNorm + pre-norm (stable depth), SwiGLU (better
   FFN), MoE (huge params, sparse compute), MTP (richer training signal).

---

## Component index

| # | File | Component | One-line idea |
|---|------|-----------|---------------|
| 01 | `01.rope.py` | Rotary Position Embeddings | Rotate q,k by position; dot-product depends only on relative offset. +NTK/PI/YaRN length extension. |
| 02 | `02.alibi.py` | Attention with Linear Biases | No position vectors; add a per-head linear distance penalty. Extrapolates to longer sequences. |
| 03 | `03.gqa_mqa.py` | MHA / GQA / MQA | Let query heads share KV heads (`n_kv_heads`); shrinks the KV cache. |
| 04 | `04.mla.py` | Multi-head Latent Attention | Cache a small low-rank latent instead of full K,V; decoupled RoPE keeps position. |
| 05 | `05.swiglu.py` | SwiGLU / GeGLU FFN | Gated FFN: `act(W_gate x) ⊙ (W_up x)`. The 2/3 hidden-dim trick keeps params matched. |
| 06 | `06.rmsnorm.py` | RMSNorm | Normalize by RMS only (no mean, no bias). +pre/post-norm, DeepNorm scaling. |
| 07 | `07.moe.py` | Mixture-of-Experts | Top-k routing, load-balance aux loss, capacity, shared expert. Big params, sparse compute. |
| 08 | `08.kv_cache.py` | KV cache | Reuse past K,V so decoding is O(L). +paged-attention concept. |
| 09 | `09.sliding_window.py` | Sliding-window attention | Each query sees only the last W keys → O(L·W). +Longformer global tokens. |
| 10 | `10.flash_attention.py` | FlashAttention | Tiling + online softmax: exact attention without the O(L²) score matrix in HBM. |
| 11 | `11.speculative_decoding.py` | Speculative decoding | Cheap draft proposes K tokens; target verifies in one pass. Exact, faster. |
| 12 | `12.ssm_mamba.py` | SSM / Mamba | Selective state-space recurrence; O(L) time, O(1) state, input-dependent gating. |
| 13 | `13.rwkv.py` | RWKV | Softmax-free WKV linear attention; trains like a transformer, runs like an RNN. |
| 14 | `14.mtp.py` | Multi-token prediction | Predict t+1, t+2, ... per position (DeepSeek-V3). Richer signal + self-speculation. |
| 15 | `15.quantization.py` | Quantization | int8 (sym/asym) + int4 group quant; dequant error + size. GPTQ/AWQ/GGUF/bnb concepts. |

---

## Innovation → models that introduced / use it

| Innovation | Introduced by (paper, year) | Used by |
|------------|------------------------------|---------|
| RoPE | RoFormer, Su et al. **2021** | LLaMA 1/2/3, Mistral, Qwen, GPT-NeoX, PaLM, DeepSeek, Gemma |
| RoPE scaling (PI / NTK / YaRN) | Chen 2023; bloc97 2023; YaRN, Peng et al. **2023** | Long-context LLaMA, Code Llama, Qwen, Yi-200K |
| ALiBi | Press et al. **2021** | BLOOM, MPT, BloombergGPT, Falcon (early), Replit |
| MQA | Shazeer **2019** | PaLM, Falcon |
| GQA | Ainslie et al. **2023** | LLaMA-2 70B, LLaMA-3, Mistral, Qwen2, Gemma |
| MLA | DeepSeek-V2 **2024** | DeepSeek-V2 / V3 / R1 |
| SwiGLU / GeGLU | Shazeer **2020** | LLaMA, Mistral, Qwen, DeepSeek (SwiGLU); Gemma, T5-v1.1 (GeGLU) |
| RMSNorm | Zhang & Sennrich **2019** | LLaMA, Mistral, Qwen, DeepSeek, Gemma, T5 |
| Pre-norm | Xiong et al. **2020** | GPT-2 onward, essentially all decoder LLMs |
| DeepNorm (deep post-norm) | DeepNet, Wang et al. **2022** | GLM-130B, deep encoder-decoders |
| Sparse MoE | Shazeer et al. **2017**; Switch, Fedus et al. **2021** | Switch-T, GLaM, Mixtral 8x7B/8x22B, DeepSeek-V3, Grok, DBRX, Qwen-MoE |
| Shared-expert MoE | DeepSeekMoE, Dai et al. **2024** | DeepSeek-V2 / V3 |
| KV cache | standard since GPT-2 era | every autoregressive transformer |
| Paged attention | vLLM, Kwon et al. **2023** | vLLM, TGI, most serving stacks |
| Sliding-window attn | Longformer / BigBird **2020**; Mistral **2023** | Longformer, BigBird, Mistral-7B, Gemma-2 (alt. layers) |
| FlashAttention | Dao et al. **2022** (FA-2 2023, FA-3 2024) | PyTorch SDPA, vLLM, all efficient training |
| Speculative decoding | Leviathan et al. **2023**; Chen et al. **2023** | production serving; Medusa / EAGLE variants |
| Mamba / selective SSM | S4 **2021**; Mamba, Gu & Dao **2023** | Mamba, Mamba-2, Jamba (hybrid), Codestral-Mamba |
| RWKV | Peng et al. **2023** (v5/v6 2024) | RWKV-4/5/6 ("Eagle"/"Finch") |
| Multi-token prediction | Gloeckle et al. **2024**; DeepSeek-V3 **2024** | DeepSeek-V3 |
| Quantization (int8/int4) | LLM.int8 **2022**; GPTQ, AWQ, QLoRA **2023** | bitsandbytes, llama.cpp/GGUF, AutoGPTQ, AutoAWQ |

---

## References (primary sources)

- Vaswani et al. (2017), *Attention Is All You Need*.
- Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
- Chen et al. (2023), *Extending Context Window of LLMs via Position Interpolation*; bloc97 (2023), *NTK-Aware Scaled RoPE*; Peng et al. (2023), *YaRN*.
- Press, Smith & Lewis (2021), *Train Short, Test Long: Attention with Linear Biases*.
- Shazeer (2019), *Fast Transformer Decoding: One Write-Head is All You Need* (MQA); Ainslie et al. (2023), *GQA*.
- DeepSeek-AI (2024), *DeepSeek-V2* and *DeepSeek-V3 Technical Report* (MLA, MTP, shared-expert MoE).
- Shazeer (2020), *GLU Variants Improve Transformer*.
- Zhang & Sennrich (2019), *Root Mean Square Layer Normalization*; Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture*; Wang et al. (2022), *DeepNet*.
- Shazeer et al. (2017), *Outrageously Large Neural Networks*; Fedus, Zoph & Shazeer (2021), *Switch Transformers*; Lepikhin et al. (2020), *GShard*; Dai et al. (2024), *DeepSeekMoE*.
- Kwon et al. (2023), *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM).
- Beltagy, Peters & Cohan (2020), *Longformer*; Zaheer et al. (2020), *Big Bird*; Jiang et al. (2023), *Mistral 7B*.
- Dao et al. (2022), *FlashAttention*; Dao (2023), *FlashAttention-2*; Milakov & Gimelshein (2018), *Online normalizer calculation for softmax*.
- Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding*; Chen et al. (2023), *Accelerating LLM Decoding with Speculative Sampling*.
- Gu, Goel & Ré (2021), *S4*; Gu & Dao (2023), *Mamba*; Dao & Gu (2024), *Transformers are SSMs* (Mamba-2).
- Peng et al. (2023), *RWKV: Reinventing RNNs for the Transformer Era*; Peng et al. (2024), *Eagle and Finch*.
- Gloeckle et al. (2024), *Better & Faster LLMs via Multi-token Prediction*.
- Dettmers et al. (2022), *LLM.int8()*; Dettmers et al. (2023), *QLoRA* (NF4); Frantar et al. (2023), *GPTQ*; Lin et al. (2023), *AWQ*.

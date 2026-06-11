# Hugging Face Transformers — University-Grade Tutorial

> Official docs: https://huggingface.co/docs/transformers/index  
> Model Hub: https://huggingface.co/models  
> Source: https://github.com/huggingface/transformers

---

## 1. What Is the Transformers Library?

`transformers` is Hugging Face's unified PyTorch/TensorFlow/JAX library for pretrained language models.
It provides:

- **Pretrained weights** for thousands of models (BERT, GPT-2, T5, LLaMA, Falcon, Mistral …)
- **Auto* classes** that resolve the correct architecture from a `config.json`
- **Pipelines** for one-line inference on 30+ NLP/vision/audio tasks
- **Trainer API** for fine-tuning with minimal boilerplate
- **Generation utilities** (beam search, sampling, constrained decoding)
- **Quantization hooks** (bitsandbytes, GPTQ, AWQ, GGUF)

---

## 2. Transformer Architecture in Words

The original Transformer (Vaswani et al. 2017, https://arxiv.org/abs/1706.03762) stacks:

```
Input tokens
  └─► Token embeddings + Positional embeddings
        └─► N × TransformerLayer
              ├─ Multi-Head Self-Attention   (Q, K, V projections)
              │     score = softmax(QKᵀ / √dₖ) · V
              ├─ Add & LayerNorm
              ├─ Feed-Forward Network  (two linear layers + activation)
              └─ Add & LayerNorm
        └─► Task head (LM head, classifier, …)
```

**Three families:**

| Family | Arch | Representative | Use-case |
|--------|------|----------------|----------|
| Encoder-only | BERT-style | bert-base, roberta | Classification, NER, QA |
| Decoder-only | GPT-style | GPT-2, LLaMA | Text generation |
| Encoder-Decoder | T5/BART | t5-small, bart-large | Translation, Summarization |

---

## 3. Auto* Class Resolution

Every model checkpoint ships a `config.json`. The `model_type` field is the key:

```json
{
  "model_type": "bert",
  "hidden_size": 768,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  ...
}
```

`AutoConfig.from_pretrained("bert-base-uncased")` reads this file and returns a `BertConfig`.  
`AutoTokenizer.from_pretrained(...)` returns `BertTokenizerFast`.  
`AutoModel.from_pretrained(...)` returns `BertModel`.

**Task heads follow the same pattern:**

```python
from transformers import (
    AutoModelForSequenceClassification,  # adds a linear classifier on [CLS]
    AutoModelForCausalLM,                # adds an LM head for next-token prediction
    AutoModelForMaskedLM,                # adds an LM head, predicts masked tokens
    AutoModelForTokenClassification,     # per-token linear classifier
    AutoModelForQuestionAnswering,       # start/end span logits
    AutoModelForSeq2SeqLM,              # encoder-decoder with cross-attention
)
```

Docs: https://huggingface.co/docs/transformers/model_doc/auto

---

## 4. The config.json Contract

`config.json` encodes every hyperparameter:

```json
{
  "architectures": ["BertForMaskedLM"],
  "model_type": "bert",
  "hidden_size": 768,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "vocab_size": 30522,
  "id2label": {"0": "NEGATIVE", "1": "POSITIVE"},
  "label2id": {"NEGATIVE": 0, "POSITIVE": 1}
}
```

You can override any field:

```python
config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_hidden_layers = 2       # smaller model
config.hidden_dropout_prob = 0.1
model = AutoModel.from_config(config)   # random weights
```

---

## 5. Three Approaches: Pipelines vs Manual vs Trainer

### 5.1 Pipelines — one-liner inference

```python
from transformers import pipeline

clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
result = clf("I love this!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

Pros: trivial to use, handles tokenization + postprocessing.  
Cons: less flexible, harder to customise.

Docs: https://huggingface.co/docs/transformers/main_classes/pipelines

### 5.2 Manual (tokenise → forward → decode)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tok("Hello, world!", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
# out.logits: (batch, seq_len, vocab_size)
```

Pros: full control over every tensor.  
Cons: more code.

### 5.3 Trainer — fine-tuning

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(output_dir="./ckpt", num_train_epochs=3, per_device_train_batch_size=16)
trainer = Trainer(model=model, args=args, train_dataset=ds, compute_metrics=compute_metrics)
trainer.train()
```

Pros: handles gradient accumulation, mixed precision, evaluation loops, checkpointing.  
Cons: opinionated; override callbacks for custom behaviour.

Docs: https://huggingface.co/docs/transformers/main_classes/trainer

---

## 6. Installation

```bash
pip install transformers torch datasets evaluate accelerate
# For GPU quantization:
pip install bitsandbytes
# For GPTQ:
pip install auto-gptq
# For AWQ:
pip install autoawq
```

---

## 7. Full Feature Tour

### Tokenization

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = tok("Hello world!", padding="max_length", truncation=True, max_length=16, return_tensors="pt")
# enc.input_ids, enc.attention_mask, enc.token_type_ids
decoded = tok.decode(enc.input_ids[0], skip_special_tokens=True)
```

### Batched inference

```python
pipe = pipeline("text-classification", model="...", batch_size=32, device=0)
results = pipe(["text1", "text2", "text3"])
```

### Streaming generation

```python
from transformers import TextStreamer
streamer = TextStreamer(tok)
model.generate(**inputs, streamer=streamer, max_new_tokens=200)
```

### Saving and loading

```python
model.save_pretrained("./my-model")
tok.save_pretrained("./my-model")
model2 = AutoModel.from_pretrained("./my-model")
```

---

## 8. Task Taxonomy Table

| Task | Pipeline string | AutoModel class |
|------|----------------|-----------------|
| Text Classification | `"text-classification"` | `AutoModelForSequenceClassification` |
| Token Classification / NER | `"token-classification"` | `AutoModelForTokenClassification` |
| Question Answering | `"question-answering"` | `AutoModelForQuestionAnswering` |
| Fill-Mask | `"fill-mask"` | `AutoModelForMaskedLM` |
| Text Generation | `"text-generation"` | `AutoModelForCausalLM` |
| Summarization | `"summarization"` | `AutoModelForSeq2SeqLM` |
| Translation | `"translation_xx_to_yy"` | `AutoModelForSeq2SeqLM` |
| Zero-Shot Classification | `"zero-shot-classification"` | entailment model |
| Feature Extraction | `"feature-extraction"` | `AutoModel` |
| Image Classification | `"image-classification"` | `AutoModelForImageClassification` |
| ASR | `"automatic-speech-recognition"` | `AutoModelForSpeechSeq2Seq` |

---

## 9. Generation Strategies Table

| Strategy | Key Args | When to use |
|----------|----------|-------------|
| Greedy | `do_sample=False` | Deterministic, short outputs |
| Beam Search | `num_beams=5` | Balanced quality/diversity |
| Top-k Sampling | `do_sample=True, top_k=50` | Creative, diverse outputs |
| Top-p (Nucleus) | `do_sample=True, top_p=0.92` | Avoids low-prob tokens |
| Temperature | `temperature=0.7` | Scale confidence (lower=sharper) |
| Contrastive Search | `penalty_alpha=0.6, top_k=4` | Coherent long generation |
| Typical Sampling | `typical_p=0.92` | More human-like text |

Docs: https://huggingface.co/docs/transformers/generation_strategies

---

## 10. Quantization

### 10.1 bitsandbytes (int8 / int4)

Reduces memory ~2–4×. Requires GPU.

```python
from transformers import BitsAndBytesConfig
bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
# or
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_use_double_quant=True,     # QLoRA trick
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", quantization_config=bnb_cfg)
```

Docs: https://huggingface.co/docs/transformers/quantization/bitsandbytes

### 10.2 GPTQ

Post-training quantization using Hessian-based weight compression.
Requires `auto-gptq` package. Weights stored as int4; fast GPU inference.

```python
from transformers import GPTQConfig
gptq_cfg = GPTQConfig(bits=4, dataset="c4", tokenizer=tok)
model = AutoModelForCausalLM.from_pretrained("...", quantization_config=gptq_cfg)
```

Docs: https://huggingface.co/docs/transformers/quantization/gptq

### 10.3 AWQ (Activation-aware Weight Quantization)

Preserves salient weights based on activation magnitudes. Often better quality than GPTQ at 4-bit.
Requires `autoawq` package.

```python
from transformers import AwqConfig
awq_cfg = AwqConfig(bits=4, fuse_max_seq_len=512, do_fuse=True)
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-AWQ", quantization_config=awq_cfg)
```

Docs: https://huggingface.co/docs/transformers/quantization/awq

---

## 11. Gotchas & Comparison Notes

| Issue | Solution |
|-------|----------|
| Missing pad token (GPT-2) | `tok.pad_token = tok.eos_token` or set `pad_token_id` |
| Slow generation | Use `torch.compile(model)`, `use_cache=True` (default) |
| OOM on large models | Use quantization, `device_map="auto"`, gradient checkpointing |
| NaN loss | Lower learning rate, check data preprocessing |
| Tokenizer mismatch | Always save tokenizer alongside model |
| `trust_remote_code` | Only set True for known repos |
| Right-side padding for generation | Set `tok.padding_side = "left"` for batch generation |

---

## 12. Tutorial Files

| File | Topic |
|------|-------|
| `01.pipelines.py` | Pipeline API, all major tasks |
| `02.automodel.py` | Auto* classes, manual forward pass |
| `03.generation.py` | Decoding strategies |
| `04.attention_hidden_states.py` | Inspecting internals |
| `05.tasks.py` | End-to-end task examples |
| `06.trainer.py` | Fine-tuning with Trainer |
| `07.quantization.py` | Quantization concepts + demo |

---

*All examples run on CPU with tiny models. Network failures are gracefully handled.*

"""
05.tasks.py — End-to-End Task Examples
=======================================
Concrete worked examples for five major NLP tasks using tiny models.
Each section shows: input → model → decode → human-readable output.

Official docs:
  https://huggingface.co/docs/transformers/tasks/sequence_classification
  https://huggingface.co/docs/transformers/tasks/token_classification
  https://huggingface.co/docs/transformers/tasks/question_answering
  https://huggingface.co/docs/transformers/tasks/summarization
  https://huggingface.co/docs/transformers/tasks/language_modeling

Tiny models:
  prajjwal1/bert-tiny   — encoder (classification, NER, QA)
  sshleifer/tiny-gpt2   — decoder (generation)
  hf-internal-testing/tiny-random-t5 — seq2seq (summarization)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BertConfig, BertForSequenceClassification,
    BertForTokenClassification,
    BertForQuestionAnswering,
    GPT2Config, GPT2LMHeadModel,
    T5Config, T5ForConditionalGeneration,
)

# ─── Task 1: Text Classification ─────────────────────────────────────────────
banner("Task 1: Text Classification (Sentiment Analysis)")
# Model: BERT-based with a linear head on the [CLS] token
# Input: text string → Output: predicted class + confidence

ok_tok, tok1 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, clf = safe(
    AutoModelForSequenceClassification.from_pretrained,
    "prajjwal1/bert-tiny",
    num_labels=2,
    ignore_mismatched_sizes=True,
)

if ok_tok and ok_mdl:
    clf.eval()
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    samples = [
        "This is a wonderful movie!",
        "The product broke after one day.",
        "Average experience, nothing special.",
    ]
    for text in samples:
        inputs = tok1(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = clf(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = probs.argmax().item()
        print(f"  {text!r}")
        print(f"    → {id2label[pred_id]} (conf={probs[pred_id]:.3f})")
else:
    note_skip(f"clf: tok={ok_tok}, mdl={ok_mdl}")
    # Fallback: local model with random weights
    cfg = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, vocab_size=1000, num_labels=2)
    clf_fb = BertForSequenceClassification(cfg)
    clf_fb.eval()
    dummy = {"input_ids": torch.randint(1, 1000, (1, 8)),
             "attention_mask": torch.ones(1, 8, dtype=torch.long)}
    with torch.no_grad():
        out = clf_fb(**dummy)
    probs = F.softmax(out.logits, dim=-1)[0]
    print(f"  [fallback] logits={out.logits}, pred={probs.argmax().item()}")

# ─── Task 2: Named Entity Recognition (Token Classification) ──────────────────
banner("Task 2: Named Entity Recognition (Token Classification)")
# Model: BERT with a linear head PER TOKEN
# Input: text → Output: per-token label (B-PER, I-PER, O, B-ORG, etc.)
# id2label maps integer class index → BIO tag string

ok_tok, tok2 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, ner = safe(
    AutoModelForTokenClassification.from_pretrained,
    "prajjwal1/bert-tiny",
    num_labels=9,                 # typical NER: O B-PER I-PER B-ORG I-ORG B-LOC I-LOC B-MISC I-MISC
    ignore_mismatched_sizes=True,
)

if ok_tok and ok_mdl:
    ner.eval()
    id2label_ner = {
        0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG",
        4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC",
    }
    text = "Albert Einstein was born in Ulm, Germany."
    inputs = tok2(text, return_tensors="pt", truncation=True, max_length=64)
    tokens = tok2.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    with torch.no_grad():
        logits = ner(**inputs).logits          # (1, seq_len, num_labels)

    preds = logits[0].argmax(dim=-1).tolist()
    print(f"  Text: {text!r}")
    print(f"  {'Token':20} {'Pred':10}")
    print(f"  {'-'*30}")
    for token, pred_id in zip(tokens, preds):
        print(f"  {token:20} {id2label_ner[pred_id]}")
else:
    note_skip(f"ner: tok={ok_tok}, mdl={ok_mdl}")
    cfg = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, vocab_size=1000, num_labels=9)
    ner_fb = BertForTokenClassification(cfg)
    ner_fb.eval()
    dummy = {"input_ids": torch.randint(1, 1000, (1, 8)),
             "attention_mask": torch.ones(1, 8, dtype=torch.long)}
    with torch.no_grad():
        out = ner_fb(**dummy)
    preds = out.logits[0].argmax(dim=-1).tolist()
    print(f"  [fallback] per-token preds: {preds}")

# ─── Task 3: Extractive Question Answering ────────────────────────────────────
banner("Task 3: Extractive Question Answering")
# Model: BERT with two heads — one for start position, one for end position
# Input: (question, context) → Output: text span from context

ok_tok, tok3 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, qa_model = safe(
    AutoModelForQuestionAnswering.from_pretrained,
    "prajjwal1/bert-tiny",
    ignore_mismatched_sizes=True,
)

if ok_tok and ok_mdl:
    qa_model.eval()

    qa_pairs = [
        {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Paris is its capital city.",
        },
        {
            "question": "Who wrote the Transformers paper?",
            "context": "The paper 'Attention Is All You Need' was written by Vaswani et al. in 2017.",
        },
    ]

    for pair in qa_pairs:
        inputs = tok3(
            pair["question"], pair["context"],
            return_tensors="pt", truncation=True, max_length=128,
        )
        with torch.no_grad():
            out = qa_model(**inputs)

        # start/end positions are relative to the flat input_ids tensor
        start = out.start_logits.argmax().item()
        end = out.end_logits.argmax().item()

        # Ensure start <= end
        if end < start:
            end = start

        answer_ids = inputs["input_ids"][0, start: end + 1]
        answer = tok3.decode(answer_ids, skip_special_tokens=True)

        print(f"  Q: {pair['question']}")
        print(f"  A: {answer!r}  (span [{start}:{end+1}])")
        print()
else:
    note_skip(f"qa: tok={ok_tok}, mdl={ok_mdl}")
    cfg = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, vocab_size=1000)
    qa_fb = BertForQuestionAnswering(cfg)
    qa_fb.eval()
    dummy = {"input_ids": torch.randint(1, 1000, (1, 12)),
             "attention_mask": torch.ones(1, 12, dtype=torch.long)}
    with torch.no_grad():
        out = qa_fb(**dummy)
    start = out.start_logits.argmax().item()
    end = out.end_logits.argmax().item()
    print(f"  [fallback] start={start}, end={end}")

# ─── Task 4: Summarization ────────────────────────────────────────────────────
banner("Task 4: Summarization (Seq2Seq)")
# Model: encoder-decoder (T5)
# Input: long text → Output: compressed summary
# T5 uses a "task prefix": "summarize: ..."

ok_tok, tok4 = safe(AutoTokenizer.from_pretrained, "hf-internal-testing/tiny-random-t5")
ok_mdl, t5 = safe(AutoModelForSeq2SeqLM.from_pretrained, "hf-internal-testing/tiny-random-t5")

if ok_tok and ok_mdl:
    t5.eval()
    article = (
        "summarize: The transformer architecture was introduced in the paper "
        "'Attention Is All You Need' by Vaswani et al. in 2017. "
        "It replaced recurrent networks with self-attention mechanisms, "
        "enabling parallelization and better long-range dependencies. "
        "The architecture has since become foundational for NLP, vision, and beyond."
    )
    inputs = tok4(article, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        summary_ids = t5.generate(
            **inputs,
            max_new_tokens=30,
            min_length=5,
            num_beams=2,
            early_stopping=True,
        )
    summary = tok4.decode(summary_ids[0], skip_special_tokens=True)
    print(f"  Input length  : {inputs['input_ids'].shape[1]} tokens")
    print(f"  Output length : {summary_ids.shape[1]} tokens")
    print(f"  Summary: {summary!r}")
else:
    note_skip(f"t5: tok={ok_tok}, mdl={ok_mdl}")
    cfg = T5Config(d_model=64, d_ff=128, num_heads=2, num_layers=2,
                   d_kv=32, vocab_size=1000)
    t5_fb = T5ForConditionalGeneration(cfg)
    t5_fb.eval()
    enc_ids = torch.randint(1, 1000, (1, 20))
    with torch.no_grad():
        out = t5_fb.generate(input_ids=enc_ids, max_new_tokens=10,
                             decoder_start_token_id=0)
    print(f"  [fallback] generated ids shape: {out.shape}")

# ─── Task 5: Text Generation ─────────────────────────────────────────────────
banner("Task 5: Text Generation (Causal LM)")
# Model: GPT-2 decoder-only
# Input: prompt → Output: continuation
# The model autoregressively predicts the next token given all previous tokens.

ok_tok, tok5 = safe(AutoTokenizer.from_pretrained, "sshleifer/tiny-gpt2")
ok_mdl, gpt = safe(AutoModelForCausalLM.from_pretrained, "sshleifer/tiny-gpt2")

if ok_tok and ok_mdl:
    tok5.pad_token = tok5.eos_token
    gpt.eval()

    prompts = [
        "The history of machine learning",
        "In the beginning, there was",
        "Scientists discovered that",
    ]
    for prompt in prompts:
        inputs = tok5(prompt, return_tensors="pt")
        torch.manual_seed(42)
        with torch.no_grad():
            gen_ids = gpt.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.9,
                top_k=40,
                pad_token_id=tok5.eos_token_id,
            )
        full_text = tok5.decode(gen_ids[0], skip_special_tokens=True)
        continuation = full_text[len(prompt):]
        print(f"  Prompt     : {prompt!r}")
        print(f"  Continuation: {continuation!r}")
        print()
else:
    note_skip(f"gpt: tok={ok_tok}, mdl={ok_mdl}")
    cfg = GPT2Config(n_embd=64, n_head=2, n_layer=2, vocab_size=1000,
                     n_positions=64, n_ctx=64)
    gpt_fb = GPT2LMHeadModel(cfg)
    gpt_fb.eval()
    dummy = torch.randint(1, 1000, (1, 5))
    torch.manual_seed(42)
    with torch.no_grad():
        out = gpt_fb.generate(dummy, max_new_tokens=10, do_sample=True,
                              pad_token_id=0)
    print(f"  [fallback] generated shape: {out.shape}, ids: {out[0].tolist()}")

print("\n[DONE] 05.tasks.py complete")

"""
05.tasks.py — End-to-End Task Examples
=======================================
This module walks through 6 NLP tasks end-to-end:
  1. Text Classification (sentiment)
  2. Named Entity Recognition
  3. Extractive Question Answering
  4. Summarization (seq2seq)
  5. Causal Language Model generation
  6. Feature extraction / semantic similarity

Each example shows: raw text → tokenize → forward → decode/interpret output.

Official docs:
  - https://huggingface.co/docs/transformers/task_summary
  - https://huggingface.co/docs/transformers/tasks/sequence_classification

Tiny models used:
  prajjwal1/bert-tiny, sshleifer/tiny-gpt2,
  hf-internal-testing/tiny-random-t5
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)
import numpy as np
np.random.seed(0)

from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BertConfig, BertModel, BertForSequenceClassification,
)

# ─── Task 1: Text Classification (Sentiment) ─────────────────────────────────
banner("Task 1: Text Classification (Sentiment Analysis)")
# Input: raw text string
# Output: label + confidence score
# Architecture: BERT encoder → [CLS] → linear head → softmax

ok_tok, tok = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, clf = safe(
    AutoModelForSequenceClassification.from_pretrained,
    "prajjwal1/bert-tiny",
    num_labels=2,
    ignore_mismatched_sizes=True,
)

if ok_tok and ok_mdl:
    clf.eval()
    texts = [
        "I absolutely love this product, it exceeded my expectations!",
        "Terrible experience. The service was awful and rude.",
        "The package arrived on time.",
    ]
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            logits = clf(**enc).logits            # (1, 2)
        probs = torch.softmax(logits, dim=-1)[0]  # (2,)
        pred_id = probs.argmax().item()
        print(f"  {probs[pred_id]:.3f} {id2label[pred_id]:8}  |  {text[:55]}")
else:
    note_skip(f"tok={ok_tok} clf={ok_mdl}")
    # Fallback: show the decode steps with random logits
    cfg = BertConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=128, vocab_size=1000, num_labels=2)
    fb_clf = BertForSequenceClassification(cfg)
    fb_clf.eval()
    dummy = {"input_ids": torch.randint(0, 1000, (1, 8)),
             "attention_mask": torch.ones(1, 8, dtype=torch.long)}
    with torch.no_grad():
        logits = fb_clf(**dummy).logits
    probs = torch.softmax(logits, dim=-1)
    print(f"  [fallback] logits={logits}  probs={probs}")
    print("  Shape: (batch=1, num_labels=2); argmax → predicted class")

# ─── Task 2: Named Entity Recognition (token classification) ─────────────────
banner("Task 2: Named Entity Recognition (Token Classification)")
# Input: sentence
# Output: per-token label (B-ORG, I-PER, O, …)
# Architecture: BERT → per-token linear head (no softmax pooling)

ok_tok, tok2 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, ner_model = safe(
    AutoModelForTokenClassification.from_pretrained,
    "prajjwal1/bert-tiny",
    num_labels=9,                     # IOB2 tags: O, B-PER, I-PER, B-ORG, …
    ignore_mismatched_sizes=True,
)

if ok_tok and ok_mdl:
    ner_model.eval()
    ner_text = "Hugging Face was founded in New York by researchers."
    enc = tok2(ner_text, return_tensors="pt")
    tokens = tok2.convert_ids_to_tokens(enc["input_ids"][0])
    with torch.no_grad():
        logits = ner_model(**enc).logits    # (1, seq_len, num_labels)
    preds = logits[0].argmax(dim=-1)        # (seq_len,)
    # IOB2 label set (illustrative)
    ner_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                  "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    print(f"  {'Token':20} {'Label':10}")
    print(f"  {'-'*30}")
    for tok_str, pred_id in zip(tokens, preds.tolist()):
        label = ner_labels[pred_id] if pred_id < len(ner_labels) else f"LABEL_{pred_id}"
        if tok_str not in ("[CLS]", "[SEP]", "<s>", "</s>"):
            print(f"  {tok_str:20} {label}")
else:
    note_skip(f"NER failed: tok={ok_tok}")
    print("  API shape: logits (batch, seq_len, num_labels)")
    print("  Each token gets a label → decode with id2label mapping")

# ─── Task 3: Extractive Question Answering ────────────────────────────────────
banner("Task 3: Extractive Question Answering")
# Input: question + context passage
# Output: start_idx, end_idx into context → extracted span
# Architecture: BERT → two linear heads (start_logits, end_logits)

ok_tok, tok3 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, qa_model = safe(
    AutoModelForQuestionAnswering.from_pretrained,
    "prajjwal1/bert-tiny",
)

qa_pairs = [
    {
        "question": "When was the Eiffel Tower built?",
        "context": "The Eiffel Tower was built from 1887 to 1889 in Paris, France.",
    },
    {
        "question": "Who wrote the Transformer paper?",
        "context": "The paper 'Attention is All You Need' was written by Vaswani et al. in 2017.",
    },
]

if ok_tok and ok_mdl:
    qa_model.eval()
    for pair in qa_pairs:
        enc = tok3(
            pair["question"], pair["context"],
            return_tensors="pt",
            truncation="only_second",
            max_length=128,
        )
        with torch.no_grad():
            out = qa_model(**enc)
        start = out.start_logits.argmax().item()
        end = out.end_logits.argmax().item() + 1
        # Decode the span; handle degenerate cases
        if end > start:
            answer_ids = enc["input_ids"][0][start:end]
            answer = tok3.decode(answer_ids, skip_special_tokens=True)
        else:
            answer = "(no valid span)"
        print(f"  Q: {pair['question']}")
        print(f"  A: {answer!r}  (span [{start}:{end}])")
        print()
else:
    note_skip(f"QA failed: tok={ok_tok}")
    print("  API shape:")
    print("    out.start_logits: (1, seq_len)")
    print("    out.end_logits  : (1, seq_len)")
    print("    answer = decode(input_ids[start:end+1])")

# ─── Task 4: Summarization (Seq2Seq) ─────────────────────────────────────────
banner("Task 4: Summarization (Seq2Seq)")
# Input: long document
# Output: shorter summary
# Architecture: T5/BART encoder processes doc → decoder generates summary
# model.generate() handles the autoregressive loop

ok_tok, tok4 = safe(AutoTokenizer.from_pretrained, "hf-internal-testing/tiny-random-t5")
ok_mdl, t5 = safe(AutoModelForSeq2SeqLM.from_pretrained, "hf-internal-testing/tiny-random-t5")

if ok_tok and ok_mdl:
    t5.eval()
    article = (
        "Researchers at MIT have developed a new neural network architecture "
        "that significantly reduces training time. The model uses a novel "
        "attention mechanism that scales linearly with sequence length. "
        "Experiments show 3x speedup on standard NLP benchmarks."
    )
    # T5 expects task prefix: "summarize: <text>"
    enc = tok4("summarize: " + article, return_tensors="pt",
               truncation=True, max_length=128)
    with torch.no_grad():
        gen_ids = t5.generate(
            **enc,
            max_new_tokens=30,
            num_beams=2,
            early_stopping=True,
        )
    summary = tok4.decode(gen_ids[0], skip_special_tokens=True)
    print(f"  Input  ({len(article)} chars): {article[:80]}…")
    print(f"  Summary: {summary!r}")
    print(f"  Token compression: {enc['input_ids'].shape[1]} → {gen_ids.shape[1]} tokens")
else:
    note_skip(f"T5 failed: tok={ok_tok}")
    print("  API shape:")
    print("    gen_ids = model.generate(**enc, max_new_tokens=30)")
    print("    summary = tokenizer.decode(gen_ids[0], skip_special_tokens=True)")

# ─── Task 5: Causal Language Model (Text Generation) ────────────────────────
banner("Task 5: Causal Language Model (Text Generation)")
# Input: prompt text
# Output: prompt + model continuation
# Architecture: GPT-style decoder-only

ok_tok, tok5 = safe(AutoTokenizer.from_pretrained, "sshleifer/tiny-gpt2")
ok_mdl, gpt = safe(AutoModelForCausalLM.from_pretrained, "sshleifer/tiny-gpt2")

if ok_tok and ok_mdl:
    tok5.pad_token = tok5.eos_token
    gpt.eval()
    prompts = [
        "The key to understanding deep learning is",
        "In the year 2050, robots will",
    ]
    for prompt in prompts:
        enc = tok5(prompt, return_tensors="pt")
        n_prompt = enc["input_ids"].shape[1]
        torch.manual_seed(42)
        with torch.no_grad():
            out = gpt.generate(
                **enc,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tok5.eos_token_id,
            )
        # Decode only the new tokens
        new_ids = out[0, n_prompt:]
        continuation = tok5.decode(new_ids, skip_special_tokens=True)
        full_text = tok5.decode(out[0], skip_special_tokens=True)
        print(f"  Prompt:  {prompt!r}")
        print(f"  New:     {continuation!r}")
        print(f"  Full:    {full_text!r}")
        print()
else:
    note_skip(f"GPT-2 failed: tok={ok_tok}")
    print("  API shape:")
    print("    out = model.generate(**enc, max_new_tokens=20, do_sample=True)")
    print("    text = tokenizer.decode(out[0], skip_special_tokens=True)")

# ─── Task 6: Feature Extraction & Semantic Similarity ────────────────────────
banner("Task 6: Feature Extraction & Semantic Similarity")
# Input: two sentences
# Output: cosine similarity of their embeddings
# Architecture: BERT encoder → mean-pool hidden states

ok_tok, tok6 = safe(AutoTokenizer.from_pretrained, "prajjwal1/bert-tiny")
ok_mdl, enc_model = safe(AutoModel.from_pretrained, "prajjwal1/bert-tiny")

def mean_pool(model_output, attention_mask):
    """Masked mean-pool over token dimension."""
    hs = model_output.last_hidden_state          # (B, L, H)
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    return (hs * mask).sum(1) / mask.sum(1)      # (B, H)

sentence_pairs = [
    ("A dog is playing in the park.", "A puppy is running outside."),
    ("The stock market rose sharply today.", "Scientists discovered a new planet."),
]

if ok_tok and ok_mdl:
    enc_model.eval()
    print(f"  {'Sentence A':45} | {'Sentence B':45} | Cosine")
    print(f"  {'-'*100}")
    for sent_a, sent_b in sentence_pairs:
        def encode(text):
            inputs = tok6(text, return_tensors="pt",
                          padding=True, truncation=True, max_length=64)
            with torch.no_grad():
                out = enc_model(**inputs)
            emb = mean_pool(out, inputs["attention_mask"])
            return torch.nn.functional.normalize(emb, dim=-1)  # unit vector

        emb_a = encode(sent_a)
        emb_b = encode(sent_b)
        cos = (emb_a * emb_b).sum().item()
        print(f"  {sent_a[:44]:45} | {sent_b[:44]:45} | {cos:.4f}")
else:
    note_skip("feature extraction failed")
    # Fallback: demonstrate with random unit vectors
    a = torch.nn.functional.normalize(torch.randn(1, 128), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(1, 128), dim=-1)
    cos = (a * b).sum().item()
    print(f"  [fallback] random unit vector cosine similarity: {cos:.4f}")
    print("  Real usage: encode sentences, compute cosine similarity")

print("\n[DONE] 05.tasks.py complete")

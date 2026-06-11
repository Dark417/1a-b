"""
01.pipelines.py — Hugging Face Pipeline API
============================================
Pipelines are the highest-level abstraction: pass a task string and an optional
model name, get back predictions.  Under the hood they:
  1. Load AutoTokenizer + AutoModel for the task
  2. Preprocess (tokenize) input
  3. Run forward pass
  4. Postprocess logits → human-readable output

Official docs: https://huggingface.co/docs/transformers/main_classes/pipelines

Tiny models used:
  - prajjwal/bert-tiny  (encoder, for BERT-based tasks)
  - sshleifer/tiny-gpt2 (decoder, for generation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _lib import safe, banner, note_skip

import torch
torch.set_num_threads(1)
torch.manual_seed(0)

# ─── 1. Sentiment Analysis (text-classification) ─────────────────────────────
banner("1. Sentiment Analysis / Text Classification")
# task="text-classification" routes to AutoModelForSequenceClassification
# The pipeline returns [{'label': ..., 'score': ...}]

ok, clf = safe(
    __import__("transformers").pipeline,
    "text-classification",
    model="prajjwal1/bert-tiny",
)
if ok:
    result = clf("I absolutely love this framework!")
    print(f"  Sentiment result: {result}")
    # Batch inference — pass a list; pipeline handles padding automatically
    batch = clf(["Great product!", "Terrible experience."], batch_size=2)
    print(f"  Batch results:    {batch}")
else:
    note_skip(str(clf))
    # Fallback: show what the output structure looks like
    print("  API shape: [{'label': 'POSITIVE', 'score': 0.98}]")

# ─── 2. Text Generation ───────────────────────────────────────────────────────
banner("2. Text Generation (CausalLM)")
# task="text-generation" routes to AutoModelForCausalLM
# Important: GPT-2 family has no pad token — set pad_token_id=eos_token_id

ok, gen = safe(
    __import__("transformers").pipeline,
    "text-generation",
    model="sshleifer/tiny-gpt2",
    pad_token_id=50256,   # eos_token_id for GPT-2; avoids the padding warning
)
if ok:
    out = gen("The future of AI is", max_new_tokens=20, do_sample=False)
    print(f"  Generated: {out[0]['generated_text']}")
    # num_return_sequences > 1 requires do_sample=True or num_beams > 1
    outs = gen("Once upon a time", max_new_tokens=15, do_sample=True,
               temperature=0.8, num_return_sequences=2)
    for i, o in enumerate(outs):
        print(f"  Seq {i}: {o['generated_text']}")
else:
    note_skip(str(gen))
    print("  API shape: [{'generated_text': 'The future of AI is bright...'}]")

# ─── 3. Fill-Mask (MLM) ───────────────────────────────────────────────────────
banner("3. Fill-Mask (Masked Language Model)")
# task="fill-mask" routes to AutoModelForMaskedLM
# Input must contain the tokenizer's mask token, typically [MASK] for BERT

ok, filler = safe(
    __import__("transformers").pipeline,
    "fill-mask",
    model="prajjwal1/bert-tiny",
)
if ok:
    # BERT tokenizer uses [MASK]
    preds = filler("Paris is the [MASK] of France.")
    for p in preds[:3]:
        print(f"  {p['token_str']!r:15} score={p['score']:.4f}  → {p['sequence']}")
else:
    note_skip(str(filler))
    print("  API shape: [{'token_str': 'capital', 'score': 0.92, 'sequence': '...'}]")

# ─── 4. Named Entity Recognition (NER / token-classification) ─────────────────
banner("4. Named Entity Recognition")
# task="ner" or "token-classification"
# aggregation_strategy="simple" merges subword tokens into full entity spans

ok, ner = safe(
    __import__("transformers").pipeline,
    "ner",
    model="prajjwal1/bert-tiny",
    aggregation_strategy="simple",
)
if ok:
    entities = ner("Hugging Face is based in New York City.")
    for e in entities:
        print(f"  {e['entity_group']:10} '{e['word']}' score={e['score']:.3f}")
else:
    note_skip(str(ner))
    print("  API shape: [{'entity_group': 'ORG', 'word': 'Hugging Face', 'score': 0.99}]")

# ─── 5. Question Answering (extractive) ───────────────────────────────────────
banner("5. Extractive Question Answering")
# task="question-answering" routes to AutoModelForQuestionAnswering
# Model picks a start and end span from the context

context = (
    "Hugging Face was founded in 2016. "
    "It is known for the Transformers library."
)
ok, qa = safe(
    __import__("transformers").pipeline,
    "question-answering",
    model="prajjwal1/bert-tiny",
)
if ok:
    answer = qa(question="When was Hugging Face founded?", context=context)
    print(f"  Answer: '{answer['answer']}'  score={answer['score']:.4f}")
    print(f"  Span:   start={answer['start']} end={answer['end']}")
else:
    note_skip(str(qa))
    print("  API shape: {'answer': '2016', 'score': 0.99, 'start': 24, 'end': 28}")

# ─── 6. Zero-Shot Classification ──────────────────────────────────────────────
banner("6. Zero-Shot Classification")
# Uses an NLI (natural language inference) model.
# The model asks: does this text entail label X?
# Works without ANY task-specific fine-tuning.

ok, zs = safe(
    __import__("transformers").pipeline,
    "zero-shot-classification",
    model="prajjwal1/bert-tiny",   # tiny NLI; real use: cross-encoder/nli-*
)
if ok:
    res = zs(
        "This tutorial teaches you how to use Transformers.",
        candidate_labels=["education", "sports", "politics"],
    )
    for label, score in zip(res["labels"], res["scores"]):
        print(f"  {label:15} {score:.4f}")
else:
    note_skip(str(zs))
    print("  API shape: {'labels': ['education', ...], 'scores': [0.95, ...]}")

# ─── 7. Summarization ─────────────────────────────────────────────────────────
banner("7. Summarization (Seq2Seq)")
# task="summarization" routes to AutoModelForSeq2SeqLM
# Uses encoder-decoder model: T5/BART etc.

ok, summ = safe(
    __import__("transformers").pipeline,
    "summarization",
    model="hf-internal-testing/tiny-random-t5",
)
if ok:
    article = (
        "Transformers are a type of neural network architecture introduced in "
        "2017. They rely on self-attention mechanisms instead of recurrence. "
        "They have become the dominant architecture in NLP and beyond."
    )
    summary = summ(article, max_new_tokens=30, min_length=5)
    print(f"  Summary: {summary[0]['summary_text']}")
else:
    note_skip(str(summ))
    print("  API shape: [{'summary_text': 'Transformers use self-attention...'}]")

# ─── 8. Pipeline kwargs: returning all scores, batch_size ─────────────────────
banner("8. Pipeline Advanced: return_all_scores + batch_size")
# return_all_scores=True (or top_k=None) returns scores for all labels

ok, clf2 = safe(
    __import__("transformers").pipeline,
    "text-classification",
    model="prajjwal1/bert-tiny",
)
if ok:
    # top_k=None → return all class scores (equivalent to old return_all_scores=True)
    all_scores = clf2("Interesting!", top_k=None)
    print(f"  All scores: {all_scores}")
else:
    note_skip(str(clf2))
    print("  API shape: [[{'label':'LABEL_0','score':0.4},{'label':'LABEL_1','score':0.6}]]")

print("\n[DONE] 01.pipelines.py complete")

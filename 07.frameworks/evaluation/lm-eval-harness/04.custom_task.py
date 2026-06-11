"""04 · Define & run a brand-new custom task end-to-end (offline).

Putting files 01–03 together: this is the workflow you'd actually follow to add
a task to lm-evaluation-harness, reproduced offline so it *runs*. We:

  1. write the task as a **YAML-style config dict** (mirrors the harness's
     ``.yaml`` task files);
  2. compile it into a runnable Task (``doc_to_text`` / target / metrics);
  3. run a mock model and aggregate the metric;
  4. show the **real CLI** you'd use against a real model, gated behind a
     try/except so this file still exits 0 with no installs.

The real harness loads YAML like this (sentiment classification as 2-way MC):

    task: my_sentiment
    dataset_path: csv
    dataset_kwargs: {data_files: data.csv}
    output_type: multiple_choice
    doc_to_text: "Review: {{text}}\\nSentiment:"
    doc_to_choices: ["negative", "positive"]
    doc_to_target: label
    metric_list:
      - metric: acc

New-task guide:
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md

Run:  python 04.custom_task.py     (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import banner, note_skip, safe, tokenize  # noqa: E402


# ---------------------------------------------------------------------------
# 1) The task config — the same fields the real YAML exposes.
# ---------------------------------------------------------------------------
TASK_YAML = {
    "task": "my_sentiment",
    "output_type": "multiple_choice",
    "doc_to_text": "Review: {text}\nSentiment:",
    "doc_to_choices": ["negative", "positive"],
    "doc_to_target": "label",          # name of the gold field (an index)
    "metric_list": ["acc"],
}

# Tiny labelled dataset (label: 0=negative, 1=positive).
DOCS = [
    {"text": "An absolute masterpiece, I loved every minute.", "label": 1},
    {"text": "Boring, predictable, and far too long.", "label": 0},
    {"text": "Best film of the year, brilliant acting.", "label": 1},
    {"text": "A complete waste of time, terrible.", "label": 0},
    {"text": "Charming and heartfelt, a real delight.", "label": 1},
]


# ---------------------------------------------------------------------------
# 2) Compile the config into a runnable task.
# ---------------------------------------------------------------------------
def render_text(doc: dict) -> str:
    return TASK_YAML["doc_to_text"].format(**doc)


# ---------------------------------------------------------------------------
# A sentiment-aware mock LM: loglikelihood favours the choice whose sentiment
# matches lexical cues in the review. Deterministic.
# ---------------------------------------------------------------------------
POS_CUES = {"masterpiece", "loved", "best", "brilliant", "charming",
            "heartfelt", "delight", "great", "good"}
NEG_CUES = {"boring", "predictable", "waste", "terrible", "bad", "worst",
            "dull", "awful"}


class SentimentLM:
    def loglikelihood(self, context: str, continuation: str) -> float:
        toks = set(tokenize(context))
        pos = len(toks & POS_CUES)
        neg = len(toks & NEG_CUES)
        choice = continuation.strip().lower()
        # Higher log-prob (less negative) for the matching sentiment.
        if choice == "positive":
            score = pos - neg
        else:
            score = neg - pos
        # Map an integer "evidence" score to a log-prob via a logistic-ish curve.
        return -math.log(1 + math.exp(-score))


def run_task(lm: SentimentLM) -> float:
    choices = TASK_YAML["doc_to_choices"]
    correct = 0
    for doc in DOCS:
        ctx = render_text(doc)
        lls = [lm.loglikelihood(ctx, " " + c) for c in choices]
        pred = max(range(len(choices)), key=lambda i: lls[i])
        gold = doc[TASK_YAML["doc_to_target"]]
        correct += int(pred == gold)
        mark = "ok " if pred == gold else "MISS"
        print(f"  [{mark}] pred={choices[pred]:8s} gold={choices[gold]:8s} "
              f"| {doc['text'][:42]}")
    return correct / len(DOCS)


# ---------------------------------------------------------------------------
# 4) The real CLI, guarded. If lm-eval is installed AND a model is available
#    you could run this for real; otherwise we just show it and skip.
# ---------------------------------------------------------------------------
def try_real_cli() -> None:
    ok, mod = safe(__import__, "lm_eval")
    if not ok:
        note_skip("`lm-eval` not installed (pip install lm-eval)")
        print("       Real run would be:")
        print("         lm_eval --model hf \\")
        print("                 --model_args pretrained=EleutherAI/pythia-160m \\")
        print("                 --tasks my_sentiment \\")
        print("                 --include_path ./my_tasks \\")
        print("                 --num_fewshot 2 --limit 50")
        return
    print(f"[ok] lm-eval is installed (version {getattr(mod, '__version__', '?')}).")
    print("     To run for real, drop the YAML above into ./my_tasks/my_sentiment.yaml")
    print("     and point --include_path at it. We do NOT auto-download a model here.")


def main() -> int:
    banner(f"Custom task: '{TASK_YAML['task']}'  (compiled from YAML config)")
    print("Config fields:")
    for k, v in TASK_YAML.items():
        print(f"  {k:16s}: {v}")

    banner("Running the task offline with a mock LM")
    acc = run_task(SentimentLM())
    print(f"\n  acc = {acc:.3f}  (over {len(DOCS)} docs)")

    banner("The real CLI (guarded)")
    try_real_cli()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

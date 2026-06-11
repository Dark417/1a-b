"""app.py · A mini lm-eval-harness: register tasks, run a model, aggregate.

End-to-end demo tying files 01–04 together into the shape of the real harness:

    register tasks  ->  for each (task, doc): build requests
                    ->  run them through the LM adapter
                    ->  score per-doc  ->  aggregate  ->  results table

It runs THREE tasks (a multiple-choice task, a generation task, and the custom
sentiment task) against one mock model, then prints a leaderboard-style table
with a bootstrap standard error — exactly what `lm_eval` writes to its results
JSON.

Run:  python app.py     (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import banner, normalize, tokenize  # noqa: E402


# ---------------------------------------------------------------------------
# Model adapter: the two primitives every lm-eval model exposes.
# ---------------------------------------------------------------------------
class MockModel:
    """Knows a few capitals, a few facts, and reads sentiment cues."""

    KNOWN_CAPITALS = {"rome", "madrid", "ottawa", "paris", "tokyo"}
    POS = {"masterpiece", "loved", "brilliant", "great", "delight"}
    NEG = {"boring", "waste", "terrible", "awful", "dull"}
    FACTS = {"hamlet": "William Shakespeare", "speed of light": "299792458"}

    def loglikelihood(self, context: str, continuation: str) -> float:
        toks = tokenize(continuation)
        ctx = context.lower()
        # Capitals task: reward a continuation that is a capital named in context.
        if any("capital" in line for line in ctx.splitlines()):
            return sum(math.log(0.6) if t in self.KNOWN_CAPITALS else math.log(0.1)
                       for t in toks) if toks else -100.0
        # Sentiment task.
        c = continuation.strip().lower()
        ct = set(tokenize(context))
        ev = len(ct & self.POS) - len(ct & self.NEG)
        ev = ev if c == "positive" else -ev
        return -math.log(1 + math.exp(-ev))

    def generate_until(self, context: str, stop: list[str]) -> str:
        out = "unknown"
        for key, ans in self.FACTS.items():
            if key in context.lower():
                out = ans
                break
        for s in stop:
            i = out.find(s)
            if i != -1:
                out = out[:i]
        return out


# ---------------------------------------------------------------------------
# Task registry.
# ---------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    output_type: str
    docs: list[dict]
    doc_to_text: Callable[[dict], str]
    doc_to_target: Callable[[dict], object]
    doc_to_choices: Callable[[dict], list[str]] = field(default=lambda d: [])
    metrics: tuple[str, ...] = ("acc",)


def em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = tokenize(normalize(pred)), tokenize(normalize(gold))
    if not p or not g:
        return float(not p and not g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# The evaluation loop.
# ---------------------------------------------------------------------------
def evaluate(model: MockModel, task: Task) -> dict[str, tuple[float, float]]:
    """Return {metric: (mean, bootstrap_stderr)}."""
    per_doc: dict[str, list[float]] = {m: [] for m in task.metrics}
    for doc in task.docs:
        ctx = task.doc_to_text(doc)
        if task.output_type == "multiple_choice":
            choices = task.doc_to_choices(doc)
            lls = [model.loglikelihood(ctx, " " + c) for c in choices]
            pred = max(range(len(choices)), key=lambda i: lls[i])
            gold = task.doc_to_target(doc)
            per_doc["acc"].append(float(pred == gold))
        else:  # generate_until
            pred = model.generate_until(ctx, stop=["\n"])
            gold = task.doc_to_target(doc)
            if "exact_match" in per_doc:
                per_doc["exact_match"].append(em(pred, gold))
            if "f1" in per_doc:
                per_doc["f1"].append(f1(pred, gold))
    return {m: (mean(v), bootstrap_stderr(v)) for m, v in per_doc.items()}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_stderr(xs: list[float], iters: int = 1000, seed: int = 0) -> float:
    """Standard error of the mean via bootstrap resampling (harness default)."""
    if len(xs) < 2:
        return 0.0
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    mu = sum(means) / iters
    var = sum((m - mu) ** 2 for m in means) / (iters - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Define the three tasks.
# ---------------------------------------------------------------------------
def build_tasks() -> list[Task]:
    capitals = Task(
        name="capitals",
        output_type="multiple_choice",
        docs=[
            {"country": "Italy", "choices": ["Milan", "Rome", "Naples", "Turin"], "answer": 1},
            {"country": "Spain", "choices": ["Madrid", "Barcelona", "Seville"], "answer": 0},
            {"country": "Canada", "choices": ["Toronto", "Ottawa", "Montreal"], "answer": 1},
        ],
        doc_to_text=lambda d: f"Question: What is the capital of {d['country']}?\nAnswer:",
        doc_to_target=lambda d: d["answer"],
        doc_to_choices=lambda d: d["choices"],
        metrics=("acc",),
    )
    facts = Task(
        name="facts_qa",
        output_type="generate_until",
        docs=[
            {"q": "Who wrote Hamlet?", "gold": "William Shakespeare"},
            {"q": "Speed of light in m/s?", "gold": "299792458"},
        ],
        doc_to_text=lambda d: d["q"],
        doc_to_target=lambda d: d["gold"],
        metrics=("exact_match", "f1"),
    )
    sentiment = Task(
        name="sentiment",
        output_type="multiple_choice",
        docs=[
            {"text": "A masterpiece, I loved it.", "label": 1},
            {"text": "Boring and a waste of time.", "label": 0},
            {"text": "Brilliant and a real delight.", "label": 1},
            {"text": "Awful, dull, terrible.", "label": 0},
        ],
        doc_to_text=lambda d: f"Review: {d['text']}\nSentiment:",
        doc_to_target=lambda d: d["label"],
        doc_to_choices=lambda d: ["negative", "positive"],
        metrics=("acc",),
    )
    return [capitals, facts, sentiment]


def main() -> int:
    model = MockModel()
    tasks = build_tasks()

    banner("Mini-harness: running 3 tasks against 1 model")
    results: dict[str, dict] = {}
    for task in tasks:
        results[task.name] = evaluate(model, task)
        print(f"  ran '{task.name}' ({task.output_type}, {len(task.docs)} docs)")

    banner("Results table (mean ± bootstrap stderr)")
    print(f"  {'task':12s} {'metric':14s} {'value':>8s} {'stderr':>8s}")
    print("  " + "-" * 44)
    for tname, metrics in results.items():
        for m, (val, se) in metrics.items():
            print(f"  {tname:12s} {m:14s} {val:8.3f} {se:8.3f}")

    print("\nThis is the structure lm_eval writes to results.json: per-task,")
    print("per-metric values with a bootstrap standard error. Swap MockModel")
    print("for the `hf` adapter and the loop is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

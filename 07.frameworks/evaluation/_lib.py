"""
Shared helpers for the 07.frameworks/evaluation tutorials.
=========================================================

Everything here is deterministic and offline. The centrepiece is a
``MockLLM`` / ``MockJudge`` so that evaluation loops that normally call a
hosted model (GPT-4 as a judge, an embedding API, an NLI model) can run with
``python file.py`` → exit 0 and **no API key / no network**.

The mock is intentionally simple and *rule-based* so its verdicts are
reproducible and explainable in a tutorial. Real evaluators swap it for a
genuine LLM; the surrounding harness code is identical.
"""

from __future__ import annotations
import re
import math
import hashlib
from collections import Counter
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────
def banner(title: str) -> None:
    """Print a bordered section title."""
    line = "=" * (len(title) + 4)
    print(f"\n{line}")
    print(f"| {title} |")
    print(f"{line}")


def note_skip(msg: str) -> None:
    """Standardised skip notice when an optional dep / network is missing."""
    print(f"[skip] optional dependency or network unavailable — using offline fallback")
    print(f"       reason: {msg}")


def safe(fn: Callable, *a, **k):
    """Call fn; return (True, result) or (False, exception)."""
    try:
        return True, fn(*a, **k)
    except Exception as exc:  # noqa: BLE001 — tutorials must never crash
        return False, exc


# ─────────────────────────────────────────────────────────────────────────────
# Tiny tokenizer (shared by BLEU/ROUGE/EM and the mock NLI)
# ─────────────────────────────────────────────────────────────────────────────
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Deterministic, dependency-free."""
    return _WORD.findall(text.lower())


def normalize(text: str) -> str:
    """SQuAD-style normalisation: lowercase, strip articles/punct/extra space."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic MockLLM + MockJudge
# ─────────────────────────────────────────────────────────────────────────────
class MockLLM:
    """A deterministic stand-in for a chat model.

    It supports the handful of behaviours evaluation harnesses ask of a model:
      * ``generate(prompt)``        — produce an answer (looked up / templated)
      * ``judge_score(prompt)``     — return a float in [0, 1] for LLM-as-judge
      * ``classify(prompt, labels)``— pick a label (for G-Eval / verdict style)

    Determinism: outputs are a pure function of the input string (hashing +
    keyword rules), so every run is identical and explainable in a tutorial.
    """

    def __init__(self, knowledge: dict[str, str] | None = None):
        # Optional canned answers keyed by a substring of the question.
        self.knowledge = knowledge or {}

    # -- text generation ------------------------------------------------------
    def generate(self, prompt: str) -> str:
        low = prompt.lower()
        for key, ans in self.knowledge.items():
            if key.lower() in low:
                return ans
        # Fall back to an extractive-ish echo so downstream metrics have signal.
        toks = tokenize(prompt)
        return " ".join(toks[-8:]) if toks else "no answer"

    # -- LLM-as-judge: numeric score in [0,1] --------------------------------
    def judge_score(self, prompt: str) -> float:
        """Rule-based judge. Rewards overlap / positive cues, penalises
        contradiction / 'I don't know' / refusal cues. Deterministic."""
        low = prompt.lower()
        score = 0.5
        positive = ("correct", "accurate", "supported", "relevant",
                    "faithful", "helpful", "yes")
        negative = ("incorrect", "wrong", "unsupported", "irrelevant",
                    "contradict", "hallucinat", "no information", "cannot",
                    "i don't know", "unknown", "refuse")
        for w in positive:
            if w in low:
                score += 0.12
        for w in negative:
            if w in low:
                score -= 0.18
        # Add a tiny deterministic jitter so ties break reproducibly.
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000
        score += (h / 1000 - 0.5) * 0.02
        return max(0.0, min(1.0, score))

    # -- classification / verdict --------------------------------------------
    def classify(self, prompt: str, labels: list[str]) -> str:
        s = self.judge_score(prompt)
        # Map score onto an ordered label list (low→high).
        idx = min(len(labels) - 1, int(s * len(labels)))
        return labels[idx]


class MockEmbedder:
    """Deterministic bag-of-words hashing embedder for cosine similarity.

    Good enough to make answer-relevancy / context-precision style metrics
    *run* offline; not semantically deep, but stable and explainable.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        n = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / n for v in vec]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


# ─────────────────────────────────────────────────────────────────────────────
# A mock NLI (entailment) classifier — used by faithfulness-style metrics
# ─────────────────────────────────────────────────────────────────────────────
def mock_nli(premise: str, hypothesis: str) -> str:
    """Return 'entailment' | 'contradiction' | 'neutral'.

    Heuristic: token overlap of the hypothesis with the premise. If most
    content words of the hypothesis appear in the premise → entailment.
    A handful of negation cues flips toward contradiction. Deterministic.
    """
    p = set(tokenize(premise))
    h = tokenize(hypothesis)
    if not h:
        return "neutral"
    content = [t for t in h if len(t) > 2]
    if not content:
        content = h
    overlap = sum(1 for t in content if t in p) / len(content)
    neg = {"not", "no", "never", "isn", "wasn", "didn", "false"}
    has_neg_mismatch = bool(set(h) & neg) and not (set(tokenize(premise)) & neg)
    if has_neg_mismatch and overlap > 0.4:
        return "contradiction"
    if overlap >= 0.6:
        return "entailment"
    if overlap <= 0.2:
        return "contradiction" if has_neg_mismatch else "neutral"
    return "neutral"

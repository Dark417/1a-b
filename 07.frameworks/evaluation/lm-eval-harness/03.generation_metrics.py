"""03 · Generation tasks: `generate_until` + exact-match / F1 (the GSM8K mechanism).

The other half of the harness. For open-ended tasks the model **generates** text
(via ``generate_until(context, stop)``) and the generation is matched against the
gold target. Three reference metrics, implemented here from scratch so you see
the math:

  * **exact_match** — after SQuAD-style normalisation (lowercase, strip
    articles/punctuation/extra whitespace), is prediction == gold? Binary.
  * **token-F1** — treat prediction and gold as bags of tokens; F1 of the
    overlap. Rewards partial answers. (This is SQuAD's secondary metric.)
  * **regex / flexible extract** — pull the answer out of a chain-of-thought
    with a pattern (GSM8K extracts the last number: ``-?[\\d,]+``).

The mock LM here "answers" by retrieving from a tiny knowledge dict and, for the
math task, emitting a short chain-of-thought ending in the number — exactly the
shape GSM8K parsing expects.

EM/F1 recipe: Rajpurkar et al. 2016 (SQuAD), https://arxiv.org/abs/1606.05250
GSM8K answer extraction: Cobbe et al. 2021, https://arxiv.org/abs/2110.14168

Run:  python 03.generation_metrics.py    (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import banner, normalize, tokenize  # noqa: E402


# ---------------------------------------------------------------------------
# Reference metrics — from scratch.
# ---------------------------------------------------------------------------
def exact_match(pred: str, gold: str) -> float:
    """1.0 iff normalised strings are identical."""
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token-level F1 over normalised tokens."""
    p_toks = tokenize(normalize(pred))
    g_toks = tokenize(normalize(gold))
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


_GSM8K_NUMBER = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_last_number(text: str) -> str:
    """GSM8K-style: the answer is the LAST number in the generation."""
    nums = _GSM8K_NUMBER.findall(text)
    return nums[-1].replace(",", "") if nums else ""


# ---------------------------------------------------------------------------
# A mock generation LM: generate_until(context, stop) -> str
# ---------------------------------------------------------------------------
class MockGenLM:
    def __init__(self, knowledge: dict[str, str]):
        self.knowledge = knowledge

    def generate_until(self, context: str, stop: list[str]) -> str:
        for key, ans in self.knowledge.items():
            if key.lower() in context.lower():
                out = ans
                break
        else:
            out = "unknown"
        # Honour stop sequences (the harness truncates at the first one).
        for s in stop:
            idx = out.find(s)
            if idx != -1:
                out = out[:idx]
        return out


# ---------------------------------------------------------------------------
# Two generation tasks: short-answer QA and GSM8K-style math.
# ---------------------------------------------------------------------------
QA_DOCS = [
    {"q": "Who wrote Hamlet?", "gold": "William Shakespeare"},
    {"q": "What gas do plants absorb?", "gold": "carbon dioxide"},
    {"q": "Capital of Italy?", "gold": "Rome"},
]
MATH_DOCS = [
    {"q": "Natalia sold 48 clips in April and half as many in May. Total?",
     "gold": "72"},
    {"q": "A robe takes 2 bolts blue and half that white. How many bolts?",
     "gold": "3"},
]


def run_qa(lm: MockGenLM) -> dict[str, float]:
    ems, f1s = [], []
    for d in QA_DOCS:
        pred = lm.generate_until(d["q"], stop=["\n"])
        em = exact_match(pred, d["gold"])
        f1 = token_f1(pred, d["gold"])
        ems.append(em)
        f1s.append(f1)
        print(f"  Q: {d['q']}")
        print(f"     pred='{pred}'  gold='{d['gold']}'  EM={em:.0f}  F1={f1:.2f}")
    return {"exact_match": sum(ems) / len(ems), "f1": sum(f1s) / len(f1s)}


def run_math(lm: MockGenLM) -> dict[str, float]:
    correct = 0
    for d in MATH_DOCS:
        cot = lm.generate_until(d["q"], stop=[])
        pred = extract_last_number(cot)
        ok = int(pred == d["gold"])
        correct += ok
        print(f"  Q: {d['q']}")
        print(f"     CoT='{cot}'")
        print(f"     extracted={pred!r}  gold={d['gold']!r}  correct={ok}")
    return {"exact_match (flexible-extract)": correct / len(MATH_DOCS)}


def main() -> int:
    qa_lm = MockGenLM({
        "hamlet": "William Shakespeare",
        "gas do plants": "Carbon dioxide.",
        "capital of italy": "Rome",
    })
    math_lm = MockGenLM({
        "natalia": "April 48, May 48/2=24, total 48+24=72. The answer is 72.",
        "robe": "Blue 2 bolts, white 2/2=1 bolt, total 2+1=3. The answer is 3.",
    })

    banner("Short-answer QA: exact_match + token-F1")
    print("(F1 rewards partial overlap; EM is all-or-nothing after normalisation)\n")
    qa = run_qa(qa_lm)

    banner("GSM8K-style math: generate CoT, then flexible-extract the last number")
    math = run_math(math_lm)

    banner("Aggregate")
    for k, v in {**qa, **math}.items():
        print(f"  {k:34s} = {v:.3f}")

    print("\nKey lesson: generation metrics depend on (a) normalisation and")
    print("(b) the extraction regex. A correct answer phrased differently can")
    print("fail EM but pass F1 — always report which metric and which extractor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

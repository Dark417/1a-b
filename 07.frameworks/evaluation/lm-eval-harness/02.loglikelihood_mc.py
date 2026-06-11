"""02 · Multiple-choice scoring via log-likelihood (the MMLU mechanism).

This is the single most important idea in academic LLM eval: **the model never
generates**. To score a multiple-choice question the harness asks the model, for
each candidate continuation, "what log-probability do you assign to producing
*this* string given the context?" — i.e. ``loglikelihood(context, continuation)``
— and picks the **argmax**. No parsing, no "the answer is B" extraction, no
sensitivity to formatting.

Two metrics fall out of the per-choice log-likelihoods:

  * ``acc``      — argmax over the *raw* summed log-prob of each continuation.
  * ``acc_norm`` — argmax over log-prob **normalised by continuation length**
    (in characters or tokens). This corrects the bias toward *shorter* answers
    (fewer tokens → higher total log-prob purely by length). MMLU/HellaSwag
    report ``acc_norm``.

We supply a deterministic **mock LM** implementing the same two primitives the
real harness's model adapters expose: ``loglikelihood`` and ``generate_until``.
Swap it for ``hf``/``vllm`` and the scoring loop below is unchanged.

Reference — output types & metrics:
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
HellaSwag acc_norm rationale: Zellers et al. 2019, https://arxiv.org/abs/1905.07830

Run:  python 02.loglikelihood_mc.py     (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import banner, tokenize  # noqa: E402


# Re-declare the tiny task inline (file-01 module name starts with a digit, which
# is awkward to import; the dataset is tiny so we just repeat it).
TEST_DOCS = [
    {"country": "Italy", "choices": ["Milan", "Rome", "Naples", "Turin"], "answer": 1},
    {"country": "Spain", "choices": ["Madrid", "Barcelona", "Seville", "Valencia"], "answer": 0},
    {"country": "Canada", "choices": ["Toronto", "Ottawa", "Montreal", "Vancouver"], "answer": 1},
]


def context_for(doc: dict) -> str:
    lines = [f"Question: What is the capital of {doc['country']}?", "Answer:"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A mock LM exposing the harness's two primitives. It is a unigram language
# model whose word probabilities are biased toward a small "knowledge" set, so
# the correct capital gets a higher log-likelihood. Deterministic.
# ---------------------------------------------------------------------------
class MockLM:
    """Stand-in for an ``hf``/``vllm`` adapter.

    loglikelihood(context, continuation) -> (logprob, is_greedy)
    generate_until(context, stop)        -> str
    """

    def __init__(self, capitals: dict[str, str]):
        # Words the model "knows" are correct capitals get extra probability.
        self.known = {c.lower() for c in capitals.values()}
        # A tiny background unigram vocabulary so every word has nonzero prob.
        self.vocab_logp = -3.0  # log prob for an unknown/background word

    def loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        """Sum of per-token log-probabilities of `continuation` given `context`.

        Real adapters compute this from the model's logits with a single forward
        pass; here we use a transparent unigram rule so the demo is explainable.
        """
        # Which capital does this context ask about? (peek for the demo)
        toks = tokenize(continuation)
        if not toks:
            return -100.0, False
        logp = 0.0
        for t in toks:
            if t in self.known and t in context.lower() + " " + continuation.lower():
                # The model is "more confident" about real capitals it recognises.
                logp += math.log(0.55)
            elif t in self.known:
                logp += math.log(0.35)
            else:
                logp += self.vocab_logp
        # is_greedy: would greedy decoding have produced exactly this string?
        return logp, False

    def generate_until(self, context: str, stop: list[str]) -> str:
        # Not used for multiple-choice; provided for adapter completeness.
        return "(generation not used for multiple_choice)"


# ---------------------------------------------------------------------------
# The scoring loop — IDENTICAL to what the real harness runs.
# ---------------------------------------------------------------------------
def score_mc(lm: MockLM, docs: list[dict]) -> dict[str, float]:
    n = len(docs)
    correct_acc = 0
    correct_acc_norm = 0
    for doc in docs:
        ctx = context_for(doc)
        choices = doc["choices"]
        # One loglikelihood request PER choice (the harness batches these).
        lls = [lm.loglikelihood(ctx, " " + c)[0] for c in choices]
        # acc: raw argmax.
        pred_acc = max(range(len(choices)), key=lambda i: lls[i])
        # acc_norm: normalise by continuation length in CHARACTERS (harness default).
        lls_norm = [ll / max(1, len(c)) for ll, c in zip(lls, choices)]
        pred_norm = max(range(len(choices)), key=lambda i: lls_norm[i])

        gold = doc["answer"]
        correct_acc += int(pred_acc == gold)
        correct_acc_norm += int(pred_norm == gold)

        print(f"  {doc['country']:8s} | "
              f"lls={[round(x, 2) for x in lls]} "
              f"-> acc pick='{choices[pred_acc]}' "
              f"acc_norm pick='{choices[pred_norm]}' "
              f"(gold='{choices[gold]}')")
    return {"acc": correct_acc / n, "acc_norm": correct_acc_norm / n}


def main() -> int:
    capitals = {"Italy": "Rome", "Spain": "Madrid", "Canada": "Ottawa"}
    lm = MockLM(capitals)

    banner("Log-likelihood multiple-choice scoring (MMLU mechanism)")
    print("For each choice we request loglikelihood(context, ' '+choice)")
    print("and take the argmax. No text is generated.\n")
    metrics = score_mc(lm, TEST_DOCS)

    banner("Aggregate metrics")
    for k, v in metrics.items():
        print(f"  {k:9s} = {v:.3f}")

    print("\nWhy acc_norm? Longer continuations accumulate more (negative) log-prob")
    print("purely from having more tokens. Normalising by length removes that")
    print("surface bias — this is why HellaSwag/MMLU report acc_norm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

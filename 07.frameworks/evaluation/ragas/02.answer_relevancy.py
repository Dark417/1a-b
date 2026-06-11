"""02 · Answer relevancy, from scratch (RAGAS).

Answer-relevancy answers: *does the answer actually address the question that was
asked?* — independently of whether it is correct. A rambling, off-topic, or
evasive answer scores low even if it contains true statements.

The clever trick RAGAS uses (reverse-questioning):

  1. Feed the **answer** to an LLM and ask it to generate N questions that this
     answer would be a good response to.
  2. Embed those generated questions and the **original** question.
  3. ``answer_relevancy = mean cosine similarity`` between each generated
     question and the original.

Intuition: if the answer is on-topic, the questions it "implies" cluster around
the real question → high similarity. If it's evasive/off-topic, the implied
questions drift away → low similarity. RAGAS also penalises *non-committal*
answers ("I don't know") by detecting them and zeroing the score.

We use a deterministic **MockLLM** (to reverse-generate questions) and a
**MockEmbedder** (bag-of-words cosine) from ``_lib.py`` so it runs offline.

Docs: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/

Run:  python 02.answer_relevancy.py     (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import MockEmbedder, banner, tokenize  # noqa: E402


# A non-committal detector (RAGAS does this with the judge LLM).
NONCOMMITTAL = ("i don't know", "i do not know", "not sure", "cannot answer",
                "no idea", "unable to")


def reverse_generate_questions(answer: str, n: int = 3) -> list[str]:
    """Mock the 'given this answer, what question does it answer?' step.

    Deterministic stand-in: build candidate questions by templating salient
    content words from the answer. A real RAGAS run calls an LLM here.
    """
    toks = [t for t in tokenize(answer) if len(t) > 3]
    # Use the most 'contentful' tokens as question topics.
    seen, topics = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            topics.append(t)
        if len(topics) >= n:
            break
    if not topics:
        topics = ["this"]
    templates = ["what is {}", "tell me about {}", "where is {}"]
    return [templates[i % len(templates)].format(topics[i % len(topics)])
            for i in range(n)]


def answer_relevancy(question: str, answer: str, embedder: MockEmbedder,
                     n: int = 3) -> tuple[float, list[str]]:
    if any(p in answer.lower() for p in NONCOMMITTAL):
        return 0.0, []  # non-committal answers get 0 (RAGAS convention)
    gen_qs = reverse_generate_questions(answer, n=n)
    q_vec = embedder.embed(question)
    sims = [MockEmbedder.cosine(q_vec, embedder.embed(gq)) for gq in gen_qs]
    return sum(sims) / len(sims), gen_qs


CASES = [
    {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris, a city on the river Seine.",
        "label": "on-topic answer",
    },
    {
        "question": "What is the capital of France?",
        "answer": "France has a rich history of cheese and wine production.",
        "label": "off-topic answer",
    },
    {
        "question": "What is the capital of France?",
        "answer": "I don't know the answer to that.",
        "label": "non-committal answer",
    },
]


def main() -> int:
    emb = MockEmbedder(dim=128)
    banner("Answer-relevancy = mean cosine(generated_qs, original_q)")
    for case in CASES:
        score, gen = answer_relevancy(case["question"], case["answer"], emb)
        print(f"\n[{case['label']}]  answer_relevancy = {score:.3f}")
        print(f"  Q: {case['question']}")
        print(f"  A: {case['answer']}")
        if gen:
            print(f"  reverse-generated questions: {gen}")
        else:
            print("  (non-committal detected -> score forced to 0)")

    print("\nLesson: relevancy measures FOCUS, not correctness. A fluent but")
    print("off-topic answer scores low; always pair with faithfulness/accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

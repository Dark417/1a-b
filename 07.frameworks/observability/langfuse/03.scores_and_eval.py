"""03 — Scores & evaluation in Langfuse.

Run: python 03.scores_and_eval.py   (offline; exits 0)

A *score* is an evaluation attached to a trace or observation. Three value types:
  - NUMERIC      (e.g. relevance 0.0..1.0)
  - CATEGORICAL  (e.g. "good"/"bad", a label)
  - BOOLEAN      (e.g. hallucinated True/False)

Sources you'll use in practice:
  - user feedback (thumbs up/down)
  - LLM-as-judge  (a model grades the output)
  - code/heuristic checks (regex, length, JSON-valid)

We compute three scores offline with a deterministic mock judge and attach them.
With keys set, they appear on the trace in the UI and aggregate in dashboards.
"""
from __future__ import annotations

import json

from _common import MockLLM, get_langfuse, observe

lf = get_langfuse()
llm = MockLLM("gpt-4o-mini")


# ---- deterministic, offline evaluators (stand-ins for real judges) ----------
def judge_relevance(question: str, answer: str) -> float:
    """Toy LLM-as-judge: overlap of question keywords with the answer -> 0..1."""
    q = set(question.lower().replace("?", "").split())
    a = set(answer.lower().split())
    return round(len(q & a) / max(1, len(q)), 3)


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def user_feedback(answer: str) -> str:
    return "good" if answer else "bad"


def add_score(name, value, *, data_type=None, comment=""):
    """Attach a score to the current trace, tolerant of client variants."""
    for fn in ("score_current_trace", "create_score"):
        method = getattr(lf, fn, None)
        if method is None:
            continue
        try:
            kwargs = {"name": name, "value": value, "comment": comment}
            if data_type:
                kwargs["data_type"] = data_type
            method(**kwargs)
            return
        except Exception:
            continue


@observe()
def answer_and_score(question: str) -> dict:
    out = llm.chat(question)
    answer = out["output"]

    rel = judge_relevance(question, answer)
    add_score("relevance", rel, data_type="NUMERIC", comment="mock llm-judge")
    add_score(
        "json_valid",
        is_valid_json(answer),
        data_type="BOOLEAN",
        comment="code check",
    )
    add_score(
        "user_feedback",
        user_feedback(answer),
        data_type="CATEGORICAL",
        comment="thumbs",
    )
    return {"answer": answer, "relevance": rel}


def main() -> None:
    q = "What is the capital of France?"
    result = answer_and_score(q)
    print("answer:", result["answer"])
    print("relevance score:", result["relevance"])

    try:
        lf.flush()
    except Exception:
        pass

    assert 0.0 <= result["relevance"] <= 1.0
    print("OK: numeric + boolean + categorical scores attached to the trace.")


if __name__ == "__main__":
    main()

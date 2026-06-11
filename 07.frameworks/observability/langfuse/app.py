"""app.py — end-to-end Langfuse-traced RAG pipeline, offline-safe.

Run: python app.py   (exits 0 with or without langfuse / API keys)

Pulls every feature together:
  - prompt management (fetch + compile a versioned prompt)
  - nested @observe spans (retrieve) and a generation (LLM)
  - trace metadata (user/session/tags)
  - scores (relevance + grounding) attached to the trace

With LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST set, open the trace in the UI.
"""
from __future__ import annotations

from _common import MockLLM, get_langfuse, is_exporting, observe

lf = get_langfuse()
llm = MockLLM("gpt-4o-mini")

CORPUS = {
    "paris": "Paris is the capital of France, on the Seine.",
    "eiffel": "The Eiffel Tower is a Paris landmark built in 1889.",
    "berlin": "Berlin is the capital of Germany.",
}


def _load_prompt_module():
    # File is named 04.prompt_management.py (not importable normally); load it.
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(__file__), "04.prompt_management.py")
    spec = importlib.util.spec_from_file_location("prompt_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@observe()
def retrieve(question: str, k: int = 2) -> list[str]:
    q = set(question.lower().replace("?", "").split())
    scored = sorted(
        CORPUS.values(), key=lambda d: len(q & set(d.lower().split())), reverse=True
    )
    return scored[:k]


@observe(as_type="generation")
def generate(prompt: str) -> str:
    out = llm.chat(prompt)
    try:
        lf.update_current_generation(
            model=out["model"],
            input=prompt,
            output=out["output"],
            usage_details={"input": out["input_tokens"], "output": out["output_tokens"]},
        )
    except Exception:
        pass
    return out["output"]


def add_score(name, value, comment=""):
    for fn in ("score_current_trace", "create_score"):
        m = getattr(lf, fn, None)
        if m:
            try:
                m(name=name, value=value, comment=comment)
                return
            except Exception:
                continue


@observe()
def rag(question: str, prompt_template) -> dict:
    docs = retrieve(question, k=2)
    prompt = prompt_template.compile(question=question, context=" ".join(docs))
    answer = generate(prompt)

    # evaluation scores
    q = set(question.lower().replace("?", "").split())
    relevance = round(len(q & set(answer.lower().split())) / max(1, len(q)), 3)
    grounded = 1.0 if any(answer.split()[-1].strip(".") in d for d in docs) else 0.0
    add_score("relevance", relevance, "mock judge")
    add_score("grounded", grounded, "overlap heuristic")

    try:
        lf.update_current_trace(
            name="rag-app",
            user_id="demo",
            session_id="s1",
            tags=["app", "offline"],
            input={"question": question},
            output={"answer": answer},
        )
    except Exception:
        pass
    return {"answer": answer, "relevance": relevance, "docs": docs}


def main() -> None:
    print("exporting to langfuse server:", is_exporting())
    pmod = _load_prompt_module()
    template = pmod.get_prompt("qa", label="production")

    for q in ["What is the capital of France?", "Where is the Eiffel Tower?"]:
        res = rag(q, template)
        print(f"\nQ: {q}\nA: {res['answer']}  (relevance={res['relevance']})")

    try:
        lf.flush()
    except Exception:
        pass

    print("\nOK: end-to-end RAG traced (prompt + spans + generation + scores).")


if __name__ == "__main__":
    main()

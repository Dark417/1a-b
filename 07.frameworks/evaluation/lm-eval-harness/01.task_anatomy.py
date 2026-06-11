"""01 · The anatomy of a Task (lm-evaluation-harness), from scratch.

A *Task* in lm-eval-harness is a small contract that turns a dataset row into a
prompt + gold target and declares how to score it. The three functions you
*always* implement are:

  * ``doc_to_text(doc)``   — render the row into the question/context shown to
    the model (the "context");
  * ``doc_to_target(doc)`` — the gold answer (a string, or an index into a list
    of choices for multiple-choice tasks);
  * ``doc_to_choices(doc)``— for multiple-choice, the candidate continuations.

On top of that sits **few-shot assembly**: the harness samples ``num_fewshot``
labelled examples from a *training/dev* split and prepends them, each rendered
with the SAME ``doc_to_text`` + the gold target, separated by a delimiter. The
test doc is rendered last *without* its target — that's the prompt.

This file builds a Task object and shows exactly how a 0-shot and a 3-shot
prompt are assembled. No model is called here; the next files run the model.

Docs (task guide):
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
New-task tutorial:
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md

Run:  python 01.task_anatomy.py      (stdlib only, exit 0, offline)
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import banner  # noqa: E402


# ---------------------------------------------------------------------------
# A tiny dataset. Each "doc" is a dict — exactly how HF datasets rows arrive.
# Task: capital-of-country, framed as 4-way multiple choice.
# ---------------------------------------------------------------------------
TRAIN_DOCS = [
    {"country": "France", "choices": ["Paris", "Rome", "Berlin", "Madrid"], "answer": 0},
    {"country": "Japan", "choices": ["Seoul", "Tokyo", "Beijing", "Osaka"], "answer": 1},
    {"country": "Egypt", "choices": ["Cairo", "Giza", "Luxor", "Alexandria"], "answer": 0},
    {"country": "Brazil", "choices": ["Rio", "Brasília", "São Paulo", "Salvador"], "answer": 1},
]
TEST_DOCS = [
    {"country": "Italy", "choices": ["Milan", "Rome", "Naples", "Turin"], "answer": 1},
    {"country": "Spain", "choices": ["Madrid", "Barcelona", "Seville", "Valencia"], "answer": 0},
]


# ---------------------------------------------------------------------------
# The Task contract. In the real harness this is a YAML file + a Python class;
# here we use a dataclass so the moving parts are explicit and inspectable.
# ---------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    train_docs: list[dict]
    test_docs: list[dict]
    # output_type drives scoring: "multiple_choice" (loglikelihood) vs
    # "generate_until" (free generation). See files 02 and 03.
    output_type: str = "multiple_choice"
    # The functions that define the task:
    doc_to_text: Callable[[dict], str] = field(default=lambda d: "")
    doc_to_target: Callable[[dict], int] = field(default=lambda d: 0)
    doc_to_choices: Callable[[dict], list[str]] = field(default=lambda d: [])
    # Delimiter between few-shot examples (the harness default is "\n\n").
    fewshot_delimiter: str = "\n\n"
    target_delimiter: str = " "  # between the rendered text and its target

    def render_example(self, doc: dict, *, with_target: bool) -> str:
        """One labelled example: '<text><target_delim><gold choice>'.

        For the test doc we omit the target so the model can complete it.
        """
        text = self.doc_to_text(doc)
        if not with_target:
            return text
        gold_idx = self.doc_to_target(doc)
        gold = self.doc_to_choices(doc)[gold_idx]
        return f"{text}{self.target_delimiter}{gold}"

    def build_prompt(self, test_doc: dict, num_fewshot: int, seed: int = 0) -> str:
        """Assemble the full few-shot prompt for one test doc.

        The harness samples shots from the train/dev split (NOT the test split,
        to avoid leakage), renders each WITH its gold target, then appends the
        test doc WITHOUT its target.
        """
        rng = random.Random(seed)
        pool = list(self.train_docs)
        rng.shuffle(pool)
        shots = pool[:num_fewshot]
        parts = [self.render_example(d, with_target=True) for d in shots]
        parts.append(self.render_example(test_doc, with_target=False))
        return self.fewshot_delimiter.join(parts)


# ---------------------------------------------------------------------------
# Define our concrete task by plugging in the three functions.
# ---------------------------------------------------------------------------
def make_capitals_task() -> Task:
    def doc_to_text(doc: dict) -> str:
        # MMLU-style: a question + lettered choices, asking for the answer.
        lines = [f"Question: What is the capital of {doc['country']}?"]
        for letter, choice in zip("ABCD", doc["choices"]):
            lines.append(f"{letter}. {choice}")
        lines.append("Answer:")
        return "\n".join(lines)

    return Task(
        name="capitals",
        train_docs=TRAIN_DOCS,
        test_docs=TEST_DOCS,
        output_type="multiple_choice",
        doc_to_text=doc_to_text,
        doc_to_target=lambda d: d["answer"],
        doc_to_choices=lambda d: d["choices"],
    )


def main() -> int:
    task = make_capitals_task()
    test_doc = task.test_docs[0]

    banner(f"Task '{task.name}'  (output_type={task.output_type})")
    print("A single rendered example WITH its gold target:")
    print("-" * 60)
    print(task.render_example(task.train_docs[0], with_target=True))
    print("-" * 60)

    banner("0-shot prompt (num_fewshot=0)")
    print(task.build_prompt(test_doc, num_fewshot=0))

    banner("3-shot prompt (num_fewshot=3)")
    print(task.build_prompt(test_doc, num_fewshot=3, seed=0))

    print("\nNote: the gold target for the TEST doc is hidden from the prompt;")
    print("the model must produce it. In file 02 we score it via log-likelihood.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

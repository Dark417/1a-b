"""04 — Langfuse prompt management (versioned prompts), offline-safe.

Run: python 04.prompt_management.py   (offline; exits 0)

Prompt management lets you store named, versioned prompts in Langfuse, deploy a
version by *label* (e.g. "production"), fetch it at runtime, and compile its
{{variables}}. Generations can be linked to the exact prompt version used, so you
can correlate a prompt change with a quality/cost change.

Offline we can't fetch from a server, so this file demonstrates the SAME API
shape with a local in-memory prompt store + a `.compile()` that mirrors
Langfuse's `{{var}}` templating. If real keys are present we try the real
`get_prompt` and fall back gracefully.
"""
from __future__ import annotations

import re

from _common import MockLLM, get_langfuse

lf = get_langfuse()
llm = MockLLM("gpt-4o-mini")


# ---- a local stand-in mirroring Langfuse's ChatPromptClient/TextPromptClient -
class LocalPrompt:
    """Mimics langfuse prompt objects: .prompt text, .compile(**vars), .config."""

    def __init__(self, name, prompt, version, labels, config=None):
        self.name = name
        self.prompt = prompt
        self.version = version
        self.labels = labels
        self.config = config or {}

    def compile(self, **variables) -> str:
        # Langfuse templates use {{variable}} double-brace syntax.
        def repl(m):
            return str(variables.get(m.group(1).strip(), m.group(0)))

        return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", repl, self.prompt)


# A tiny offline registry keyed by (name, label) — what the server stores.
_LOCAL_REGISTRY = {
    ("qa", "production"): LocalPrompt(
        name="qa",
        version=2,
        labels=["production", "latest"],
        prompt=(
            "You are a helpful assistant.\n"
            "Answer the question using ONLY the context.\n"
            "Question: {{question}}\n"
            "Context: {{context}}\n"
            "Answer:"
        ),
        config={"model": "gpt-4o-mini", "temperature": 0.0},
    ),
    ("qa", "latest"): LocalPrompt(
        name="qa",
        version=1,
        labels=["latest"],
        prompt="Q: {{question}}\nA:",
        config={"model": "gpt-4o-mini"},
    ),
}


def get_prompt(name: str, label: str = "production") -> LocalPrompt:
    """Try the real Langfuse prompt store; fall back to the local registry."""
    try:
        real = lf.get_prompt(name, label=label)
        # Wrap the real client so the rest of the example is uniform.
        return real  # real object also has .compile and .config
    except Exception:
        return _LOCAL_REGISTRY[(name, label)]


def main() -> None:
    prompt = get_prompt("qa", label="production")
    print(f"fetched prompt '{prompt.name}' v{getattr(prompt, 'version', '?')} "
          f"labels={getattr(prompt, 'labels', [])}")
    print(f"config: {getattr(prompt, 'config', {})}")

    # Compile the template with runtime variables.
    compiled = prompt.compile(
        question="What is the capital of France?",
        context="Paris is the capital of France.",
    )
    print("\n--- compiled prompt ---")
    print(compiled)

    # Use it. In production you'd link the generation to prompt=prompt so the UI
    # shows which version produced this output.
    out = llm.chat(compiled)
    print("\nanswer:", out["output"])

    try:
        lf.flush()
    except Exception:
        pass

    assert "{{" not in compiled, "all variables should be substituted"
    assert "Paris" in compiled
    print("\nOK: fetched a labelled prompt version and compiled its {{variables}}.")


if __name__ == "__main__":
    main()

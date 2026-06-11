"""Shared helpers for the Langfuse tutorial — offline-safe.

- get_langfuse(): returns a real Langfuse client if the SDK is importable.
  Without API keys the v3+ client auto-disables into a no-op, so every call is
  still valid and the script runs offline. If the package is missing entirely we
  return a tiny stub so the examples STILL run and exit 0.
- observe: the real @observe decorator if available, else a no-op passthrough.
- MockLLM: deterministic, offline stand-in for a chat model.
"""
from __future__ import annotations

import contextlib
import functools
import os

LANGFUSE_AVAILABLE = False
try:  # the real SDK (v3/v4, OpenTelemetry-based)
    from langfuse import Langfuse, get_client, observe  # type: ignore

    LANGFUSE_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when package missing
    Langfuse = None  # type: ignore

    def observe(_fn=None, **_kw):  # type: ignore
        """No-op stand-in for langfuse.observe when the SDK isn't installed."""

        def deco(fn):
            @functools.wraps(fn)
            def wrap(*a, **k):
                return fn(*a, **k)

            return wrap

        return deco(_fn) if _fn is not None else deco

    def get_client(*_a, **_k):  # type: ignore
        return _StubClient()


class _StubSpan:
    """Mimics the v3 observation context manager surface as a no-op."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, *a, **k):
        return self

    def update_trace(self, *a, **k):
        return self

    def score(self, *a, **k):
        return self

    def score_trace(self, *a, **k):
        return self


class _StubClient:
    """A no-op client used only when langfuse isn't installed at all."""

    def start_as_current_observation(self, *a, **k):
        return _StubSpan()

    def start_as_current_span(self, *a, **k):
        return _StubSpan()

    def start_as_current_generation(self, *a, **k):
        return _StubSpan()

    def update_current_trace(self, *a, **k):
        pass

    def update_current_generation(self, *a, **k):
        pass

    def update_current_span(self, *a, **k):
        pass

    def create_score(self, *a, **k):
        pass

    def score_current_trace(self, *a, **k):
        pass

    def score_current_span(self, *a, **k):
        pass

    def get_prompt(self, *a, **k):
        raise RuntimeError("langfuse not installed")

    def auth_check(self):
        return False

    def flush(self):
        pass


def get_langfuse():
    """Return a Langfuse client (real or stub). Never raises, never needs net."""
    if LANGFUSE_AVAILABLE:
        # Without keys this is auto-disabled (no-op export) — perfect offline.
        return get_client()
    return _StubClient()


def is_exporting() -> bool:
    """True only if real keys are present and auth succeeds (i.e. data is sent)."""
    if not LANGFUSE_AVAILABLE:
        return False
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return False
    with contextlib.suppress(Exception):
        return bool(get_client().auth_check())
    return False


class MockLLM:
    """Deterministic offline chat model. Returns a canned reply + token usage."""

    def __init__(self, model: str = "mock-llm"):
        self.model = model

    @staticmethod
    def _tok(text) -> int:
        return max(1, len(str(text).split()))

    def chat(self, prompt: str) -> dict:
        completion = f"[{self.model}] " + _canned(prompt)
        return {
            "model": self.model,
            "output": completion,
            "input_tokens": self._tok(prompt),
            "output_tokens": self._tok(completion),
        }


def _canned(prompt: str) -> str:
    p = prompt.lower()
    if "capital" in p and "france" in p:
        return "Paris is the capital of France."
    if "eiffel" in p:
        return "The Eiffel Tower is in Paris."
    if "color" in p:
        return "Red, green, and blue."
    return "This is a deterministic mock answer."

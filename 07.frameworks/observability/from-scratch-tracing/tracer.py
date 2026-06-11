"""A tiny, dependency-free LLM tracer — the core of every observability tool.

This module implements the primitives that Langfuse, Phoenix and OpenTelemetry
all share, using only the Python standard library:

    Trace  -> a top-level run (a trace_id)
    Span   -> a timed node in a tree (name, attributes, events, children)
    Generation -> a Span specialised for an LLM call (model, I/O, usage, cost)
    Score  -> a number/label attached to a span after the fact (an eval)

The key trick that makes nesting ergonomic — so a deeply nested function knows
its parent without you threading it through every call — is a per-context
*stack of the current span*, held in a `contextvars.ContextVar`. This is exactly
how OpenTelemetry propagates the "current span".

References:
- OpenTelemetry traces: https://opentelemetry.io/docs/concepts/signals/traces/
- GenAI semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- contextvars: https://docs.python.org/3/library/contextvars.html
"""
from __future__ import annotations

import contextvars
import functools
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Context propagation: a stack of the currently-open spans for THIS context.
# A deeply nested span reads the top of this stack to find its parent.
# ---------------------------------------------------------------------------
_SPAN_STACK: contextvars.ContextVar[tuple["Span", ...]] = contextvars.ContextVar(
    "_span_stack", default=()
)


@dataclass
class Score:
    """An evaluation attached to a span/trace: name, value, optional comment.

    Mirrors Langfuse 'scores' and Phoenix 'evals': a numeric or categorical
    judgement (user feedback, LLM-as-judge, a regex check) recorded after a run.
    """

    name: str
    value: float
    comment: str = ""


@dataclass
class Span:
    """A timed node in the trace tree.

    Attributes mirror OTel/OpenInference: arbitrary key/values describing the
    operation. `kind="generation"` marks an LLM call (the special case everyone
    cares about) which additionally carries `usage` and `cost_usd`.
    """

    name: str
    kind: str = "span"  # "span" | "generation" | "retriever" | "tool" | ...
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    trace_id: str = ""
    parent_id: Optional[str] = None
    start: float = 0.0
    end: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[float, str]] = field(default_factory=list)
    scores: list[Score] = field(default_factory=list)
    children: list["Span"] = field(default_factory=list)
    # LLM-only:
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0

    @property
    def latency_ms(self) -> float:
        if self.end is None:
            return 0.0
        return (self.end - self.start) * 1000.0

    def set(self, **attrs: Any) -> "Span":
        """Attach attributes (chainable). Redact sensitive keys here in prod."""
        self.attributes.update(attrs)
        return self

    def event(self, message: str) -> "Span":
        """Record a point-in-time event (a log line tied to wall-clock)."""
        self.events.append((time.perf_counter(), message))
        return self

    def score(self, name: str, value: float, comment: str = "") -> "Span":
        """Attach an evaluation score to this span."""
        self.scores.append(Score(name, float(value), comment))
        return self

    # roll-ups -----------------------------------------------------------
    def rollup_tokens(self) -> dict[str, int]:
        """Sum usage across this span and all descendants (no double-count:
        usage lives only on generation spans)."""
        total: dict[str, int] = dict(self.usage)
        for c in self.children:
            for k, v in c.rollup_tokens().items():
                total[k] = total.get(k, 0) + v
        return total

    def rollup_cost(self) -> float:
        return self.cost_usd + sum(c.rollup_cost() for c in self.children)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable export (what an OTLP exporter would ship)."""
        d: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "latency_ms": round(self.latency_ms, 3),
            "attributes": self.attributes,
            "scores": [s.__dict__ for s in self.scores],
        }
        if self.usage:
            d["usage"] = self.usage
        if self.cost_usd:
            d["cost_usd"] = round(self.cost_usd, 6)
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


# Default price table: USD per 1,000 tokens (toy numbers, illustrative only).
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    # model: (input_per_1k, output_per_1k)
    "mock-llm": (0.0, 0.0),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-haiku": (0.0008, 0.004),
    "llama-local": (0.0, 0.0),
}


class Tracer:
    """Collects root spans, owns the price table, renders the tree."""

    def __init__(self, prices: Optional[dict[str, tuple[float, float]]] = None):
        self.prices = dict(DEFAULT_PRICES)
        if prices:
            self.prices.update(prices)
        self.roots: list[Span] = []

    @contextmanager
    def span(self, name: str, kind: str = "span", **attrs: Any):
        """Open a span; auto-nest under the current span via the context stack.

        Usage:
            with tracer.span("agent.run") as s:
                with tracer.span("retriever.search", k=5):
                    ...
        """
        stack = _SPAN_STACK.get()
        parent = stack[-1] if stack else None
        sp = Span(
            name=name,
            kind=kind,
            trace_id=parent.trace_id if parent else uuid.uuid4().hex,
            parent_id=parent.span_id if parent else None,
            start=time.perf_counter(),
            attributes=dict(attrs),
        )
        token = _SPAN_STACK.set(stack + (sp,))
        try:
            yield sp
        finally:
            sp.end = time.perf_counter()
            _SPAN_STACK.reset(token)
            if parent is not None:
                parent.children.append(sp)
            else:
                self.roots.append(sp)

    def generation(self, name: str = "llm.generate", **attrs: Any):
        """Convenience: open a span of kind='generation' for an LLM call."""
        return self.span(name, kind="generation", **attrs)

    def record_generation(
        self,
        span: Span,
        *,
        model: str,
        prompt: Any,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> Span:
        """Fill an open generation span with LLM I/O + usage and compute cost
        from the price table. This is the heart of 'LLM observability'."""
        in_price, out_price = self.prices.get(model, (0.0, 0.0))
        cost = (prompt_tokens / 1000.0) * in_price + (
            completion_tokens / 1000.0
        ) * out_price
        span.set(model=model, input=prompt, output=completion)
        span.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        span.cost_usd = cost
        return span

    # rendering ----------------------------------------------------------
    def print_tree(self) -> None:
        for root in self.roots:
            self._print(root, 0)

    def _print(self, sp: Span, depth: int) -> None:
        pad = "  " * depth
        tag = f"[{sp.kind.upper()}] " if sp.kind != "span" else ""
        line = f"{pad}{tag}{sp.name}  {sp.latency_ms:6.2f} ms"
        if sp.usage:
            line += (
                f"  tok={sp.usage.get('total_tokens', 0)}"
                f" (${sp.cost_usd:.6f})"
            )
        print(line)
        for k, v in sp.attributes.items():
            vs = str(v)
            if len(vs) > 60:
                vs = vs[:57] + "..."
            print(f"{pad}    · {k}: {vs}")
        for s in sp.scores:
            extra = f" — {s.comment}" if s.comment else ""
            print(f"{pad}    ◆ score[{s.name}] = {s.value}{extra}")
        for c in sp.children:
            self._print(c, depth + 1)

    def summary(self) -> dict[str, Any]:
        tokens: dict[str, int] = {}
        cost = 0.0
        for r in self.roots:
            for k, v in r.rollup_tokens().items():
                tokens[k] = tokens.get(k, 0) + v
            cost += r.rollup_cost()
        return {"traces": len(self.roots), "tokens": tokens, "cost_usd": round(cost, 6)}


# ---------------------------------------------------------------------------
# The @observe decorator — the Langfuse ergonomic: wrap any function in a span,
# capturing args as input and the return value as output, automatically.
# ---------------------------------------------------------------------------
_DEFAULT_TRACER = Tracer()


def get_tracer() -> Tracer:
    return _DEFAULT_TRACER


def observe(
    _fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    kind: str = "span",
    tracer: Optional[Tracer] = None,
):
    """Decorator that traces a function call as a span.

    Example:
        @observe
        def retrieve(q): ...

        @observe(kind="generation", name="llm")
        def generate(prompt): ...
    """

    def decorate(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tr = tracer or _DEFAULT_TRACER
            with tr.span(name or fn.__name__, kind=kind) as sp:
                sp.set(input={"args": _short(args), "kwargs": _short(kwargs)})
                result = fn(*args, **kwargs)
                sp.set(output=_short(result))
                return result

        return wrapper

    if _fn is not None:  # used as @observe without parens
        return decorate(_fn)
    return decorate


def _short(obj: Any, limit: int = 200) -> Any:
    s = repr(obj)
    return s if len(s) <= limit else s[:limit] + "..."


# A deterministic, offline "LLM" used by the examples so everything runs without
# a network. It returns a canned reply and reports plausible token counts.
class MockLLM:
    """Deterministic stand-in for a chat model. No network, fully reproducible."""

    def __init__(self, model: str = "mock-llm"):
        self.model = model

    @staticmethod
    def _count_tokens(text: str) -> int:
        # Toy tokenizer: ~1 token per whitespace-word (good enough to teach).
        return max(1, len(str(text).split()))

    def generate(self, prompt: str) -> dict[str, Any]:
        completion = f"[mock:{self.model}] answer to: {str(prompt)[:40]}"
        return {
            "model": self.model,
            "completion": completion,
            "prompt_tokens": self._count_tokens(prompt),
            "completion_tokens": self._count_tokens(completion),
        }

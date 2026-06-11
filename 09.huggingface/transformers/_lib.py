"""
Shared helper utilities for the transformers tutorial.
Provides safe(), banner(), and note_skip().
"""


def safe(fn, *a, **k):
    """
    Call fn(*a, **k) and return (True, result) on success,
    or (False, exception) on any exception.
    Usage:
        ok, result = safe(some_function, arg1, arg2)
    """
    try:
        return True, fn(*a, **k)
    except Exception as exc:
        return False, exc


def banner(title: str) -> None:
    """Print a bordered section title."""
    line = "=" * (len(title) + 4)
    print(f"\n{line}")
    print(f"| {title} |")
    print(f"{line}")


def note_skip(msg: str) -> None:
    """Print a standardised skip notice."""
    print(f"[skip] needs network/model — demonstrating API shape with local fallback")
    print(f"       reason: {msg}")

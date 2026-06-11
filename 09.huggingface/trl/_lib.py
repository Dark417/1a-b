"""
Shared utilities for the trl/ examples.
"""
import sys


def banner(title: str) -> None:
    """Print a prominent section banner."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def note_skip(msg: str) -> None:
    """Print a standardised skip/fallback notice."""
    print(f"[skip] {msg}")


def safe(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs), catching ALL exceptions.

    Returns (True, result) on success, (False, exception) on failure.
    Never raises.
    """
    try:
        return True, fn(*args, **kwargs)
    except Exception as exc:
        return False, exc

"""Shared utilities for the diffusers examples."""

import sys
import traceback


def banner(title: str) -> None:
    """Print a section banner."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def note_skip(msg: str) -> None:
    """Print a standardized skip notice."""
    print(f"[skip] {msg}")


def safe(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) safely.

    Returns (True, result) on success, (False, exception) on any exception.
    Never raises; always returns a 2-tuple.
    """
    try:
        result = fn(*args, **kwargs)
        return True, result
    except Exception as exc:
        return False, exc

"""Shared utilities for datasets/ examples."""

import sys


def banner(title: str) -> None:
    """Print a section banner."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def note_skip(msg: str) -> None:
    """Print a skip notice (network unavailable, etc.)."""
    print(f"[skip] {msg}")


def safe(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) and return (ok: bool, result).
    On any exception returns (False, exception_instance).
    """
    try:
        return True, fn(*args, **kwargs)
    except Exception as exc:
        return False, exc

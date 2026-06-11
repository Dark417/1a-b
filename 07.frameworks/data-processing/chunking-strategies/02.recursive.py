"""02 — Recursive character text splitting (LangChain's default, from scratch).

The single most popular general-purpose chunker is LangChain's
`RecursiveCharacterTextSplitter`. Its idea is elegant: instead of cutting at
arbitrary offsets, try to cut at *natural boundaries*, from coarsest to finest.

Algorithm (clean-room reimplementation from the public docs/behavior):

  separators = ["\\n\\n", "\\n", ". ", " ", ""]   # paragraph, line, sentence, word, char

  def split(text, separators):
      pick the FIRST separator that actually occurs in `text`
      split text on it -> pieces
      for each piece:
          if len(piece) <= chunk_size:
              keep it (it fits)
          else:
              RECURSE with the remaining (finer) separators
      then GREEDILY MERGE adjacent small pieces back together up to chunk_size,
      adding `chunk_overlap` carry-over between merged windows.

The recursion gives you the largest semantically-meaningful unit that still fits
the budget: whole paragraphs when they fit, else whole lines, else sentences,
else words, and only as a last resort hard character cuts.

Gotcha named here: **separator ORDER is the whole ballgame.** Put `" "` before
`"\\n"` and you destroy paragraph structure. The list must go coarse -> fine.
For code or markdown you swap in language-aware separator lists (LangChain ships
`from_language(Language.PYTHON)` etc.) — we show a markdown variant.

References:
  - LangChain text-splitter concepts:
    https://python.langchain.com/docs/concepts/text_splitters/
  - RecursiveCharacterTextSplitter how-to:
    https://python.langchain.com/docs/how_to/recursive_text_splitter/
  - Source behavior (libs/text-splitters): https://github.com/langchain-ai/langchain
"""
from __future__ import annotations

from typing import List


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_keep_separator(text: str, sep: str) -> List[str]:
    """Split on `sep` but re-attach the separator to the END of each piece.

    Keeping separators preserves the original text on re-join (important so that
    char offsets and meaning survive). Empty separator => split into characters.
    """
    if sep == "":
        return list(text)
    parts = text.split(sep)
    out: List[str] = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + sep)
        else:
            if p:
                out.append(p)
    return out


def _merge_splits(splits: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedily pack consecutive fine splits into <=chunk_size windows.

    This mirrors LangChain's `_merge_splits`: accumulate pieces until adding the
    next would exceed chunk_size, emit the buffer, then slide back by
    chunk_overlap characters to start the next window (the overlap carry-over).
    """
    merged: List[str] = []
    buf: List[str] = []
    total = 0
    for piece in splits:
        plen = len(piece)
        if total + plen > chunk_size and buf:
            merged.append("".join(buf))
            # Drop pieces from the FRONT until the remaining buffer is within the
            # overlap budget — that remainder becomes the head of the next window.
            while total > chunk_overlap and buf:
                total -= len(buf[0])
                buf.pop(0)
        buf.append(piece)
        total += plen
    if buf:
        merged.append("".join(buf))
    return merged


def recursive_split(
    text: str,
    chunk_size: int = 160,
    chunk_overlap: int = 24,
    separators: List[str] | None = None,
) -> List[str]:
    """Reimplementation of RecursiveCharacterTextSplitter.split_text."""
    if separators is None:
        separators = DEFAULT_SEPARATORS

    # 1. Choose the first separator that appears (fall through to the last "").
    sep = separators[-1]
    remaining = separators[separators.index(sep) + 1 :]
    for i, s in enumerate(separators):
        if s == "":
            sep = s
            remaining = separators[i + 1 :]
            break
        if s in text:
            sep = s
            remaining = separators[i + 1 :]
            break

    # 2. Split on the chosen separator.
    splits = _split_keep_separator(text, sep)

    # 3. For each piece: keep if it fits, else recurse with finer separators.
    good: List[str] = []
    to_merge: List[str] = []

    def flush_merge() -> None:
        if to_merge:
            good.extend(_merge_splits(to_merge, chunk_size, chunk_overlap))
            to_merge.clear()

    for piece in splits:
        if len(piece) <= chunk_size:
            to_merge.append(piece)
        else:
            flush_merge()
            if remaining:
                good.extend(recursive_split(piece, chunk_size, chunk_overlap, remaining))
            else:
                # No finer separator left: hard-cut the oversized piece.
                for j in range(0, len(piece), chunk_size):
                    good.append(piece[j : j + chunk_size])
    flush_merge()
    return [c for c in good if c.strip()]


PARAGRAPHS = (
    "Chunking for RAG.\n\n"
    "Why chunk at all? Embedding models have a maximum context, and a single "
    "vector can only summarize so much. Splitting documents into focused chunks "
    "keeps each vector topically tight, which sharpens retrieval precision.\n\n"
    "The recursive splitter tries paragraph breaks first, then line breaks, then "
    "sentence boundaries, then words. It always prefers the largest natural unit "
    "that fits the size budget. This keeps related text together far better than "
    "a blind fixed-size window.\n\n"
    "Order matters. If you put the space separator before the newline separator, "
    "the splitter would shatter paragraphs into words and lose all structure."
)

MARKDOWN = (
    "# Title\n\n"
    "Intro paragraph about the document.\n\n"
    "## Section A\n"
    "- bullet one\n- bullet two\n- bullet three\n\n"
    "## Section B\n"
    "Some narrative text that runs a little longer than the others so we can see "
    "how the recursive splitter packs it into a window."
)


def main() -> None:
    print("=" * 72)
    print("02 · RECURSIVE CHARACTER SPLITTING (LangChain default, from scratch)")
    print("=" * 72)

    print("\n[A] prose, chunk_size=160, overlap=24, separators=["
          r'"\n\n","\n",". "," ",""]')
    chunks = recursive_split(PARAGRAPHS, chunk_size=160, chunk_overlap=24)
    for i, c in enumerate(chunks):
        print(f"  --- chunk {i} ({len(c)} chars) ---")
        print("  " + c.replace("\n", "\\n"))

    print("\n[B] same text but WRONG separator order (space before newline):")
    bad = recursive_split(
        PARAGRAPHS, chunk_size=160, chunk_overlap=24,
        separators=[" ", "\n\n", "\n", ""],
    )
    first_bad = bad[0][:70].replace("\n", "\\n")
    print(f"  -> {len(bad)} chunks; first chunk = {first_bad!r}")
    print("  paragraph structure is destroyed because ' ' matched first.")

    print("\n[C] markdown-aware separators (headings first):")
    md_seps = ["\n## ", "\n# ", "\n\n", "\n", " ", ""]
    md_chunks = recursive_split(MARKDOWN, chunk_size=120, chunk_overlap=12, separators=md_seps)
    for i, c in enumerate(md_chunks):
        print(f"  --- md chunk {i} ({len(c)} chars) ---")
        print("  " + c.replace("\n", "\\n"))

    print("\nTakeaway: recursion + coarse->fine separators keeps natural units")
    print("intact. It is the sensible default for arbitrary prose.")


if __name__ == "__main__":
    main()

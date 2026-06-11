"""01 — Fixed-size chunking (character- and word-based) with overlap.

The simplest chunker: slide a fixed-width window over the text. It is fast,
deterministic, and has *no* dependencies — but it is "structure-blind": it will
happily cut a sentence, a word, or a markdown table in half because it only
counts units (characters or words), never meaning.

This file implements both variants from scratch and visualizes the central
failure mode (mid-token / mid-sentence boundaries) so you can *feel* why the
later strategies (recursive, token-based, semantic) exist.

Key parameters (the universal chunking knobs):
  - chunk_size : window width in units (chars or words).
  - overlap    : how many trailing units of chunk i are repeated at the start
                 of chunk i+1. Overlap fights "context fragmentation": a fact
                 split across a boundary survives in at least one chunk.
  - stride     : chunk_size - overlap. The window advances by `stride` each step.

Gotcha named here: overlap *duplicates* tokens. Total emitted tokens grow by
roughly chunk_size / stride, inflating your embedding bill and your index size.

References:
  - LangChain "CharacterTextSplitter" / text-splitter concepts:
    https://python.langchain.com/docs/concepts/text_splitters/
  - LlamaIndex SentenceSplitter / TokenTextSplitter node parsers:
    https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
  - Greg Kamradt, "5 Levels of Text Splitting" (Level 1 = fixed size):
    https://github.com/FullStackRetrieval-com/RetrievalTutorials
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


SAMPLE = (
    "Retrieval-augmented generation grounds a language model in your own data. "
    "First you split documents into chunks. Then you embed each chunk into a "
    "vector. At query time you embed the question and retrieve the nearest "
    "chunks by cosine similarity. The chunk size you pick is a tradeoff: too "
    "large and the embedding blurs many topics into one fuzzy vector; too small "
    "and each chunk lacks the context needed to be useful on its own."
)


@dataclass
class Chunk:
    """A chunk plus the provenance metadata every serious RAG pipeline carries."""

    text: str
    index: int
    start: int  # char offset into the source (inclusive)
    end: int    # char offset into the source (exclusive)
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # compact, demo-friendly
        preview = self.text.replace("\n", "\\n")
        if len(preview) > 48:
            preview = preview[:45] + "..."
        return f"Chunk(#{self.index} [{self.start}:{self.end}] {preview!r})"


# ---------------------------------------------------------------------------
# Variant A: fixed CHARACTER window.
# ---------------------------------------------------------------------------
def fixed_char_chunks(text: str, chunk_size: int = 120, overlap: int = 20) -> List[Chunk]:
    """Slide a `chunk_size`-character window with `overlap` chars of carry-over.

    stride = chunk_size - overlap. We guard overlap < chunk_size, else stride
    would be <= 0 and the loop would never advance (a classic infinite-loop bug
    in hand-rolled chunkers).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size (else stride<=0)")

    stride = chunk_size - overlap
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(Chunk(text=text[start:end], index=idx, start=start, end=end))
        idx += 1
        if end == n:
            break
        start += stride
    return chunks


# ---------------------------------------------------------------------------
# Variant B: fixed WORD window. Counts whitespace tokens instead of chars, so
# it never cuts *inside* a word — but it still cuts mid-sentence.
# ---------------------------------------------------------------------------
def fixed_word_chunks(text: str, words_per_chunk: int = 20, overlap_words: int = 4) -> List[Chunk]:
    """Window over whitespace-delimited words; map back to char offsets.

    We track char offsets by re-walking the original string so metadata stays
    honest (start/end point into the *source*, not into a re-joined copy).
    """
    if overlap_words >= words_per_chunk:
        raise ValueError("overlap_words must be smaller than words_per_chunk")

    # Tokenize while remembering each word's (start, end) char span.
    spans: List[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not text[j].isspace():
            j += 1
        spans.append((i, j))
        i = j

    stride = words_per_chunk - overlap_words
    chunks: List[Chunk] = []
    idx = 0
    w = 0
    while w < len(spans):
        window = spans[w : w + words_per_chunk]
        c_start = window[0][0]
        c_end = window[-1][1]
        chunks.append(Chunk(text=text[c_start:c_end], index=idx, start=c_start, end=c_end))
        idx += 1
        if w + words_per_chunk >= len(spans):
            break
        w += stride
    return chunks


def show_boundary_problems(chunks: List[Chunk]) -> None:
    """Highlight chunks that begin or end mid-word — the fixed-size failure."""
    print("  boundary audit (does a chunk start/end mid-word?):")
    for c in chunks:
        starts_mid = bool(c.text) and not c.text[0].isspace() and c.start > 0
        ends_mid = bool(c.text) and not c.text[-1].isspace()
        # A '*mid-word*' cut: the char just outside the chunk is alphanumeric.
        head_cut = c.start > 0 and c.text[:1].isalnum()
        tail_cut = ends_mid and c.text[-1:].isalnum()
        flag = "  <-- mid-word cut" if (head_cut or tail_cut) else ""
        print(f"    {c}{flag}")


def main() -> None:
    print("=" * 72)
    print("01 · FIXED-SIZE CHUNKING (the structure-blind baseline)")
    print("=" * 72)
    print(f"source length: {len(SAMPLE)} chars\n")

    print("[A] fixed CHARACTER chunks (size=120, overlap=20)")
    cc = fixed_char_chunks(SAMPLE, chunk_size=120, overlap=20)
    for c in cc:
        print("   ", c)
    show_boundary_problems(cc)

    # Demonstrate the overlap-duplication tax.
    emitted = sum(len(c.text) for c in cc)
    print(f"\n  overlap tax: source={len(SAMPLE)} chars, "
          f"emitted={emitted} chars across {len(cc)} chunks "
          f"({emitted / len(SAMPLE):.2f}x duplication).")

    print("\n[B] fixed WORD chunks (20 words, 4-word overlap)")
    wc = fixed_word_chunks(SAMPLE, words_per_chunk=20, overlap_words=4)
    for c in wc:
        print("   ", c)
    print("  note: word chunks never split a word, but still cut mid-SENTENCE.")

    print("\nTakeaway: fixed-size is O(n), dependency-free, and reproducible, but")
    print("it ignores structure. The next strategies add structure-awareness.")


if __name__ == "__main__":
    main()

"""03 — Token-based chunking (budget by *tokens*, not characters).

Why tokens and not characters? Because every downstream cost and limit is
measured in **tokens**, the sub-word units an LLM/embedding model actually
consumes:

  - An embedding model has a hard `max_seq_length` (e.g. all-MiniLM-L6-v2 = 256
    tokens, OpenAI text-embedding-3 = 8191 tokens, BGE-M3 = 8192). Feed it more
    and the extras are silently **truncated** — the tail of your chunk never
    reaches the vector. Sizing chunks in *characters* lets this happen by
    accident; sizing in *tokens* prevents it.
  - You pay per token. A token budget is the honest currency for cost planning.

The catch: tokenization is model-specific. GPT-4 uses tiktoken's `cl100k_base`/
`o200k_base` BPE; BERT-family models use WordPiece; T5 uses SentencePiece. The
*same* string is a different number of tokens under each. The pragmatic rule:
**chunk with the SAME tokenizer your embedding model uses.** If you cannot, use
a conservative estimator and leave headroom.

This file ships:
  1. A from-scratch whitespace+punctuation "word/punct" tokenizer (teaching).
  2. A character-ratio token *estimator* (the famous "~4 chars/token" / "~0.75
     words per token" heuristic for English) for when no real tokenizer is at
     hand.
  3. A real `tiktoken` path *if it is importable* — otherwise we fall back to
     the estimator. The file MUST exit 0 with no tiktoken and no network.

Heuristics (English, OpenAI's own rule of thumb,
https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them):
  - ~4 characters per token, OR
  - ~100 tokens ≈ 75 words  (=> tokens ≈ words / 0.75 ≈ words * 1.33).
These are *estimates*; real BPE merges punctuation, whitespace, and frequent
substrings unpredictably. Never trust an estimate near a hard model limit.

References:
  - tiktoken (OpenAI BPE tokenizer): https://github.com/openai/tiktoken
  - Hugging Face tokenizers: https://huggingface.co/docs/tokenizers/
  - LlamaIndex TokenTextSplitter:
    https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
  - LangChain TokenTextSplitter / split by tokens:
    https://python.langchain.com/docs/how_to/split_by_token/
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# A tiny, deterministic "word + punctuation" tokenizer (the teaching tokenizer).
# It splits runs of word characters and treats each punctuation mark as its own
# token — a crude but honest stand-in that captures the key property real
# tokenizers have: punctuation costs tokens too.
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    """Whitespace+punct tokenizer. Returns a list of token strings."""
    return _TOKEN_RE.findall(text)


def estimate_tokens_by_chars(text: str, chars_per_token: float = 4.0) -> int:
    """The '~4 chars per token' estimator. Conservative, model-agnostic."""
    return max(1, round(len(text) / chars_per_token))


def estimate_tokens_by_words(text: str, words_per_token: float = 0.75) -> int:
    """The '100 tokens ≈ 75 words' estimator => tokens ≈ words / 0.75."""
    words = len(text.split())
    return max(1, round(words / words_per_token))


# ---------------------------------------------------------------------------
# Pick the best available token counter. Prefer the REAL tiktoken BPE; degrade
# to the from-scratch estimator. We never import at module top level so a
# missing/old tiktoken can never break `import`.
# ---------------------------------------------------------------------------
def get_token_counter() -> Tuple[Callable[[str], int], str]:
    """Return (count_fn, backend_name). Tries tiktoken, else estimator."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 family

        def count(text: str) -> int:
            return len(enc.encode(text))

        return count, "tiktoken:cl100k_base"
    except Exception as exc:  # noqa: BLE001 — any failure -> estimator fallback
        print(f"[skip] tiktoken not usable ({type(exc).__name__}) — "
              "using char-ratio estimator. See README.")

        def count(text: str) -> int:
            return estimate_tokens_by_chars(text)

        return count, "estimator:4-chars-per-token"


@dataclass
class TokenChunk:
    text: str
    index: int
    n_tokens: int


def chunk_by_tokens(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = 32,
    overlap_tokens: int = 6,
) -> List[TokenChunk]:
    """Greedy token-budget packing of *whole words* with token overlap.

    Strategy: walk word-spaced units, accumulating until adding the next word
    would exceed `max_tokens`. Emit the window, then slide back so the last
    `overlap_tokens` (approx, measured in tokens) carry into the next window.

    We pack *words* (not raw tokens) so we never split inside a word — this is
    how LangChain/LlamaIndex token splitters behave in practice (they re-decode
    to text boundaries). The token COUNT, however, uses the real counter.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    words = text.split()
    chunks: List[TokenChunk] = []
    i = 0
    idx = 0
    n = len(words)
    while i < n:
        # Grow the window word-by-word until the token budget is hit.
        j = i
        cur = ""
        while j < n:
            candidate = (cur + " " + words[j]).strip()
            if count_tokens(candidate) > max_tokens and j > i:
                break
            cur = candidate
            j += 1
        chunk_tokens = count_tokens(cur)
        chunks.append(TokenChunk(text=cur, index=idx, n_tokens=chunk_tokens))
        idx += 1
        if j >= n:
            break
        # Slide back: keep popping words off the back of the just-emitted window
        # until the tail is ~overlap_tokens, and restart there.
        back = j
        tail = ""
        while back > i:
            cand = (words[back - 1] + " " + tail).strip()
            if count_tokens(cand) > overlap_tokens:
                break
            tail = cand
            back -= 1
        # Ensure forward progress (avoid infinite loop if a single word already
        # exceeds the overlap budget).
        i = back if back > i else j
    return chunks


SAMPLE = (
    "Token budgets matter because embedding models truncate silently. "
    "The all-MiniLM-L6-v2 model caps input at 256 tokens, so a chunk that looks "
    "fine in characters can lose its tail before it ever becomes a vector. "
    "Counting in tokens with the model's own tokenizer is the only safe way to "
    "stay under the limit while keeping chunks as large as the budget allows."
)


def main() -> None:
    print("=" * 72)
    print("03 · TOKEN-BASED CHUNKING (budget by tokens, not characters)")
    print("=" * 72)

    # Show the three counting methods on one string so the gap is visible.
    toks = simple_tokenize(SAMPLE)
    print(f"\nsource: {len(SAMPLE)} chars, {len(SAMPLE.split())} words, "
          f"{len(toks)} word/punct tokens (teaching tokenizer)")
    print(f"  estimate (4 chars/token):   ~{estimate_tokens_by_chars(SAMPLE)} tokens")
    print(f"  estimate (words / 0.75):    ~{estimate_tokens_by_words(SAMPLE)} tokens")

    count_tokens, backend = get_token_counter()
    print(f"  active backend: {backend}")
    print(f"  -> backend token count:     {count_tokens(SAMPLE)} tokens")

    print("\n[A] chunk_by_tokens(max_tokens=32, overlap_tokens=6)")
    chunks = chunk_by_tokens(SAMPLE, count_tokens, max_tokens=32, overlap_tokens=6)
    for c in chunks:
        preview = c.text if len(c.text) <= 70 else c.text[:67] + "..."
        print(f"  chunk {c.index}: {c.n_tokens} tok | {preview!r}")
    over = [c for c in chunks if c.n_tokens > 32]
    print(f"\n  chunks over budget: {len(over)} (should be 0 — the budget holds)")

    print("\nGotcha: the SAME text is a different token count under every")
    print("tokenizer. Always chunk with the tokenizer your model uses, and leave")
    print("headroom — estimators drift, especially with code, URLs, or CJK text.")


if __name__ == "__main__":
    main()

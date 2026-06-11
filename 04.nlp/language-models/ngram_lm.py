"""
n-gram Language Model
=====================
A classical (count-based) language model: estimate the probability of the next
token from the previous ``n-1`` tokens via the chain rule and maximum-likelihood
counts. Raw MLE assigns probability 0 to any unseen n-gram (catastrophic for
perplexity), so we add **smoothing**: Laplace/add-k, and a simplified
**Kneser-Ney** with absolute discounting + a continuation backoff. We measure
quality with **perplexity** and generate text by sampling.

Variants implemented here:
    - Add-k / Laplace smoothing (k=1 is classic Laplace)
    - Stupid-backoff (cheap, unnormalized, great for big corpora)
    - Simplified interpolated Kneser-Ney (absolute discounting + continuation)
    - Arbitrary order n (bigram, trigram, ...)

Training techniques demonstrated:
    - Smoothing as a remedy for zero-count sparsity
    - Held-out perplexity as the model-selection metric
    - Temperature-free multinomial sampling for generation

References:
    - Jurafsky & Martin, "Speech and Language Processing", ch. 3 (N-gram LMs)
    - Chen & Goodman (1998), "An Empirical Study of Smoothing Techniques"
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

SEED = 0
BOS = "<s>"   # beginning-of-sentence padding token
EOS = "</s>"  # end-of-sentence token (lets the model decide when to stop)


def _tokenize(text: str) -> list[str]:
    """Lowercase, keep words; split sentences on . ! ?"""
    import re
    sents = re.split(r"[.!?]+", text.lower())
    return [s.split() for s in sents if s.split()]


# ---------------------------------------------------------------------------
# 1. NumPy / pure-Python implementation (counts + smoothing, math explicit)
# ---------------------------------------------------------------------------
class NgramLM:
    r"""
    Count-based n-gram language model.

    **Chain rule.** Any sequence factorizes exactly as
        P(w_1..w_T) = \prod_t P(w_t | w_1..w_{t-1}).
    The **n-gram (Markov) assumption** truncates the history to the last n-1
    tokens:
        P(w_t | w_1..w_{t-1}) ≈ P(w_t | w_{t-n+1}..w_{t-1}).

    **MLE estimate** is just normalized counts:
        P(w | h) = count(h, w) / count(h).
    Unseen (h, w) -> 0, so we smooth. Modes:
      - "addk": (count(h,w)+k) / (count(h)+k|V|)
      - "backoff": stupid backoff, recursively shorten h with factor alpha
      - "kn": interpolated Kneser-Ney (absolute discount D + continuation prob)
    """

    def __init__(self, n=3, mode="kn", k=1.0, discount=0.75, alpha=0.4, seed=SEED):
        self.n = n
        self.mode = mode
        self.k = k
        self.D = discount
        self.alpha = alpha
        self.seed = seed

    # -- fitting: just collect counts of every order up to n -----------------
    def fit(self, sentences: list[list[str]]):
        self.vocab = {BOS, EOS}
        for s in sentences:
            self.vocab.update(s)
        self.V = len(self.vocab)
        # ngrams[m] maps a (context tuple of length m-1) -> Counter(next token)
        self.ngrams = [defaultdict(Counter) for _ in range(self.n + 1)]
        # Kneser-Ney continuation stats are defined at the BIGRAM level:
        #   N1+(.,w) = number of distinct words that precede w (as a bigram),
        #   N1+(.,.) = number of distinct bigram types in the corpus.
        self.cont_before = defaultdict(set)  # w -> {preceding words}
        bigram_types = set()
        for s in sentences:
            padded = [BOS] * (self.n - 1) + s + [EOS]
            for m in range(1, self.n + 1):
                for i in range(len(padded) - m + 1):
                    gram = tuple(padded[i:i + m])
                    ctx, w = gram[:-1], gram[-1]
                    self.ngrams[m][ctx][w] += 1
            # bigram-level continuation counts (independent of model order n)
            bpad = [BOS] + s + [EOS]
            for a, b in zip(bpad, bpad[1:]):
                self.cont_before[b].add(a)
                bigram_types.add((a, b))
        self.total_cont = max(len(bigram_types), 1)  # N1+(.,.)
        return self

    # -- probability of one token given its (already-truncated) history ------
    def prob(self, word: str, context: tuple) -> float:
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        if self.mode == "addk":
            return self._prob_addk(word, context)
        if self.mode == "backoff":
            return self._prob_backoff(word, context)
        if self.mode == "kn":
            return self._prob_kn(word, context, self.n)
        raise ValueError(self.mode)

    def _prob_addk(self, word, context):
        # P(w|h) = (c(h,w)+k) / (c(h)+k|V|)
        counter = self.ngrams[len(context) + 1].get(context, Counter())
        num = counter.get(word, 0) + self.k
        den = sum(counter.values()) + self.k * self.V
        return num / den

    def _prob_backoff(self, word, context):
        # Stupid backoff (Brants 2007): use MLE if seen, else alpha * back off.
        # NOT a true distribution (doesn't sum to 1) but excellent in practice.
        counter = self.ngrams[len(context) + 1].get(context, Counter())
        c_hw = counter.get(word, 0)
        if c_hw > 0:
            return c_hw / sum(counter.values())
        if not context:
            # unigram floor so we never return exactly 0
            uni = self.ngrams[1][()]
            return self.alpha * (uni.get(word, 0) + 1) / (sum(uni.values()) + self.V)
        return self.alpha * self._prob_backoff(word, context[1:])

    def _prob_kn(self, word, context, order):
        r"""
        Interpolated Kneser-Ney with a single discount D:

            P_KN(w|h) = max(c(h,w)-D, 0)/c(h)  +  lambda(h) * P_KN(w|h')

        where lambda(h) = D * N1+(h, .) / c(h) is the leftover mass, and the
        lowest-order term uses the **continuation probability**
            P_cont(w) = N1+(. , w) / N1+(. , .)
        i.e. how many *distinct* contexts w follows, not how often it occurs.
        That is the Kneser-Ney insight: "Francisco" is frequent but only after
        "San", so its continuation probability is low.
        """
        if order == 1:
            # continuation probability + tiny floor so unseen words aren't 0
            n_w = len(self.cont_before.get(word, ()))
            return (n_w + 1.0) / (self.total_cont + self.V)
        counter = self.ngrams[order].get(context, Counter())
        c_h = sum(counter.values())
        if c_h == 0:
            return self._prob_kn(word, context[1:], order - 1)
        c_hw = counter.get(word, 0)
        first = max(c_hw - self.D, 0.0) / c_h
        n1 = len(counter)  # number of distinct words seen after this context
        lam = self.D * n1 / c_h
        return first + lam * self._prob_kn(word, context[1:], order - 1)

    # -- sentence / corpus log-probability and perplexity --------------------
    def log_prob_sentence(self, s: list[str]) -> tuple[float, int]:
        padded = [BOS] * (self.n - 1) + s + [EOS]
        logp, count = 0.0, 0
        for i in range(self.n - 1, len(padded)):
            ctx = tuple(padded[i - self.n + 1:i])
            p = self.prob(padded[i], ctx)
            logp += math.log(max(p, 1e-12))
            count += 1
        return logp, count

    def perplexity(self, sentences: list[list[str]]) -> float:
        # PP = exp(-(1/N) * sum log P(w_i | history))
        total_lp, total_n = 0.0, 0
        for s in sentences:
            lp, n = self.log_prob_sentence(s)
            total_lp += lp
            total_n += n
        return math.exp(-total_lp / max(total_n, 1))

    # -- generation: sample next token from the smoothed distribution --------
    def generate(self, max_len=20) -> list[str]:
        rng = np.random.default_rng(self.seed)
        vocab = sorted(self.vocab - {BOS})
        history = [BOS] * (self.n - 1)
        out = []
        for _ in range(max_len):
            ctx = tuple(history[-(self.n - 1):]) if self.n > 1 else ()
            probs = np.array([self.prob(w, ctx) for w in vocab], float)
            probs = probs / probs.sum()  # renormalize (backoff isn't normalized)
            w = vocab[rng.choice(len(vocab), p=probs)]
            if w == EOS:
                break
            out.append(w)
            history.append(w)
        return out


# ---------------------------------------------------------------------------
# 2. Reference: nothing to "train" in PyTorch for a count model.
#    The neural counterpart lives in nlp/language-models/neural_lm.py.
#    Here we expose a tiny helper to compare smoothing methods.
# ---------------------------------------------------------------------------
def compare_smoothing(train, test, n=3):
    """Return {mode: perplexity} on held-out text for several smoothers."""
    results = {}
    for mode in ("addk", "backoff", "kn"):
        lm = NgramLM(n=n, mode=mode).fit(train)
        results[mode] = lm.perplexity(test)
    return results


# ---------------------------------------------------------------------------
# 3. Demo — a tiny built-in toy corpus
# ---------------------------------------------------------------------------
def toy_corpus() -> str:
    return (
        "the cat sat on the mat. "
        "the dog sat on the log. "
        "the cat chased the dog. "
        "the dog chased the cat around the mat. "
        "a happy cat naps on the warm mat. "
        "a happy dog runs around the green park. "
        "the cat and the dog are friends. "
        "the friendly dog naps on the log. "
        "cats and dogs sat on the mat together. "
        "the warm mat is where the cat naps."
    )


def demo():
    np.random.seed(SEED)
    sents = _tokenize(toy_corpus())
    train, test = sents[:8], sents[8:]

    print("=== Perplexity by smoothing method (trigram) ===")
    for mode, pp in compare_smoothing(train, test, n=3).items():
        print(f"  {mode:8s} test perplexity = {pp:7.3f}")

    print("\n=== Effect of add-k (k) on a bigram LM ===")
    for k in (0.01, 0.1, 1.0):
        lm = NgramLM(n=2, mode="addk", k=k).fit(train)
        print(f"  k={k:<4}  perplexity = {lm.perplexity(test):7.3f}")

    print("\n=== Generated sentences (Kneser-Ney trigram) ===")
    lm = NgramLM(n=3, mode="kn").fit(sents)
    for s in range(3):
        lm.seed = s
        print("  " + " ".join(lm.generate(max_len=12)))

    # sanity: probabilities given a context are a valid distribution (KN)
    lm = NgramLM(n=2, mode="kn").fit(sents)
    vocab = sorted(lm.vocab - {BOS})
    total = sum(lm.prob(w, ("the",)) for w in vocab)
    print(f"\nKN bigram P(.|'the') sums to {total:.4f} (should be ~1)")


if __name__ == "__main__":
    demo()

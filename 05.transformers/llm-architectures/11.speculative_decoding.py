"""
Speculative Decoding — Draft-and-Verify
=======================================
Autoregressive decoding is sequential: one forward pass of the big TARGET model
per token. Speculative decoding accelerates it without changing the output
distribution: a small, cheap DRAFT model proposes several tokens at once, then a
SINGLE target forward pass verifies them in parallel. Accepted tokens are free
(the target would have produced them anyway); the first rejection is corrected
and decoding resumes.

For greedy decoding the verification rule is simple: run the target over
[prompt + draft], and at each drafted position check whether the target's argmax
equals the drafted token. Accept the longest matching prefix; at the first
mismatch, replace with the target's own token and stop that round. Because the
target processes all K draft tokens in ONE pass (parallel over positions thanks to
causal attention), you get up to K+1 tokens per target call instead of 1.

    1. draft proposes  d_1..d_K  (K cheap autoregressive steps)
    2. target forward over prompt+draft -> its argmax t_1..t_K (one pass)
    3. accept d_i while d_i == t_i; on first mismatch emit t_i and break;
       if all K accepted, also emit the target's (K+1)-th "bonus" token.

KEY GUARANTEE: the emitted sequence is EXACTLY what greedy target-only decoding
would produce — speculative decoding is an exact accelerator, not an
approximation. (For sampling, the original paper uses a modified rejection-
sampling acceptance test that preserves the target's distribution.) Speed-up
depends on the acceptance rate, which depends on how well the draft mimics the
target.

What introduced it: Leviathan et al. (2023) and Chen et al. (2023), independently.
Used in production LLM serving; "self-speculation" / Medusa / EAGLE are variants
that draft with extra heads instead of a separate model.

References:
    - Leviathan, Kalman & Matias (2023), "Fast Inference from Transformers via
      Speculative Decoding"
    - Chen et al. (2023), "Accelerating Large Language Model Decoding with
      Speculative Sampling"
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyLM(nn.Module):
    """A toy causal LM: embedding -> 1 attention-ish block -> logits."""

    def __init__(self, vocab, d=32, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, d)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, vocab, bias=False)
        self.d = d

    def forward(self, ids):                          # ids: (B, L)
        x = self.emb(ids)
        L = ids.size(1)
        s = self.q(x) @ self.k(x).transpose(-1, -2) / self.d ** 0.5
        mask = torch.triu(torch.ones(L, L), 1).bool()
        s = s.masked_fill(mask, float("-inf"))
        h = s.softmax(-1) @ self.v(x)
        return self.out(h)                           # (B, L, vocab)


def greedy_target_only(target, prompt, n_new):
    """Plain greedy decoding with the target model — the reference output."""
    ids = prompt.clone()
    for _ in range(n_new):
        nxt = target(ids)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
    return ids


def speculative_greedy(target, draft, prompt, n_new, K=4):
    """
    Draft-and-verify greedy decoding. Returns (ids, n_target_calls, n_accepted).
    Output is identical to greedy_target_only.
    """
    ids = prompt.clone()
    target_calls = 0
    accepted_total = 0
    produced = 0
    while produced < n_new:
        # 1. draft K tokens autoregressively (cheap model)
        draft_ids = ids.clone()
        proposals = []
        for _ in range(K):
            d = draft(draft_ids)[:, -1].argmax(-1, keepdim=True)
            draft_ids = torch.cat([draft_ids, d], 1)
            proposals.append(d.item())

        # 2. ONE target pass over prompt + all K proposals
        t_logits = target(draft_ids)                 # (1, L+K, V)
        target_calls += 1
        base = ids.size(1)
        # target's greedy prediction AT each drafted position
        t_preds = t_logits[0, base - 1: base - 1 + K].argmax(-1)  # (K,)

        # 3. accept matching prefix; correct first mismatch
        n_acc = 0
        for i in range(K):
            if produced >= n_new:
                break
            if proposals[i] == t_preds[i].item():
                ids = torch.cat([ids, t_preds[i:i + 1].view(1, 1)], 1)
                n_acc += 1; produced += 1
            else:
                ids = torch.cat([ids, t_preds[i:i + 1].view(1, 1)], 1)
                produced += 1
                break                                # stop at first rejection
        else:
            # all K accepted -> bonus token from the target's next prediction
            if produced < n_new:
                bonus = t_logits[0, base - 1 + K].argmax(-1).view(1, 1)
                ids = torch.cat([ids, bonus], 1)
                produced += 1
        accepted_total += n_acc
    return ids[:, :prompt.size(1) + n_new], target_calls, accepted_total


# ---------------------------------------------------------------------------
# Demo — acceptance + identical output to target-only greedy
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    vocab, n_new, K = 20, 16, 4
    target = TinyLM(vocab, d=32, seed=1).eval()
    # Draft shares the target's weights here so acceptance is high (a realistic
    # draft is a smaller distilled model; same-weights makes the demo crisp).
    draft = target

    prompt = torch.randint(0, vocab, (1, 4))
    with torch.no_grad():
        ref = greedy_target_only(target, prompt, n_new)
        spec, calls, accepted = speculative_greedy(target, draft, prompt, n_new, K)

    print(f"prompt           : {prompt[0].tolist()}")
    print(f"target-only greedy: {ref[0, 4:].tolist()}")
    print(f"speculative      : {spec[0, 4:].tolist()}")
    assert torch.equal(ref, spec), "speculative must equal target-only greedy"
    print("Identical output to target-only greedy.  PASS")

    print(f"\ngenerated {n_new} tokens using {calls} target forward passes "
          f"(naive would need {n_new})")
    print(f"draft tokens accepted: {accepted}/{calls*K}  "
          f"-> acceptance rate {accepted/(calls*K):.0%}")
    print(f"speed-up factor (tokens / target-calls) = {n_new/calls:.2f}x")

    # A WEAKER draft (different weights) accepts less -> smaller speed-up.
    weak = TinyLM(vocab, d=32, seed=99).eval()
    with torch.no_grad():
        spec2, calls2, acc2 = speculative_greedy(target, weak, prompt, n_new, K)
    assert torch.equal(ref, spec2)                   # STILL exact
    print(f"\nweaker draft: still exact output, but acceptance "
          f"{acc2/(calls2*K):.0%}, speed-up {n_new/calls2:.2f}x "
          f"(quality of draft drives the gain)")


if __name__ == "__main__":
    demo()

"""
MoE — Mixture-of-Experts Feed-Forward Layer
===========================================
A dense FFN applies the SAME large matrices to every token. An MoE layer instead
holds many smaller "expert" FFNs and a lightweight router that, per token, picks
the top-k experts to run. Only k of E experts fire, so the model has a huge
*parameter* count but a small *active* compute per token — the recipe behind
sparse giants (Switch-T, Mixtral, DeepSeek-V3, Grok).

    router logits  g = x · W_gate                       (E logits per token)
    pick top-k experts; softmax over the chosen k -> weights p
    y = Σ_{i in topk}  p_i · Expert_i(x)

LOAD BALANCING. Left alone, the router collapses onto a few favourite experts
(others get no gradient -> "dead experts"). Switch Transformer adds an auxiliary
loss that pushes the routing toward uniform usage:

    aux = α · E · Σ_i  f_i · P_i
        f_i = fraction of tokens dispatched to expert i   (hard count)
        P_i = mean router probability mass on expert i    (soft, differentiable)

It is minimized when both are uniform (1/E), so it spreads load. α≈0.01.

CAPACITY. To make batched dispatch rectangular, each expert has a CAPACITY =
ceil(capacity_factor · tokens · k / E). Tokens beyond an expert's capacity are
DROPPED (they bypass via the residual). capacity_factor>1 leaves slack.

SHARED EXPERT (DeepSeek). Always-on expert(s) every token uses, capturing common
patterns so the routed experts can specialize. y = SharedFFN(x) + MoE_routed(x).

What introduced it: Shazeer et al. (2017, sparsely-gated MoE); Switch Transformer
(Fedus et al. 2021, top-1 + aux loss); GShard (top-2); Mixtral 8x7B (top-2 of 8);
DeepSeekMoE (fine-grained + shared experts).

References:
    - Shazeer et al. (2017), "Outrageously Large Neural Networks: The
      Sparsely-Gated Mixture-of-Experts Layer"
    - Fedus, Zoph & Shazeer (2021), "Switch Transformers"
    - Lepikhin et al. (2020), "GShard"
    - Dai et al. (2024), "DeepSeekMoE: Towards Ultimate Expert Specialization"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A small SwiGLU-ish FFN used as one expert."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class MoE(nn.Module):
    """
    Top-k MoE FFN with load-balancing aux loss, capacity dropping, and an
    optional always-on shared expert (DeepSeek style).
    """

    def __init__(self, d_model, d_ff, n_experts=8, top_k=2,
                 capacity_factor=1.25, aux_alpha=0.01, n_shared=0):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.cap_factor, self.aux_alpha = capacity_factor, aux_alpha
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_shared))

    def forward(self, x):
        B, L, D = x.shape
        xf = x.reshape(-1, D)                          # (T, D)
        T = xf.size(0)

        logits = self.gate(xf)                         # (T, E)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)        # (T, k)
        topv = topv / topv.sum(-1, keepdim=True)       # renormalize over chosen k

        # --- load-balancing aux loss (Switch) ---
        # f_i: fraction of (token,slot) assignments going to expert i
        one_hot = F.one_hot(topi, self.E).float().sum(1)        # (T, E) counts/token
        f = one_hot.mean(0) * (self.E / self.k)                 # hard load
        P = probs.mean(0)                                       # soft prob mass
        aux_loss = self.aux_alpha * self.E * (f * P).sum()

        # --- capacity ---
        capacity = int(self.cap_factor * T * self.k / self.E)
        capacity = max(capacity, 1)

        out = torch.zeros_like(xf)
        util = torch.zeros(self.E)
        dropped = 0
        for e in range(self.E):
            # which (token, slot) picked expert e
            sel = (topi == e)                                   # (T, k) bool
            tok_idx, slot_idx = sel.nonzero(as_tuple=True)
            if tok_idx.numel() == 0:
                continue
            if tok_idx.numel() > capacity:                      # drop overflow
                dropped += tok_idx.numel() - capacity
                tok_idx = tok_idx[:capacity]
                slot_idx = slot_idx[:capacity]
            util[e] = tok_idx.numel()
            w = topv[tok_idx, slot_idx].unsqueeze(-1)           # gate weights
            out[tok_idx] += w * self.experts[e](xf[tok_idx])

        # shared experts: every token, always
        for sh in self.shared:
            out += sh(xf)

        self.last_util = util
        self.last_dropped = dropped
        return out.reshape(B, L, D), aux_loss


# ---------------------------------------------------------------------------
# Demo — route tokens, show utilization + aux loss
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    d_model, d_ff, E, k = 32, 64, 8, 2
    moe = MoE(d_model, d_ff, n_experts=E, top_k=k, n_shared=1)
    x = torch.randn(4, 16, d_model)            # 64 tokens

    y, aux = moe(x)
    print(f"MoE: in {tuple(x.shape)} -> out {tuple(y.shape)}, top-{k} of {E} experts")
    assert y.shape == x.shape

    util = moe.last_util
    print(f"\nExpert utilization (tokens routed per expert):")
    print("  ", util.int().tolist(), f"  total={int(util.sum())}, "
          f"dropped={moe.last_dropped}")
    print(f"aux load-balance loss = {aux.item():.4f}  "
          f"(minimized at uniform 1/{E} usage)")
    assert util.sum() > 0

    # Training nudges balance: optimize the aux loss alone for a few steps and
    # watch the utilization spread out (coefficient of variation drops).
    def cv(u): return (u.std() / (u.mean() + 1e-9)).item()
    cv0 = cv(util)
    opt = torch.optim.Adam(moe.gate.parameters(), lr=0.1)
    for _ in range(50):
        _, a = moe(x)
        opt.zero_grad(); a.backward(); opt.step()
    _, _ = moe(x)
    cv1 = cv(moe.last_util)
    print(f"\nutilization CV before={cv0:.2f}  after aux-loss training={cv1:.2f}"
          f"  (lower = more balanced)")
    print("Top-k routing + aux loss + capacity + shared expert all run.  PASS")


if __name__ == "__main__":
    demo()

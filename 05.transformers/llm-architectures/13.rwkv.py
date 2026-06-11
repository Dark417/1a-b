"""
RWKV — Linear-Attention RNN/Transformer Hybrid
==============================================
RWKV ("Receptance Weighted Key Value") is a sequence model that TRAINS like a
transformer (parallelizable, attention-free) but RUNS like an RNN (O(1) state per
step, O(L) total, no growing KV cache). It replaces dot-product attention with a
LINEAR-attention WKV recurrence built from four learned projections:

    R (receptance) — a gate deciding how much of the aggregated context to read,
    W (time-decay) — how fast past keys fade (a per-channel decay),
    K (key), V (value).

The WKV operator is a decayed, softmax-free weighted sum of past values. In its
recurrent form it carries two running accumulators (a numerator and a
denominator) per channel:

    wkv_t = (a_{t-1} + exp(u + k_t) · v_t) / (b_{t-1} + exp(u + k_t))
    a_t   = exp(-w) · a_{t-1} + exp(k_t) · v_t          # decayed value sum
    b_t   = exp(-w) · b_{t-1} + exp(k_t)                # decayed key sum

where w = exp(W) is the per-channel decay rate and u ("bonus") gives the CURRENT
token extra weight. The output is gated by the receptance: y_t = σ(R) ⊙ wkv_t,
then projected. Because a_t, b_t are fixed-size and only depend on the previous
step, inference is a constant-memory RNN — yet the same computation can be
unrolled/parallelized for training.

This is the "time-mixing" block; RWKV also has a "channel-mixing" block (a gated
FFN with a token-shift). Newer RWKV-5/6 ("Eagle"/"Finch") make W input-dependent
(matrix-valued state), closing much of the gap to transformers.

What introduced it: Peng et al. (2023), an open community project (BlinkDL).

References:
    - Peng et al. (2023), "RWKV: Reinventing RNNs for the Transformer Era"
    - Peng et al. (2024), "Eagle and Finch: RWKV with Matrix-Valued States" (v5/v6)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def token_shift(x):
    """Shift the sequence by one (pad a zero at the front). Cheap 'mini-conv'
    that lets each token mix with its immediate predecessor (RWKV uses this)."""
    return F.pad(x, (0, 0, 1, -1))                   # (B, L, d) shifted right by 1


class RWKVTimeMix(nn.Module):
    """RWKV time-mixing: the WKV linear-attention recurrence."""

    def __init__(self, d_model):
        super().__init__()
        self.R = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.O = nn.Linear(d_model, d_model, bias=False)
        # per-channel time-decay W and current-token bonus u
        self.time_decay = nn.Parameter(torch.zeros(d_model))   # w = exp(.)
        self.time_bonus = nn.Parameter(torch.zeros(d_model))   # u
        # learned mix between current token and the shifted (previous) token
        self.mix_r = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_k = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(d_model) * 0.5)

    def forward(self, x):
        B, L, D = x.shape
        xs = token_shift(x)
        # token-shift interpolation: blend current and previous token per channel
        r = self.R(x * self.mix_r + xs * (1 - self.mix_r))
        k = self.K(x * self.mix_k + xs * (1 - self.mix_k))
        v = self.V(x * self.mix_v + xs * (1 - self.mix_v))
        sr = torch.sigmoid(r)                          # receptance gate

        w = torch.exp(self.time_decay)                 # decay rate > 0  (per channel)
        u = self.time_bonus

        # WKV recurrence (numerically stable softmax-free linear attention)
        a = torch.zeros(B, D)                          # running decayed value sum
        b = torch.zeros(B, D)                          # running decayed key sum
        out = []
        for t in range(L):
            kt, vt = k[:, t], v[:, t]
            # current token gets a 'bonus' weight exp(u+k); past is in a,b
            ek = torch.exp(u + kt)
            wkv = (a + ek * vt) / (b + ek + 1e-8)
            out.append(wkv)
            # decay the accumulators and fold in the current token
            ek2 = torch.exp(kt)
            a = torch.exp(-w) * a + ek2 * vt
            b = torch.exp(-w) * b + ek2
        wkv = torch.stack(out, dim=1)                  # (B, L, D)
        return self.O(sr * wkv)                        # gated output


class RWKVChannelMix(nn.Module):
    """RWKV channel-mixing: a gated FFN with token-shift (squared-ReLU gate)."""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.K = nn.Linear(d_model, d_ff, bias=False)
        self.R = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_ff, d_model, bias=False)
        self.mix_k = nn.Parameter(torch.ones(d_model) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(d_model) * 0.5)

    def forward(self, x):
        xs = token_shift(x)
        k = self.K(x * self.mix_k + xs * (1 - self.mix_k))
        r = self.R(x * self.mix_r + xs * (1 - self.mix_r))
        kv = self.V(torch.relu(k) ** 2)                # squared-ReLU (RWKV gate)
        return torch.sigmoid(r) * kv


class RWKVBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.tmix = RWKVTimeMix(d_model)
        self.cmix = RWKVChannelMix(d_model)

    def forward(self, x):
        x = x + self.tmix(self.ln1(x))
        x = x + self.cmix(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Demo — forward on a tiny sequence
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    d_model, L = 16, 8
    block = RWKVBlock(d_model)
    x = torch.randn(2, L, d_model)

    y = block(x)
    print(f"RWKVBlock: in {tuple(x.shape)} -> out {tuple(y.shape)}")
    assert y.shape == x.shape

    # The WKV recurrence is causal: future tokens cannot affect past outputs
    # (a,b accumulators only flow forward in time).
    tmix = block.tmix
    h = block.ln1(x)
    y_full = tmix(h)
    h2 = h.clone(); h2[:, -1] += 10.0
    y_pert = tmix(h2)
    early = (y_full[:, :-1] - y_pert[:, :-1]).abs().max().item()
    print(f"\nperturbing last token changes earlier outputs by {early:.2e} "
          f"(~0 -> causal WKV recurrence)")
    assert early < 1e-4

    # Constant state: the recurrence carries just (a, b) of size d_model each,
    # independent of sequence length -> O(1) memory inference, like an RNN.
    print(f"\nrecurrent state = 2 x d_model = {2*d_model} values "
          f"(independent of L; no growing KV cache)")
    print("attention-free linear-time sequence mixing.  PASS")


if __name__ == "__main__":
    demo()

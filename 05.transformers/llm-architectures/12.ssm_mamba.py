"""
State-Space Models / Mamba — Selective Scan
===========================================
Attention is O(L²) and needs a growing KV cache. State-space models (SSMs) offer
an alternative sequence mixer that is O(L) in time and O(1) in inference memory:
a linear recurrence with a fixed-size hidden state h, like an RNN but derived from
continuous-time linear systems and made parallelizable.

The continuous SSM is  h'(t) = A h(t) + B x(t),  y(t) = C h(t).  Discretized with
step Δ (zero-order hold) it becomes a linear recurrence:

    h_t = Ā h_{t-1} + B̄ x_t           Ā = exp(Δ A),  B̄ = (ΔA)^{-1}(exp(ΔA)-I)·ΔB
    y_t = C h_t  (+ D x_t)

Classic SSMs (S4) keep A, B, C, Δ FIXED across time — content-independent, so they
can't selectively remember/forget based on the input (they fail at tasks like
"copy this token"). MAMBA's insight (S6 / "selective" SSM): make B, C, and Δ
FUNCTIONS OF THE INPUT x_t. Now the recurrence can choose, per token, what to
write into and read from the state — content-based gating, like attention, but
still O(L). The price: A is no longer time-invariant, so the fast FFT/conv
parallelization of S4 doesn't apply; Mamba uses a hardware-aware parallel SCAN
instead (a sequential loop here, which is the clearest teaching form).

    Δ_t, B_t, C_t = linear(x_t)        # input-dependent (selective)
    Ā_t = exp(Δ_t · A)                 # per-step state transition
    h_t = Ā_t ⊙ h_{t-1} + (Δ_t·B_t) x_t
    y_t = C_t · h_t + D x_t

What introduced it: S4 (Gu et al. 2021); Mamba / selective SSM (Gu & Dao 2023);
Mamba-2 (2024) recasts it as a form of linear attention.

References:
    - Gu, Goel & Ré (2021), "Efficiently Modeling Long Sequences with Structured
      State Spaces" (S4)
    - Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State
      Spaces"
    - Dao & Gu (2024), "Transformers are SSMs" (Mamba-2)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """
    Minimal selective SSM (S6 core). Per-channel state of size d_state, with
    input-dependent Δ, B, C. Sequential scan (O(L)); clear over fast.
    Shapes: x (B, L, d_model).
    """

    def __init__(self, d_model, d_state=8, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # short causal depthwise conv (local mixing before the scan, as in Mamba)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                              groups=self.d_inner, padding=d_conv - 1, bias=True)
        # input-dependent Δ, B, C projections
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        # A is a learned (log) per-(channel,state) decay; D is a skip.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1)
                                            .float().repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)                          # (B, L, 2*d_inner)
        xin, z = xz.chunk(2, dim=-1)                  # x path and gate path

        # causal depthwise conv over time
        xc = self.conv(xin.transpose(1, 2))[..., :L].transpose(1, 2)
        xc = F.silu(xc)                               # (B, L, d_inner)

        # input-dependent parameters
        dbl = self.x_proj(xc)                         # (B, L, 2*d_state+1)
        dt, Bm, Cm = dbl[..., :1], dbl[..., 1:1 + self.d_state], dbl[..., 1 + self.d_state:]
        dt = F.softplus(self.dt_proj(dt))             # (B, L, d_inner) > 0
        A = -torch.exp(self.A_log)                    # (d_inner, d_state), negative

        # selective scan (sequential recurrence — O(L))
        h = torch.zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]                           # (B, d_inner)
            # discretize: Ā = exp(Δ A), B̄ x = Δ·B·x  (simplified ZOH)
            Abar = torch.exp(dt_t.unsqueeze(-1) * A)  # (B, d_inner, d_state)
            Bx = (dt_t.unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * xc[:, t].unsqueeze(-1)
            h = Abar * h + Bx                         # state update
            y = (h * Cm[:, t].unsqueeze(1)).sum(-1)   # (B, d_inner)  y = C h
            ys.append(y)
        y = torch.stack(ys, dim=1)                    # (B, L, d_inner)
        y = y + xc * self.D                           # skip connection D x

        y = y * F.silu(z)                             # gated output (Mamba)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Demo — run on a tiny sequence; show O(L) recurrence
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    d_model, L = 16, 10
    block = MambaBlock(d_model, d_state=8)
    x = torch.randn(2, L, d_model)

    y = block(x)
    print(f"MambaBlock: in {tuple(x.shape)} -> out {tuple(y.shape)}")
    assert y.shape == x.shape

    # Causality check: the recurrence is strictly causal -> changing a LATE token
    # must not affect EARLIER outputs (a property attention's causal mask shares,
    # but here it comes for free from the left-to-right scan).
    x2 = x.clone()
    x2[:, -1] += 5.0                                  # perturb only the last token
    y2 = block(x2)
    early_diff = (y[:, :-1] - y2[:, :-1]).abs().max().item()
    print(f"\nperturbing last token changes earlier outputs by {early_diff:.2e} "
          f"(~0 -> strictly causal recurrence)")
    assert early_diff < 1e-5

    # O(L) scaling: cost is linear in L (one fixed-size state update per step),
    # vs attention's O(L^2). State memory is O(d_state), independent of L.
    print(f"\nstate size = d_inner({block.d_inner}) x d_state({block.d_state}) "
          f"= {block.d_inner*block.d_state} (independent of sequence length)")
    print("compute scales O(L) (one step per token), memory O(1) in L.  PASS")


if __name__ == "__main__":
    demo()

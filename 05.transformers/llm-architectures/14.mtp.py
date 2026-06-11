"""
MTP — Multi-Token Prediction (DeepSeek-V3)
==========================================
Standard language models are trained with NEXT-token prediction: at each position
predict token t+1. Multi-token prediction (MTP) trains the model to predict
SEVERAL future tokens (t+1, t+2, ...) at once, from each position. This is a
richer training signal — the representation must encode information about more of
the future — and it improves data efficiency. As a bonus, the extra prediction
heads can be reused at inference for self-speculative decoding (draft the next few
tokens cheaply, then verify).

Two flavours:
    - Meta's MTP (Gloeckle et al. 2024): n INDEPENDENT output heads sharing the
      trunk; head i predicts token t+i directly and in parallel. Simple, parallel,
      but each head predicts t+i without seeing the predicted t+i-1.
    - DeepSeek-V3's MTP: SEQUENTIAL / causal MTP modules. Module k takes the
      trunk hidden state AND the embedding of the (already known) token at
      depth k-1, runs a small transformer block, and predicts the (k+1)-th token.
      This keeps the full causal chain at each depth:
          h^k_i = TRBlock_k( [ RMSNorm(h^{k-1}_i) ; RMSNorm(emb(x_{i+k})) ] · W )
          logits^k = SharedHead( h^k_i )   -> predicts x_{i+k+1}
      The shared embedding/head ties parameters with the main model.

Training loss is the average of the per-depth next-token cross-entropies (each
depth shifts the target further into the future), weighted by λ.

At inference the main head is used for normal generation; the MTP heads provide
extra draft tokens for speculative decoding (DeepSeek-V3 reports ~1.8x token
acceptance for the next-token MTP module).

What introduced it: Gloeckle et al. (2024) for the multi-head idea; DeepSeek-V3
(2024) for the sequential causal MTP used in a frontier model.

References:
    - Gloeckle et al. (2024), "Better & Faster Large Language Models via
      Multi-token Prediction"
    - DeepSeek-AI (2024), "DeepSeek-V3 Technical Report"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrunkBlock(nn.Module):
    """A tiny causal transformer trunk producing per-position hidden states."""

    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, -1)
        shp = lambda t: t.view(B, L, self.h, self.d_k).transpose(1, 2)
        q, k, v = shp(q), shp(k), shp(v)
        s = q @ k.transpose(-1, -2) / self.d_k ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(L, L), 1).bool(), float("-inf"))
        a = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1)
        x = x + self.o(a)
        return x + self.ff(x)


class MTPModule(nn.Module):
    """One sequential DeepSeek-style MTP depth: combine trunk hidden + next-token
    embedding, run a small block, predict one step further ahead."""

    def __init__(self, d_model):
        super().__init__()
        self.norm_h = nn.LayerNorm(d_model)
        self.norm_e = nn.LayerNorm(d_model)
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.block = TrunkBlock(d_model)

    def forward(self, h, tok_emb):
        # combine the previous-depth hidden with the embedding of the known token
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(tok_emb)], dim=-1))
        return self.block(x)


class MTPModel(nn.Module):
    """
    LM trunk + shared embedding/head + n_predict sequential MTP modules.
    Predicts tokens t+1 .. t+n_predict from each position.
    """

    def __init__(self, vocab, d_model=32, n_predict=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.trunk = TrunkBlock(d_model)
        self.mtp = nn.ModuleList(MTPModule(d_model) for _ in range(n_predict - 1))
        self.head = nn.Linear(d_model, vocab, bias=False)   # shared output head
        self.n_predict = n_predict

    def forward(self, ids):
        """Returns a list of logits, one per prediction depth (t+1, t+2, ...)."""
        h = self.trunk(self.emb(ids))                 # depth-0 hidden
        logits = [self.head(h)]                        # predicts t+1
        for k, mod in enumerate(self.mtp, start=1):
            # feed the embedding of the token k ahead (teacher-forced at train)
            shifted = F.pad(self.emb(ids), (0, 0, -k, k))  # emb(x_{i+k})
            h = mod(h, shifted)
            logits.append(self.head(h))                # predicts t+1+k
        return logits

    def loss(self, ids):
        """Average next-k cross-entropy over all depths."""
        logits = self(ids)
        total = 0.0
        for k, lg in enumerate(logits, start=1):       # depth k predicts t+k
            pred = lg[:, :-k]                          # positions with a target
            tgt = ids[:, k:]
            total = total + F.cross_entropy(pred.reshape(-1, lg.size(-1)),
                                            tgt.reshape(-1))
        return total / len(logits), logits


# ---------------------------------------------------------------------------
# Demo — predict next-2 tokens on toy data
# ---------------------------------------------------------------------------
def demo():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)

    vocab, L = 12, 10
    model = MTPModel(vocab, d_model=32, n_predict=2)

    # toy task: a deterministic sequence x_{t} = (x_{t-1} + 1) mod vocab, so the
    # model CAN learn to predict 2 steps ahead.
    seq = torch.arange(L).remainder(vocab).unsqueeze(0).repeat(8, 1)

    logits = model(seq)
    print(f"MTP heads: {len(logits)} depths, each logits shape {tuple(logits[0].shape)}")
    assert len(logits) == 2

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for step in range(200):
        loss, _ = model.loss(seq)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"final avg multi-token loss: {loss.item():.3f}")

    # check both heads on one sequence
    model.eval()
    with torch.no_grad():
        lg = model(seq[:1])
        p1 = lg[0][0].argmax(-1)                       # predicts t+1
        p2 = lg[1][0].argmax(-1)                       # predicts t+2
    print(f"\ninput          : {seq[0].tolist()}")
    print(f"true next (+1) : {seq[0,1:].tolist()}")
    print(f"head1 pred(+1) : {p1[:-1].tolist()}")
    print(f"true next (+2) : {seq[0,2:].tolist()}")
    print(f"head2 pred(+2) : {p2[:-2].tolist()}")
    acc1 = (p1[:-1] == seq[0, 1:]).float().mean().item()
    acc2 = (p2[:-2] == seq[0, 2:]).float().mean().item()
    print(f"\nnext-1 accuracy {acc1:.0%}, next-2 accuracy {acc2:.0%}")
    assert acc1 > 0.8 and acc2 > 0.8, "should learn both horizons on toy data"
    print("Model predicts both t+1 and t+2 from each position.  PASS")


if __name__ == "__main__":
    demo()

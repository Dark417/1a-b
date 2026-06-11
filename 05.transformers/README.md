# `05.transformers/`

Attention and the architectures built on it — the backbone of modern AI.

| Sub-folder | Algorithms |
|---|---|
| `attention/` | scaled dot-product, multi-head, self/cross, causal mask; positional encodings (sinusoidal, learned, RoPE, ALiBi) |
| `architectures/` | the Transformer (encoder-decoder), BERT (encoder-only), GPT (decoder-only), T5, Vision Transformer |

Training techniques highlighted: **LR warmup scheduling** (noam),
**layer normalization**, **residual connections**, **dropout** — all in
`architectures/transformer`.

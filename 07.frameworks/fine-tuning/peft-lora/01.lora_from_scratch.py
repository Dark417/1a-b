"""01 · LoRA from scratch — the math, in ~40 lines of pure PyTorch.

Before reaching for the PEFT library, build LoRA yourself so the equation is not
magic. For a frozen weight W0 (shape d×k), LoRA learns a low-rank update:

    W = W0 + (alpha / r) * B @ A
        B: d×r   A: r×k   r << min(d, k)   B initialised to 0 so ΔW = 0 at start.

Only A and B train. We wrap a real nn.Linear, freeze its weight, add the LoRA
branch, and verify a gradient step only moves A and B — not the base.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021,
https://arxiv.org/abs/2106.09685
Runs on CPU in well under a second. Exit code 0.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_num_threads(1)


class LoRALinear(nn.Module):
    """A drop-in wrapper around nn.Linear adding a trainable low-rank branch."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        # Freeze the pretrained weights — they do NOT receive gradients.
        for p in self.base.parameters():
            p.requires_grad_(False)

        d_out, d_in = base.weight.shape  # (d, k)
        self.r = r
        self.scaling = alpha / r  # the α/r factor decouples rank from LR
        self.dropout = nn.Dropout(dropout)
        # A: r×k  (init small random),  B: d×r  (init zero -> ΔW = 0 at step 0)
        self.lora_A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen = self.base(x)                       # W0 · x  (+ bias)
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T  # (B·A·x)
        return frozen + self.scaling * delta


def main() -> None:
    d_in, d_out = 32, 16
    base = nn.Linear(d_in, d_out)
    layer = LoRALinear(base, r=4, alpha=8)

    total = sum(p.numel() for p in layer.parameters())
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"params: total={total}  trainable={trainable}  "
          f"({100*trainable/total:.1f}%) — only A and B train")

    # Because B=0 at init, the LoRA output must equal the frozen output exactly.
    x = torch.randn(4, d_in)
    with torch.no_grad():
        assert torch.allclose(layer(x), base(x)), "ΔW must be 0 at init"
    print("at init: ΔW = 0  (output == frozen base) ✓")

    # One optimisation step; confirm the base weight is untouched.
    base_before = base.weight.detach().clone()
    opt = torch.optim.SGD([p for p in layer.parameters() if p.requires_grad], lr=0.1)
    target = torch.randn(4, d_out)
    for step in range(5):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    print(f"after 5 steps: loss={loss.item():.4f}")
    assert torch.equal(base.weight, base_before), "frozen base must not move"
    assert layer.lora_B.abs().sum() > 0, "B should have moved off zero"
    print("frozen base unchanged; B moved off zero — LoRA learned ΔW ✓")

    # The merged weight: W = W0 + (α/r) B A  — used at inference for zero overhead.
    merged = base.weight + layer.scaling * (layer.lora_B @ layer.lora_A)
    print(f"merged weight shape {tuple(merged.shape)} == base shape "
          f"{tuple(base.weight.shape)} — mergeable, no inference latency")


if __name__ == "__main__":
    main()

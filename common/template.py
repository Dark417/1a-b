"""
<Algorithm Name>  — TEMPLATE (copy me)
======================================
One-paragraph summary: what it is, what problem it solves, the key idea.

Variants implemented here:
    - <variant A>
    - <variant B>

Training techniques demonstrated:
    - <technique> (see training-techniques/README.md)

References:
    - <paper / textbook chapter>
"""

from __future__ import annotations

import numpy as np

SEED = 0


# ---------------------------------------------------------------------------
# 1. NumPy implementation (from scratch — no autograd). The math made explicit.
# ---------------------------------------------------------------------------
class AlgorithmNumPy:
    def __init__(self) -> None:
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AlgorithmNumPy":
        # Implement the forward pass, loss, and the gradient/update BY HAND.
        # Comment each non-obvious line with the math it represents.
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
# 2. PyTorch implementation (idiomatic — autograd / nn.Module).
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn


def get_device() -> "torch.device":
    """CUDA > MPS > CPU. See docs/gpu-setup.md."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AlgorithmTorch(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        ...


# ---------------------------------------------------------------------------
# 3. Demo — runs when called directly.
# ---------------------------------------------------------------------------
def demo() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # 1) make tiny toy data
    # 2) fit the NumPy version, print/visualize
    # 3) fit the PyTorch version, compare results
    print("TODO: implement demo()")


if __name__ == "__main__":
    demo()

"""
Small shared helpers used across the tutorials.
Kept deliberately tiny — the algorithms should stay self-contained.
"""

from __future__ import annotations

import numpy as np


def get_device():
    """CUDA > MPS > CPU. See docs/gpu-setup.md."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 0) -> None:
    """Seed NumPy and (if available) PyTorch for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def train_test_split(X, y, test_frac=0.2, seed=0):
    """Minimal split helper (so we don't always need sklearn)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def standardize(X, mean=None, std=None):
    """Zero-mean, unit-variance per feature. Returns (Xz, mean, std)."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-12
    return (X - mean) / std, mean, std


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

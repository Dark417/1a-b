"""
CycleGAN
========
CycleGAN learns to translate between two domains *without paired examples*. With
no (input, target) pairs, the adversarial loss alone is underconstrained: a
generator could map every input to a single plausible output in the target
domain and still fool the discriminator. CycleGAN adds a **cycle-consistency**
constraint - translate X->Y->X (and Y->X->Y) and you should recover the original
input - which pins down a meaningful, bijective-ish mapping. Two generators
(G: X->Y, F: Y->X) and two discriminators (D_Y, D_X) are trained jointly, plus an
optional identity loss that stabilizes color/scale. Here we use tiny synthetic
2-D domains so the whole demo runs in a few seconds on CPU.

Variants implemented here:
    - CycleGAN with two generators + two discriminators (LSGAN-style adv loss)
    - Cycle-consistency (L1) loss in both directions
    - Identity loss (G(y) ~ y, F(x) ~ x) to preserve scale/color

Training techniques demonstrated:
    - ADVERSARIAL / MINIMAX TRAINING (see training-techniques/README.md)
    - CYCLE-CONSISTENCY as a structural regularizer for unpaired translation
    - Least-squares adversarial loss (stabler than BCE for this game)

References:
    - Zhu et al. (2017), "Unpaired Image-to-Image Translation using
      Cycle-Consistent Adversarial Networks"
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SEED = 0


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Two synthetic 2-D domains (UNPAIRED): domain X is a Gaussian blob; domain Y is
# the same blob rotated 90 deg, scaled, and shifted. The ground-truth mapping is
# an affine transform, which CycleGAN should recover from samples alone.
# ---------------------------------------------------------------------------
_THETA = np.pi / 2
_ROT = np.array([[np.cos(_THETA), -np.sin(_THETA)],
                 [np.sin(_THETA), np.cos(_THETA)]], dtype=np.float32)
_SCALE = np.float32(1.5)
_SHIFT = np.array([2.0, -1.0], dtype=np.float32)


def make_domain_x(n: int = 1024, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, 2)) * np.array([1.0, 0.4], np.float32)).astype(np.float32)


def make_domain_y(n: int = 1024, seed: int = SEED + 7) -> np.ndarray:
    """Domain Y = affine(X) on *independently* drawn samples (so it is unpaired)."""
    x = make_domain_x(n, seed)
    return (_SCALE * (x @ _ROT.T) + _SHIFT).astype(np.float32)


# ---------------------------------------------------------------------------
# Generator and discriminator: small MLPs operating on 2-D points.
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """Maps a point from one domain to the other (residual-style)."""

    def __init__(self, dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Discriminator(nn.Module):
    """Scores whether a point belongs to its target domain (LSGAN: unbounded)."""

    def __init__(self, dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Trainer: jointly optimize G (X->Y), F (Y->X), D_Y, D_X.
# ---------------------------------------------------------------------------
class CycleGANTorch:
    def __init__(self, dim: int = 2, lr: float = 2e-4,
                 lambda_cyc: float = 10.0, lambda_id: float = 5.0):
        torch.manual_seed(SEED)
        self.dev = get_device()
        self.lambda_cyc, self.lambda_id = lambda_cyc, lambda_id
        self.G = Generator(dim).to(self.dev)   # X -> Y
        self.F = Generator(dim).to(self.dev)   # Y -> X
        self.D_Y = Discriminator(dim).to(self.dev)
        self.D_X = Discriminator(dim).to(self.dev)
        self.optG = torch.optim.Adam(
            list(self.G.parameters()) + list(self.F.parameters()),
            lr=lr, betas=(0.5, 0.999))
        self.optD = torch.optim.Adam(
            list(self.D_X.parameters()) + list(self.D_Y.parameters()),
            lr=lr, betas=(0.5, 0.999))
        self.l1 = nn.L1Loss()

    @staticmethod
    def _mse(out: torch.Tensor, target: float) -> torch.Tensor:
        return ((out - target) ** 2).mean()

    def fit(self, X: np.ndarray, Y: np.ndarray, steps: int = 800, batch: int = 128):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.dev)
        self.g_hist, self.cyc_hist = [], []
        for _ in range(steps):
            x = X[torch.randint(0, len(X), (batch,), device=self.dev)]
            y = Y[torch.randint(0, len(Y), (batch,), device=self.dev)]

            # --- Generators: adversarial + cycle + identity ---
            fake_y = self.G(x)              # X -> Y
            fake_x = self.F(y)              # Y -> X
            rec_x = self.F(fake_y)          # X -> Y -> X
            rec_y = self.G(fake_x)          # Y -> X -> Y
            # adversarial: fakes should look real to the target discriminator
            adv = self._mse(self.D_Y(fake_y), 1.0) + self._mse(self.D_X(fake_x), 1.0)
            cyc = self.l1(rec_x, x) + self.l1(rec_y, y)
            idt = self.l1(self.G(y), y) + self.l1(self.F(x), x)  # identity on target
            lossG = adv + self.lambda_cyc * cyc + self.lambda_id * idt
            self.optG.zero_grad(); lossG.backward(); self.optG.step()

            # --- Discriminators: real -> 1, fake -> 0 (detached) ---
            lossD = (self._mse(self.D_Y(y), 1.0) + self._mse(self.D_Y(fake_y.detach()), 0.0)
                     + self._mse(self.D_X(x), 1.0) + self._mse(self.D_X(fake_x.detach()), 0.0))
            self.optD.zero_grad(); lossD.backward(); self.optD.step()

            self.g_hist.append(lossG.item()); self.cyc_hist.append(cyc.item())
        return self

    @torch.no_grad()
    def generate(self, X: np.ndarray) -> np.ndarray:
        """Translate domain-X points to domain Y via G."""
        x = torch.as_tensor(X, dtype=torch.float32, device=self.dev)
        return self.G(x).cpu().numpy()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo():
    np.random.seed(SEED); torch.manual_seed(SEED)
    X = make_domain_x(1024)
    Y = make_domain_y(1024)
    print(f"domain X mean={X.mean(0).round(2)} std={X.std(0).round(2)}")
    print(f"domain Y mean={Y.mean(0).round(2)} std={Y.std(0).round(2)}")

    gan = CycleGANTorch().fit(X, Y, steps=800, batch=128)
    c0, c1 = np.mean(gan.cyc_hist[:50]), np.mean(gan.cyc_hist[-50:])
    print(f"cycle-consistency L1: {c0:.3f} -> {c1:.3f} (falling => X->Y->X recovers X)")

    # Quality proxy: translated X should match domain Y's distribution.
    fake_y = gan.generate(make_domain_x(800, seed=99))
    print(f"translated G(X) mean={fake_y.mean(0).round(2)} std={fake_y.std(0).round(2)}")
    print(f"target     Y    mean={Y.mean(0).round(2)} std={Y.std(0).round(2)}")
    err = np.linalg.norm(fake_y.mean(0) - Y.mean(0))
    print(f"distribution-mean gap |G(X) - Y| = {err:.3f} (small => good translation)")


if __name__ == "__main__":
    demo()

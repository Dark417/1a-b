"""
02.scheduler_math.py — DDPMScheduler API + diffusion math deep-dive

Official docs:
  https://huggingface.co/docs/diffusers/api/schedulers/ddpm
  https://huggingface.co/docs/diffusers/api/schedulers/overview
  https://arxiv.org/abs/2006.11239  (Ho et al. DDPM 2020)

Fully local — no model download required.
"""

import random
import sys

import numpy as np
import torch

# ── reproducibility ──────────────────────────────────────────────────────────
torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

from _lib import banner, note_skip, safe
from diffusers import DDPMScheduler

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Instantiate scheduler and inspect the noise schedule arrays
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 1 — DDPMScheduler: noise schedule arrays")

# 200 training timesteps keeps arrays small and printable.
scheduler = DDPMScheduler(num_train_timesteps=200, beta_schedule="linear")

print(f"num_train_timesteps : {scheduler.config.num_train_timesteps}")
print(f"beta_schedule       : {scheduler.config.beta_schedule}")
print(f"beta_start          : {scheduler.config.beta_start}")
print(f"beta_end            : {scheduler.config.beta_end}")

betas          = scheduler.betas           # shape (T,)
alphas         = scheduler.alphas          # shape (T,)  = 1 - betas
alphas_cumprod = scheduler.alphas_cumprod  # shape (T,)  = prod_{s<=t} alphas_s

T = len(betas)
print(f"\nbetas         shape={tuple(betas.shape)}"
      f"  min={betas.min().item():.6f}  max={betas.max().item():.6f}")
print(f"alphas        shape={tuple(alphas.shape)}"
      f"  min={alphas.min().item():.6f}  max={alphas.max().item():.6f}")
print(f"alphas_cumprod shape={tuple(alphas_cumprod.shape)}"
      f"  t=0→{alphas_cumprod[0].item():.6f}  t={T-1}→{alphas_cumprod[-1].item():.6f}")

# Show a few timesteps to illustrate how signal decays
print("\n  t    beta_t     alpha_t   alpha_bar_t  sqrt(ab)  sqrt(1-ab)")
for t_idx in [0, 25, 50, 100, 150, 199]:
    b  = betas[t_idx].item()
    a  = alphas[t_idx].item()
    ab = alphas_cumprod[t_idx].item()
    print(f"  {t_idx:3d}  {b:.6f}  {a:.6f}  {ab:.6f}     "
          f"{ab**0.5:.4f}     {(1-ab)**0.5:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Forward noising — add_noise  (q(x_t | x_0) closed-form)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 2 — Forward noising: add_noise (q(x_t | x_0))")

print(
    "\nMath recap:\n"
    "  x_t = sqrt(ᾱ_t) · x_0  +  sqrt(1 - ᾱ_t) · ε\n"
    "  ε ~ N(0, I)\n"
    "  Implemented by scheduler.add_noise(x0, noise, timesteps)\n"
)

# Toy 1-D signal as a [1,1,4] tensor (batch=1, C=1, spatial=4)
x0    = torch.tensor([[[0.8, -0.3, 0.5, -0.7]]])   # clean signal
noise = torch.randn_like(x0)

print(f"x0    : {x0[0,0].tolist()}")
print(f"noise : {noise[0,0].tolist()}")

for t_idx in [0, 50, 100, 150, 199]:
    t_tensor = torch.tensor([t_idx], dtype=torch.long)

    # Scheduler API
    x_t_api = scheduler.add_noise(x0.clone(), noise.clone(), t_tensor)

    # Manual implementation — should match exactly
    ab = alphas_cumprod[t_idx].item()
    x_t_manual = (ab ** 0.5) * x0 + ((1.0 - ab) ** 0.5) * noise

    api_vals    = x_t_api[0,0].tolist()
    manual_vals = x_t_manual[0,0].tolist()
    max_diff    = max(abs(a - m) for a, m in zip(api_vals, manual_vals))

    print(f"\n  t={t_idx:3d}  ᾱ_t={ab:.4f}  sqrt(ᾱ)={ab**0.5:.4f}  sqrt(1-ᾱ)={(1-ab)**0.5:.4f}")
    print(f"    x_t  (API)   : {[round(v,4) for v in api_vals]}")
    print(f"    x_t  (manual): {[round(v,4) for v in manual_vals]}")
    print(f"    max |diff|   : {max_diff:.2e}  (should be ~0)")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: set_timesteps — inference-time subset selection
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 3 — set_timesteps: selecting inference timesteps")

print(
    "\nDuring inference the scheduler skips most timesteps, running only\n"
    "num_inference_steps steps chosen to cover the full [0, T] range.\n"
)

for num_steps in [5, 10, 20]:
    scheduler.set_timesteps(num_steps)
    ts = scheduler.timesteps.tolist()
    print(f"  num_inference_steps={num_steps:2d}  "
          f"timesteps (first 6): {ts[:6]}  …  last: {ts[-1]}")

# Reset to 200 for part 4
scheduler.set_timesteps(200)

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Reverse step — scheduler.step (p_θ approximation with a toy "UNet")
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 4 — Reverse step: scheduler.step (manual + API)")

print(
    "\nMath recap (DDPM reverse step):\n"
    "  Given model prediction ε̂ at timestep t:\n"
    "    μ_θ(x_t, t) = (1/sqrt(α_t)) · [x_t − (β_t/sqrt(1−ᾱ_t)) · ε̂]\n"
    "  Then sample:\n"
    "    x_{t-1} = μ_θ  +  sqrt(β_t) · z,   z ~ N(0,I)  (stochastic)\n"
    "  scheduler.step(noise_pred, t, x_t) returns a DDPMSchedulerOutput\n"
    "  whose .prev_sample field IS x_{t-1}.\n"
)

# Pick t=50, start from a noisy x_t
t_demo  = 50
t_tensor = torch.tensor([t_demo])

x0_demo = torch.tensor([[[0.8, -0.3, 0.5, -0.7]]])
noise_gt = torch.randn(1, 1, 4, generator=torch.Generator().manual_seed(7))
x_t_demo = scheduler.add_noise(x0_demo, noise_gt, t_tensor)

# Toy "UNet": just predict the true noise (oracle) — gives the best reverse step
noise_pred = noise_gt.clone()   # oracle prediction ε̂ = ε

# Scheduler API
step_out  = scheduler.step(noise_pred, t_demo, x_t_demo)
x_prev_api = step_out.prev_sample

# Manual DDPM reverse step (deterministic mean; scheduler internally adds noise)
b_t  = betas[t_demo].item()
a_t  = alphas[t_demo].item()
ab_t = alphas_cumprod[t_demo].item()
coeff_noise = b_t / (1.0 - ab_t) ** 0.5
mu = (1.0 / a_t ** 0.5) * (x_t_demo - coeff_noise * noise_pred)

print(f"  t={t_demo}  β_t={b_t:.6f}  α_t={a_t:.6f}  ᾱ_t={ab_t:.6f}")
print(f"  x_t    : {x_t_demo[0,0].tolist()}")
print(f"  μ_θ    : {[round(v,4) for v in mu[0,0].tolist()]}  (deterministic mean)")
print(f"  x_{t_demo-1} (API, stochastic): {[round(v,4) for v in x_prev_api[0,0].tolist()]}")
print("\n  Note: API adds a small stochastic term z·sqrt(β̃_t) so x_prev != mu exactly.")

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: Full mini reverse loop (4 steps, toy 2-D image)
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 5 — Mini reverse loop (4 steps, 1×1×4×4 image)")

print("\nStarting from pure Gaussian noise and denoising with a trivial predictor.")
print("(Predictor = predict zeros; illustrates shape / data flow, not real quality.)\n")

scheduler.set_timesteps(4)
sample = torch.randn(1, 1, 4, 4, generator=torch.Generator().manual_seed(99))
print(f"t=start  shape={tuple(sample.shape)}  "
      f"mean={sample.mean():.4f}  std={sample.std():.4f}")

for step_i, t in enumerate(scheduler.timesteps):
    # "Predict zeros" as noise — not a real model, just demos the API shape
    noise_pred = torch.zeros_like(sample)
    out = scheduler.step(noise_pred, t, sample)
    sample = out.prev_sample
    print(f"  step {step_i+1}/4  t={t.item():3d}  "
          f"mean={sample.mean():.4f}  std={sample.std():.4f}")

print(f"\nFinal tensor shape : {tuple(sample.shape)}")
print(f"Value range        : [{sample.min():.4f}, {sample.max():.4f}]")

print("\n[done] 02.scheduler_math.py — exit 0")

"""
01.tiny_pipeline.py — Local UNet2DModel + DDPMScheduler denoising demo
       + optional download of hf-internal-testing/tiny-stable-diffusion-pipe

Official docs:
  https://huggingface.co/docs/diffusers/api/models/unet2d
  https://huggingface.co/docs/diffusers/api/schedulers/ddpm
  https://huggingface.co/docs/diffusers/api/pipelines/overview
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

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Fully local — tiny UNet2DModel + DDPMScheduler
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 1 — Local tiny UNet2DModel + DDPMScheduler")

from diffusers import UNet2DModel, DDPMScheduler

# Build a very small UNet that runs on CPU in milliseconds.
# sample_size=16 → 16×16 spatial resolution; down/up block channels kept tiny.
model_cfg = dict(
    sample_size=16,
    in_channels=1,
    out_channels=1,
    layers_per_block=1,
    block_out_channels=(16, 32),
    down_block_types=("DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D"),
)

print("Building tiny UNet2DModel (local, no download)...")
model = UNet2DModel(**model_cfg)
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {num_params:,}")

# Build DDPM scheduler — 100 total timesteps (fast for demo)
scheduler = DDPMScheduler(num_train_timesteps=100)
print(f"  Scheduler: DDPMScheduler, {scheduler.config.num_train_timesteps} train timesteps")

# ── Short reverse denoising loop ─────────────────────────────────────────────
NUM_STEPS = 3  # very few — purely for demo speed
scheduler.set_timesteps(NUM_STEPS)
print(f"\nRunning {NUM_STEPS}-step reverse denoising loop on CPU ...")

# Start from pure Gaussian noise  [batch=1, C=1, H=16, W=16]
image = torch.randn(1, 1, 16, 16, generator=torch.Generator().manual_seed(0))
print(f"  Initial noise  shape={image.shape}  "
      f"mean={image.mean().item():.4f}  std={image.std().item():.4f}")

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        noise_pred = model(image, t).sample          # UNet prediction
        image = scheduler.step(noise_pred, t, image).prev_sample
        print(f"  step {i+1}/{NUM_STEPS}  t={t.item():3d}  "
              f"shape={image.shape}  "
              f"mean={image.mean().item():.4f}  std={image.std().item():.4f}")

print(f"\nFinal denoised tensor — shape: {image.shape}  "
      f"min={image.min().item():.4f}  max={image.max().item():.4f}")

# Clamp to [–1, 1] range as a post-processing step (standard for pixel-space models)
image_clamped = image.clamp(-1, 1)
print(f"Clamped output     — shape: {image_clamped.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Try downloading hf-internal-testing/tiny-stable-diffusion-pipe
# ─────────────────────────────────────────────────────────────────────────────
banner("Part 2 — hf-internal-testing/tiny-stable-diffusion-pipe (graceful skip)")

TINY_PIPE_REPO = "hf-internal-testing/tiny-stable-diffusion-pipe"

def load_tiny_sd_pipe():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        TINY_PIPE_REPO,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe

ok, result = safe(load_tiny_sd_pipe)
if not ok:
    note_skip(f"needs network/model — demonstrating API shape with local fallback\n  ({result})")
    print("\nAPI shape (what the pipeline call looks like):")
    print("  pipe = StableDiffusionPipeline.from_pretrained(repo_id)")
    print("  output = pipe('a cat', num_inference_steps=2, guidance_scale=1.0)")
    print("  image  = output.images[0]   # PIL Image")
    print("  image.save('out.png')")
else:
    pipe = result
    print(f"Loaded pipeline: {type(pipe).__name__}")

    def run_inference():
        generator = torch.Generator().manual_seed(42)
        out = pipe(
            "a tiny cat",
            num_inference_steps=2,
            guidance_scale=1.0,
            height=64,
            width=64,
            generator=generator,
        )
        return out

    ok2, out = safe(run_inference)
    if ok2:
        img = out.images[0]
        import numpy as np
        arr = np.array(img)
        print(f"Pipeline output — image size: {img.size}  dtype: {arr.dtype}  shape: {arr.shape}")
    else:
        note_skip(f"inference failed: {out}")

print("\n[done] 01.tiny_pipeline.py — exit 0")

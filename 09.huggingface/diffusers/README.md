# Diffusers — University-Grade Explainer

**Official docs:** https://huggingface.co/docs/diffusers/index  
**API reference:** https://huggingface.co/docs/diffusers/api/overview

---

## What is Diffusers?

`diffusers` is Hugging Face's library for state-of-the-art diffusion models.
It provides pre-trained pipelines (Stable Diffusion, SDXL, Kandinsky, etc.),
modular model components (UNets, VAEs), and a rich scheduler zoo — all with a
consistent API that works on CPU and GPU.

---

## The Diffusion Process: Intuition and Math

### Intuition

Diffusion models learn to *reverse* a gradual noising process.  Training
teaches the network to "denoise"; inference *starts from pure Gaussian noise*
and repeatedly applies the learned denoiser until a clean image appears.

### Forward Process  q(x_t | x_0)

Given a clean image x₀, we define a Markov chain that adds Gaussian noise at
each timestep t = 1 … T:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - β_t) · x_{t-1},  β_t · I)
```

where β_t is the **noise schedule** (a small, monotonically increasing
sequence).  Thanks to the reparametrisation trick the distribution at any
arbitrary t can be written *closed-form*:

```
q(x_t | x_0) = N(x_t;  sqrt(ᾱ_t) · x_0,  (1 - ᾱ_t) · I)

αt    = 1 - βt
ᾱt   = ∏_{s=1}^{t} αs          (cumulative product)
```

So we can sample the noisy image in one shot:

```
x_t = sqrt(ᾱ_t) · x_0  +  sqrt(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

### Reverse Process  p_θ(x_{t-1} | x_t)

The reverse transition is also Gaussian (under mild conditions):

```
p_θ(x_{t-1} | x_t) = N(x_{t-1};  μ_θ(x_t, t),  Σ_θ(x_t, t))
```

The neural network (a UNet) parameterises μ_θ by *predicting the noise* ε̂.
Given the prediction we recover the mean:

```
μ_θ(x_t, t) = (1/sqrt(αt)) · [ x_t  -  (βt / sqrt(1 - ᾱt)) · ε_θ(x_t, t) ]
```

### Training Objective — Noise Prediction (DDPM)

The simplified DDPM loss is a mean-squared error between the true noise that
was added and the network's prediction:

```
L_simple = E_{t, x_0, ε} [ || ε  -  ε_θ(x_t, t) ||² ]
```

where:
- t is sampled uniformly from {1 … T}
- x_0 is a training image
- ε ~ N(0, I) is the true noise
- x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε  (forward-process sample)

**Paper:** Ho et al., "Denoising Diffusion Probabilistic Models" (2020)  
https://arxiv.org/abs/2006.11239

---

## Building Blocks

### Models

| Class | Role |
|---|---|
| `UNet2DModel` | Unconditional / class-conditional UNet (DDPM-style) |
| `UNet2DConditionModel` | Text-/image-conditioned UNet (used in SD 1.x / 2.x) |
| `AutoencoderKL` | Variational autoencoder; encodes images to latent space (SD uses 4× downsampling) |
| `Transformer2DModel` | Attention-based backbone (DiT, PixArt-α) |
| `ControlNetModel` | Side-network that injects structural conditioning (edges, depth, …) |

**Docs:** https://huggingface.co/docs/diffusers/api/models/overview

### Schedulers

Schedulers implement the *noise schedule* and the *reverse-step formula*.
They do NOT contain any learned weights; they are interchangeable plug-ins.

| Scheduler | Key property |
|---|---|
| `DDPMScheduler` | Original DDPM stochastic reverse process (1 000 steps) |
| `DDIMScheduler` | Deterministic, non-Markovian; can use far fewer steps (50) |
| `EulerDiscreteScheduler` | Euler method on the probability-flow ODE; fast + high quality |
| `DPMSolverMultistepScheduler` | DPM-Solver++; often best quality/speed trade-off |
| `PNDMScheduler` | Pseudo-numerical method; default in early SD |
| `UniPCMultistepScheduler` | Unified Predictor-Corrector; strong on SD |

Every scheduler exposes:
- `set_timesteps(num_inference_steps)` — select T inference steps from the full schedule
- `step(model_output, t, sample)` → `DDIMSchedulerOutput.prev_sample`
- `add_noise(original, noise, timesteps)` — forward-process utility

**Docs:** https://huggingface.co/docs/diffusers/api/schedulers/overview

---

## DiffusionPipeline / StableDiffusionPipeline

`DiffusionPipeline.from_pretrained(repo_id)` is the high-level entry point.
It auto-detects the pipeline class, downloads all components (tokenizer,
text encoder, VAE, UNet, scheduler), and bundles them together.

`StableDiffusionPipeline` is the specialisation for Stable Diffusion 1.x/2.x.

### Pipeline call signature

```python
image = pipe(
    prompt               = "a futuristic city at dusk",   # str or List[str]
    negative_prompt      = "blurry, low quality",          # optional
    num_inference_steps  = 50,       # number of denoising steps
    guidance_scale       = 7.5,      # classifier-free guidance strength
    height               = 512,
    width                = 512,
    generator            = torch.Generator().manual_seed(42),
    num_images_per_prompt= 1,
).images[0]
```

**Classifier-Free Guidance (CFG):**  The UNet is evaluated *twice* per step —
once with the text embedding and once with a null embedding.  The outputs are
blended:

```
ε_guided = ε_uncond + guidance_scale · (ε_cond − ε_uncond)
```

Higher `guidance_scale` → more prompt-adherent but less diverse images.
Typical range: 5–15.

**Docs:** https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img

---

## img2img, Inpainting, and ControlNet

### img2img

`StableDiffusionImg2ImgPipeline` starts denoising from a *partially noised*
real image instead of pure noise.  The `strength` parameter (0–1) controls
how much of the original to preserve.

**Docs:** https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img

### Inpainting

`StableDiffusionInpaintPipeline` accepts a mask image; only the masked region
is denoised.  The unmasked region is preserved.

**Docs:** https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint

### ControlNet

ControlNet adds a parallel encoder that takes a *conditioning image* (Canny
edges, depth map, human pose skeleton …) and injects feature maps into the
UNet at multiple resolution levels.  This gives precise structural control
without fine-tuning the base UNet.

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet
)
```

**Docs:** https://huggingface.co/docs/diffusers/api/pipelines/controlnet

---

## LoRA for Diffusion Models

LoRA (Low-Rank Adaptation) fine-tunes diffusion models by inserting tiny
rank-decomposed weight matrices (A, B) beside the original frozen weights.
Only the A/B matrices are trained, drastically reducing VRAM and storage.

In `diffusers`, LoRA weights ship as `.safetensors` files and can be loaded
at inference time:

```python
pipe.load_lora_weights("path/to/lora.safetensors")
pipe.fuse_lora(lora_scale=0.8)   # bake LoRA into the model weights
pipe.unfuse_lora()                # revert
```

**Docs:** https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference

---

## Quick-Start Code Sketch

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    "An astronaut riding a horse on Mars",
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("output.png")
```

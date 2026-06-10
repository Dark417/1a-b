# GPU Setup

Everything in this repo **runs on CPU** — the toy datasets are tiny on purpose.
GPU just makes the deep-learning and generative examples faster. This guide
covers NVIDIA CUDA, Apple Silicon (MPS), and Google Colab.

The code already picks the best device automatically:

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")          # NVIDIA GPU
    if torch.backends.mps.is_available():
        return torch.device("mps")           # Apple Silicon
    return torch.device("cpu")

device = get_device()
model = MyModel().to(device)
x = x.to(device)
```

Always move **both** the model and the input tensors to `device`, and move
results back with `.cpu()` before converting to NumPy.

---

## 1. NVIDIA GPU (CUDA) — Linux / Windows

1. **Check your GPU & driver**
   ```bash
   nvidia-smi
   ```
   Note the "CUDA Version" shown in the top-right; install a PyTorch build at or
   below that version.

2. **Install the CUDA build of PyTorch** (example: CUDA 12.1). Use a fresh
   virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
   Pick the right URL from <https://pytorch.org/get-started/locally/>.

3. **Verify**
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   # -> True NVIDIA GeForce RTX ...
   ```

### Tips
- **Out of memory?** Reduce batch size, use `torch.cuda.amp.autocast()` mixed
  precision, or call `torch.cuda.empty_cache()`.
- **Pin a GPU:** `CUDA_VISIBLE_DEVICES=0 python train.py`.
- **Determinism:** `torch.manual_seed(0)` plus
  `torch.use_deterministic_algorithms(True)` (may disable some fast kernels).

---

## 2. Apple Silicon (M1/M2/M3) — macOS

PyTorch uses the **MPS** (Metal) backend.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision          # default wheels include MPS
python -c "import torch; print(torch.backends.mps.is_available())"   # -> True
```

### Tips
- Some ops aren't implemented on MPS yet. Enable CPU fallback:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- MPS uses unified memory; very large batches can still OOM the system.

---

## 3. Google Colab (free GPU)

1. Open the notebook in Colab.
2. **Runtime → Change runtime type → Hardware accelerator → GPU (T4)**.
3. PyTorch is pre-installed. Verify:
   ```python
   import torch; print(torch.cuda.is_available())   # True
   ```
4. Clone the repo inside Colab if needed:
   ```python
   !git clone https://github.com/dark417/1a-b.git
   %cd 1a-b
   ```

---

## 4. Sanity-check script

Run this once after setup:

```python
import torch, time

def get_device():
    if torch.cuda.is_available():            return torch.device("cuda")
    if torch.backends.mps.is_available():    return torch.device("mps")
    return torch.device("cpu")

dev = get_device()
print("Using:", dev)

a = torch.randn(4096, 4096, device=dev)
b = torch.randn(4096, 4096, device=dev)
if dev.type == "cuda": torch.cuda.synchronize()
t0 = time.time()
c = a @ b
if dev.type == "cuda": torch.cuda.synchronize()
print(f"4096x4096 matmul: {time.time() - t0:.4f}s  (result {tuple(c.shape)})")
```

If the device prints `cuda` or `mps` and the matmul is fast, you're set.

---

## 5. Common issues

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is `False` | CPU-only wheel installed — reinstall with the CUDA `--index-url`. |
| `CUDA error: out of memory` | Smaller batch, mixed precision (`amp`), `empty_cache()`. |
| Driver/toolkit mismatch | Install a PyTorch CUDA build ≤ the version from `nvidia-smi`. |
| MPS op not implemented | `export PYTORCH_ENABLE_MPS_FALLBACK=1`. |
| Slower on GPU for tiny demos | Expected — kernel launch overhead dominates; GPU wins at scale. |

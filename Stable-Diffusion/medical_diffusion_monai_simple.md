# Notes — Medical Image Diffusion with MONAI

**Notebook:** `medical_diffusion_monai_simple.ipynb`
**Goal:** apply the diffusion ideas from the Stable Diffusion intro to **medical
images**, using the library built for the job.
**Stack:** MONAI core (≥1.4), MedNIST dataset. Runs on Colab/Kaggle GPU.

---

## Why MONAI instead of `diffusers`

Generic Stable Diffusion is built for **natural images + text prompts**. For
CT/MRI/X-ray there's **[MONAI](https://monai.io/)**, the standard medical-imaging
DL framework. It provides the same two diffusion building blocks, tuned for
medical data and with **no text encoder needed**:

```python
from monai.networks.nets import DiffusionModelUNet   # predicts the noise
from monai.networks.schedulers import DDPMScheduler   # adds/removes noise
```

> **Package note:** these were originally a separate `monai-generative` package,
> but were **merged into MONAI core in v1.4** (late 2024). Use core MONAI
> (`pip install "monai>=1.4.0" einops`); the old standalone package is frozen and
> breaks against recent MONAI. This is also why it "just works" on Colab/Kaggle.

## What the notebook does (5 steps)

1. **Install** MONAI core.
2. **Download MedNIST** (small 64×64 medical images) and keep one class, **HeadCT**,
   to make the task simple.
3. **Build** a small 2D `DiffusionModelUNet` + a `DDPMScheduler` (1000 timesteps).
4. **Train** — the surprisingly short diffusion loop:
   ```
   add random noise to a real image  →  ask the U-Net to predict that noise
   →  MSE loss between prediction and the real noise  →  backprop
   ```
5. **Generate** — start from pure noise and let the scheduler + U-Net denoise it
   step by step into brand-new synthetic HeadCT images that never existed.

## The core idea in one sentence

> A diffusion model is just a U-Net trained to answer: *"how much noise is in this
> image, given how far along the noising schedule we are?"* — and generation is
> running that backwards from pure noise.

## How it connects

| Concept | Where I first met it |
|---------|---------------------|
| U-Net encoder–decoder | [`../UNet-Study`](../UNet-Study/) (segmentation) |
| Diffusion / scheduler / latent space | [`stable_diffusion_intro.md`](stable_diffusion_intro.md) |
| Medical data handling (transforms, datasets) | MONAI (this notebook) |

## Honest status

- Written against the MONAI 1.4 core API and verified against the release notes,
  but **not yet executed end-to-end** on a GPU. If the first run errors, it's most
  likely a version/import detail to nudge.

## Next steps

- Train on a different class (Hand, ChestCT) or real DICOM/NIfTI scans.
- Add **latent diffusion** (`AutoencoderKL`) for higher resolution.
- Add **conditioning** (class label or segmentation mask) to control output.
- Try `spatial_dims=3` for full CT/MRI volumes.

# 03-Diffusion-Models — Stable Diffusion internals and medical diffusion

Learning diffusion models, working up from natural images to medical imaging. The U-Net from
[`../02-UNet-Segmentation/`](../02-UNet-Segmentation/) reappears here as the denoising backbone;
the pipeline studied here is what [`../05-Noise-Gate/`](../05-Noise-Gate/) later defends.

| Notebook | What it covers |
|----------|----------------|
| `stable_diffusion_intro.ipynb` | The Hugging Face `diffusers` walkthrough — text-to-image Stable Diffusion on natural images, plus the theory (VAE, U-Net, text encoder, scheduler). |
| `medical_diffusion_monai_simple.ipynb` | A minimal **medical imaging** diffusion model using **MONAI Generative**. Trains on MedNIST HeadCT images and generates new synthetic scans from noise. |

## Why MONAI for medical images?

Generic Stable Diffusion (`diffusers`) is built for natural images and text prompts.
For CT / MRI / X-ray work, **[Project MONAI](https://monai.io/)** (the standard medical-imaging DL framework) gives the same diffusion building blocks — a U-Net (`DiffusionModelUNet`) and a scheduler (`DDPMScheduler`) — tuned for medical data, with no text encoder required.

### Which package?

The generative models started as a separate `monai-generative` package but were **merged into MONAI core in v1.4** (late 2024). Use core MONAI:

```bash
pip install "monai>=1.4.0" einops
```

```python
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
```

Colab and Kaggle already ship a recent MONAI, so the notebook runs there with no extra setup. (The old standalone `monai-generative` v0.2.3 still installs but is frozen and breaks against newer MONAI — avoid it.)

Start with the intro notebook for the concepts, then the MONAI notebook for the medical application.

# Learning Pathway — Medical Imaging & Deep Learning

A record of what I studied during my research-assistant traineeship, from
deep-learning fundamentals through to generative models for medical imaging.

---

## Overview

```
Phase 1            Phase 2                 Phase 3
Fundamentals  →    U-Net segmentation  →   Diffusion / generative
(TF + PyTorch)     (ISIC, lungs)           (Stable Diffusion → MONAI)
```

The thread connecting everything: the **U-Net** architecture. I first learned it
for *segmentation* (predicting a mask), then saw the same encoder–decoder design
reused as the denoising backbone inside *diffusion* models.

---

## Phase 1 — Fundamentals

- Set up TensorFlow and PyTorch, learned the difference between the two
  frameworks (eager vs graph, data pipelines, tensor conventions).
- Worked through `Assignment-1/assignment.ipynb`.
- **Key takeaway:** the same model can be expressed in either framework, but the
  *data loading* and *tensor layout* conventions differ (e.g. channels-first in
  PyTorch `(C, H, W)` vs channels-last in Keras `(H, W, C)`).

## Phase 2 — U-Net for medical image segmentation

Folder: [`UNet-Study/`](UNet-Study/)

- Built a U-Net for **ISIC 2018 skin-lesion segmentation** — full encoder–decoder
  with skip connections, trained with a **Dice loss** and evaluated with **Dice
  coefficient** and **IoU**, using **5-fold cross-validation**.
- Hit a real **memory wall** (loading all 2,594 images into RAM at once) and
  solved it by switching to a **lazy, path-based data loader** that reads images
  on demand.
- Experimented with **bridging a PyTorch `DataLoader` into a Keras model** — a
  hands-on lesson in how the two frameworks' data pipelines differ.
- See: [`UNet-Study/pytorch-unet-kfold.md`](UNet-Study/pytorch-unet-kfold.md)

> ⚠️ Two notebooks in this phase (`tensorflow-unet-kfold`,
> `pretrained-unet-lung-segmentation`) are currently **empty stubs** and need to be
> re-exported from Kaggle — see their `.md` notes.

## Phase 3 — Generative models for medical imaging (current)

Folder: [`Stable-Diffusion/`](Stable-Diffusion/)

- Studied **Stable Diffusion** via the Hugging Face `diffusers` walkthrough:
  latent diffusion, the VAE / U-Net / text-encoder / scheduler components.
  See: [`Stable-Diffusion/stable_diffusion_intro.md`](Stable-Diffusion/stable_diffusion_intro.md)
- Applied the same ideas to **medical images** using **MONAI** (the standard
  medical-imaging DL framework). Built a minimal diffusion model that learns to
  *generate* synthetic HeadCT scans from noise.
  See: [`Stable-Diffusion/medical_diffusion_monai_simple.md`](Stable-Diffusion/medical_diffusion_monai_simple.md)
- **Key realization:** the diffusion "U-Net" is the same architecture from Phase 2,
  now predicting *noise* instead of a *segmentation mask*.

---

## What I'd explore next

- Latent diffusion (`AutoencoderKL`) for higher-resolution medical synthesis.
- Conditional generation — condition on a class label or a segmentation mask.
- 3D diffusion on full CT/MRI volumes (`spatial_dims=3`).
- Using synthetic images to augment small segmentation datasets (closing the loop
  between Phase 2 and Phase 3).

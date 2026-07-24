# Learning Pathway — Medical Imaging & Deep Learning

A record of what I studied during my research-assistant traineeship, from
deep-learning fundamentals through to generative models for medical imaging.

---

## Overview

```
Phase 1            Phase 2                 Phase 3                     Phase 4
Fundamentals  →    U-Net segmentation  →   Diffusion / generative  →   Adversarial robustness
(TF + PyTorch)     (ISIC, lungs)           (Stable Diffusion →         (attack → detect → gate
                                            MONAI)                      → benchmark → repair
                                                                        → localize)
```

The thread connecting everything: the **U-Net** architecture. I first learned it
for *segmentation* (predicting a mask), then saw the same encoder–decoder design
reused as the denoising backbone inside *diffusion* models.

---

## Phase 1 — Fundamentals

- Set up TensorFlow and PyTorch, learned the difference between the two
  frameworks (eager vs graph, data pipelines, tensor conventions).
- Worked through `01-Fundamentals/assignment.ipynb`.
- **Key takeaway:** the same model can be expressed in either framework, but the
  *data loading* and *tensor layout* conventions differ (e.g. channels-first in
  PyTorch `(C, H, W)` vs channels-last in Keras `(H, W, C)`).

## Phase 2 — U-Net for medical image segmentation

Folder: [`02-UNet-Segmentation/`](02-UNet-Segmentation/)

- Built a U-Net for **ISIC 2018 skin-lesion segmentation** — full encoder–decoder
  with skip connections, trained with a **Dice loss** and evaluated with **Dice
  coefficient** and **IoU**, using **5-fold cross-validation**.
- Hit a real **memory wall** (loading all 2,594 images into RAM at once) and
  solved it by switching to a **lazy, path-based data loader** that reads images
  on demand.
- Experimented with **bridging a PyTorch `DataLoader` into a Keras model** — a
  hands-on lesson in how the two frameworks' data pipelines differ.
- See: [`02-UNet-Segmentation/pytorch-unet-kfold.md`](02-UNet-Segmentation/pytorch-unet-kfold.md)

> ⚠️ Two notebooks in this phase (`tensorflow-unet-kfold`,
> `pretrained-unet-lung-segmentation`) are currently **empty stubs** and need to be
> re-exported from Kaggle — see their `.md` notes.

## Phase 3 — Generative models for medical imaging

Folder: [`03-Diffusion-Models/`](03-Diffusion-Models/)

- Studied **Stable Diffusion** via the Hugging Face `diffusers` walkthrough:
  latent diffusion, the VAE / U-Net / text-encoder / scheduler components.
  See: [`03-Diffusion-Models/stable_diffusion_intro.md`](03-Diffusion-Models/stable_diffusion_intro.md)
- Applied the same ideas to **medical images** using **MONAI** (the standard
  medical-imaging DL framework). Built a minimal diffusion model that learns to
  *generate* synthetic HeadCT scans from noise.
  See: [`03-Diffusion-Models/medical_diffusion_monai_simple.md`](03-Diffusion-Models/medical_diffusion_monai_simple.md)
- **Key realization:** the diffusion "U-Net" is the same architecture from Phase 2,
  now predicting *noise* instead of a *segmentation mask*.

## Phase 4 — Adversarial robustness for diffusion training (current)

Folders: [`04-Adversarial-Attacks/`](04-Adversarial-Attacks/) → [`05-Noise-Gate/`](05-Noise-Gate/)
→ [`06-Attack-Benchmark/`](06-Attack-Benchmark/) → [`07-Attack-Repair/`](07-Attack-Repair/)
→ [`08-AutoAttack-Localize/`](08-AutoAttack-Localize/)

- Learned how **FGSM/PGD adversarial attacks** fool a classifier with imperceptible
  noise, and used **Grad-CAM** to visualize how the attack shifts the model's attention.
- Built a **tiled, batched, gradient-free noise detector**: cut each image into a grid
  of tiles, score every tile's high-frequency energy in one GPU pass, flag tiles above
  a threshold calibrated on trusted clean tiles. Forward-only batching is the idea
  borrowed from **ViT-ReciproCAM** ([arXiv:2310.02588](https://arxiv.org/abs/2310.02588)).
- Deployed it as a **pre-training gate for Stable Diffusion**: screen every incoming
  training image and quarantine poisoned ones *before* `VAE.encode`, so the diffusion
  model never learns from tampered data.
- Stress-tested the gate against a wider **attack zoo** (C&W, DeepFool, FAB, Square, and a
  low-frequency/Nightshade-style poison) instead of just FGSM/PGD, and predicted that a
  high-frequency energy score would be *structurally blind* to attacks hiding in low
  spatial frequencies.
- Tested that prediction and got a **negative result**, recorded rather than buried: a bank of
  octave band-pass filters does see the low-frequency poison (LF IoU 0.05 → 0.24), but only at
  a threshold that pushes clean-image false positives to 47%. Tightening the threshold collapses
  IoU on both attack types. So the mask is not reliable enough to drive SD inpainting repair.
- **Key takeaway:** detection as a *gate* ("is this image poisoned?") and *localization*
  ("which pixels?") are different difficulties. Cheap frequency statistics are enough for the
  first and — on this evidence — not enough for the second.
- Took that negative result's own diagnosis seriously: the confound is **contrast**, so I switched
  from measuring the *energy* of the local spectrum to its *shape* (the ratio of second- to
  first-difference energy, in which contrast cancels), and evaluated it against real
  **AutoAttack** components — APGD-CE, APGD-T, FAB-T, Square — on my own Kaggle dataset.
- Two lessons I would not have predicted. **A detector can be wrong in the useful direction:**
  fitted on L∞ noise it scored smooth L2 attacks at AUC 0.34, which is not a miss but an
  *inversion* — L∞ noise flattens the local spectrum while smooth noise steepens it — so the score
  has to be two-sided. And **a learned model given hand-designed features reconstructed the
  principle behind them**: free to weight five features, it put large opposite-sign weights on two
  energy features, i.e. built a ratio of its own accord.
- **Method lesson that outlasts the project:** every wrong conclusion in this phase came from a
  measurement shortcut, not from bad code — a fixed threshold instead of a matched false-positive
  rate, unequal perturbation energy between attacks, thresholds tuned on the test image. The
  protocol is part of the result.

---

## What I'd explore next

- Latent diffusion (`AutoencoderKL`) for higher-resolution medical synthesis.
- Conditional generation — condition on a class label or a segmentation mask.
- 3D diffusion on full CT/MRI volumes (`spatial_dims=3`).
- Using synthetic images to augment small segmentation datasets (closing the loop
  between Phase 2 and Phase 3).

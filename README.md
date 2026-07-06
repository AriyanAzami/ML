# ML — Medical Imaging, Diffusion Models & Adversarial Robustness

A record of my ML research-assistant traineeship: from deep-learning fundamentals,
through U-Net segmentation and Stable Diffusion, to my current project — **detecting
adversarially poisoned images before they reach a diffusion model's training loop**.

See [LEARNING_PATHWAY.md](LEARNING_PATHWAY.md) for the full study narrative.

## Repository map

| Folder | What's in it |
|---|---|
| [`Assignment-1/`](Assignment-1/) | Deep-learning fundamentals (TensorFlow vs PyTorch basics). |
| [`UNet-Study/`](UNet-Study/) | U-Net for medical image segmentation (ISIC skin lesions, k-fold CV, Dice/IoU). |
| [`Stable-Diffusion/`](Stable-Diffusion/) | Stable Diffusion internals (VAE / UNet / scheduler) + a minimal MONAI medical diffusion model. |
| [`GradCAM-Attack/`](GradCAM-Attack/) | Adversarial attacks (FGSM/PGD) + Grad-CAM localization, culminating in a **tiled, batched noise detector** that flags which image tiles carry injected adversarial noise. |
| [`SD-Noise-Gate/`](SD-Noise-Gate/) | **Current project:** the tiled noise detector deployed as a *gate* in front of the Stable Diffusion training pipeline — screen every training image, quarantine poisoned ones before `VAE.encode`. |
| [`Resources/`](Resources/) | Reference material: the ViT-ReciproCAM paper (arXiv:2310.02588) and the Hugging Face Stable Diffusion walkthrough notebook. |

## The project in one paragraph

Stable Diffusion is trained on scraped images; an attacker can poison a subset with
imperceptible adversarial perturbations (FGSM/PGD, Nightshade/Glaze-style) so the model
learns from corrupted data. In `GradCAM-Attack/` I built a detector that cuts each image
into a grid of tiles and scores every tile's high-frequency energy in one GPU batch —
gradient-free and forward-only, following the batching insight of
[ViT-ReciproCAM](https://arxiv.org/abs/2310.02588). `SD-Noise-Gate/` places that detector
at the entrance of the SD training path (`image → VAE.encode → add noise → UNet`), so
tampered images are caught and quarantined **before** they ever influence training.

Notebooks are written for **Kaggle (GPU T4 ×2, Internet on)**.

# ML — Medical Imaging, Diffusion Models & Adversarial Robustness

A record of my ML research-assistant traineeship: from deep-learning fundamentals,
through U-Net segmentation and Stable Diffusion, to my current project — **detecting
adversarially poisoned images before they reach a diffusion model's training loop**.

Folders are numbered in the order I worked through them; each builds on the one before.
See [LEARNING_PATHWAY.md](LEARNING_PATHWAY.md) for the full study narrative.

## Repository map

| Folder | What's in it |
|---|---|
| [`01-Fundamentals/`](01-Fundamentals/) | Deep-learning fundamentals — TensorFlow vs PyTorch conventions (eager vs graph, data pipelines, tensor layout). |
| [`02-UNet-Segmentation/`](02-UNet-Segmentation/) | U-Net for medical image segmentation — ISIC 2018 skin lesions, k-fold CV, Dice/IoU. |
| [`03-Diffusion-Models/`](03-Diffusion-Models/) | Stable Diffusion internals (VAE / UNet / text encoder / scheduler) + a minimal MONAI medical diffusion model. |
| [`04-Adversarial-Attacks/`](04-Adversarial-Attacks/) | FGSM/PGD attacks + Grad-CAM localization, ending in the **tiled, batched noise detector** that scores each tile's high-frequency energy. |
| [`05-Noise-Gate/`](05-Noise-Gate/) | That detector installed as a *gate* in front of the SD training path — screen every image, quarantine poisoned ones before `VAE.encode`. |
| [`06-Attack-Benchmark/`](06-Attack-Benchmark/) | An attack zoo (C&W, DeepFool, FAB, Square, low-frequency) used to stress-test the gate and predict where it fails. |
| [`07-Attack-Repair/`](07-Attack-Repair/) | Can the gate localize *where* an attack hit, well enough to repair it with SD inpainting? Reports a **negative result** — see below. |
| [`08-AutoAttack-Localize/`](08-AutoAttack-Localize/) | **Current work:** localizing real AutoAttack perturbations. Beats `07` — but by fixing the ground truth and finding a colour-channel feature, after the predicted fix failed on real photographs. |
| [`Resources/`](Resources/) | Reference material: the ViT-ReciproCAM paper (arXiv:2310.02588) and the Hugging Face Stable Diffusion walkthrough notebook. |

## The project in one paragraph

Stable Diffusion is trained on scraped images; an attacker can poison a subset with
imperceptible adversarial perturbations (FGSM/PGD, Nightshade/Glaze-style) so the model
learns from corrupted data. In `04-Adversarial-Attacks/` I built a detector that cuts each
image into a grid of tiles and scores every tile's high-frequency energy in one GPU batch —
gradient-free and forward-only, following the batching insight of
[ViT-ReciproCAM](https://arxiv.org/abs/2310.02588). `05-Noise-Gate/` places that detector
at the entrance of the SD training path (`image → VAE.encode → add noise → UNet`), so
tampered images are caught and quarantined **before** they ever influence training.

## Where it currently stands

`06-Attack-Benchmark/` predicted that a high-frequency energy score would be blind to attacks
that hide in *low* spatial frequencies. `07-Attack-Repair/` tested that prediction and confirmed
it — as a **negative result** worth recording rather than a solved problem:

| detector | k | HF-attack IoU | LF-attack IoU | clean FP |
|---|---|---|---|---|
| HF only (the gate) | 3 | 0.32 | **0.05** | 16% |
| multi-band | 3 | 0.36 | **0.24** | **47%** |
| multi-band | 6 | 0.06 | **0.06** | 12% |

Widening to a bank of octave band-pass filters *does* let the detector see the low-frequency
poison, but not at any threshold that keeps clean images usable — so the resulting mask is not
good enough to drive Stable Diffusion inpainting. Detection as a **gate** (is this image
poisoned?) still works; **localization** (which pixels?) does not, yet.

`08-AutoAttack-Localize/` took that diagnosis at its word and proposed a contrast-invariant
spectral-*shape* statistic, validated on synthetic images. **It then ran on real photographs and
scored at chance.** Three things came out of taking that failure apart:

1. **The ground truth was wrong, and it cost the most.** Scores were computed against the region
   *drawn* for the attack — but AutoAttack spends its budget where the classifier is sensitive, so
   only ~40% of that region carried any perturbation. Scoring against the pixels actually perturbed
   moves the same detector on the same images from 0.84 to **0.97** AUC.
2. **The synthetic prototype had removed the very confound it was meant to survive** — 1/f images
   are spatially stationary; real photographs are not. A validation set missing your confound will
   confirm anything.
3. **The feature that works is cross-channel decorrelation.** Attacks perturb R, G and B
   independently; natural fine detail is luminance-correlated. It is the only feature that catches
   Square Attack, which is piecewise-constant and so invisible to every high-pass residual.

The shipped detector is `mean(SRM, chroma)` with **no fitting at all** — the fitted fusion beat
none of the features it was built from. The low-frequency poison remains the only attack in the zoo
with no detector above chance.

Notebooks are written for **Kaggle (GPU T4 ×2, Internet on)**.

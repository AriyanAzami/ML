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
| [`08-AutoAttack-Localize/`](08-AutoAttack-Localize/) | **Current work:** the fix for that negative result — score the *shape* of the local spectrum instead of its energy, so contrast cancels. Tested on real AutoAttack perturbations. |
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

`08-AutoAttack-Localize/` takes that diagnosis at its word. The confound is *contrast*, so the fix
is a statistic contrast cannot move: the ratio of second- to first-difference energy over a local
window, which measures the **shape** of the spectrum rather than its scale. Two things fell out of
measuring it — the score has to be **two-sided** (L∞ noise flattens the local spectrum, smooth L2
noise steepens it, so a one-sided detector doesn't miss smooth attacks, it ranks them *backwards*),
and a small learned fusion given five features **independently reconstructs the same ratio**. It is
evaluated on real `torchattacks` AutoAttack components, leave-one-image-out, with thresholds
calibrated on clean images only. The low-frequency poison remains the hardest case.

Notebooks are written for **Kaggle (GPU T4 ×2, Internet on)**.

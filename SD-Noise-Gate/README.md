# SD-Noise-Gate — a tiled noise-attack detector as a module in the Stable Diffusion workflow

This folder takes the **tiled FGSM/PGD noise detection** built in
[`../GradCAM-Attack/tiled_noise_attack_detection.ipynb`](../GradCAM-Attack/tiled_noise_attack_detection.ipynb)
and installs it as a **screening module in front of the Stable Diffusion training path**:

```
incoming image --> [ TILE + SCAN ] --keep--> VAE.encode -> add noise -> UNet
                          |
                          +--reject--> quarantine (never reaches training)
```

Stable Diffusion trains on scraped images; an attacker can slip in adversarially poisoned ones.
The module tiles every incoming image, scores each tile's high-frequency energy against a
threshold calibrated once on trusted-clean data, and **quarantines poisoned images before
`VAE.encode`** — the flagged tiles also localize *where* the noise is.

## The notebook

- **[`sd_pipeline_with_noise_gate.ipynb`](sd_pipeline_with_noise_gate.ipynb)** — the main,
  self-contained demo (run this). It joins three things:
  1. the **Stable Diffusion workflow** from [`../Resources/stable_diffusion.ipynb`](../Resources/stable_diffusion.ipynb)
     (generation + `VAE.encode → scheduler.add_noise` training path),
  2. the **tiled noise-attack detection** I practiced in `../GradCAM-Attack/`, and
  3. **ViT-ReciproCAM** ([arXiv:2310.02588](https://arxiv.org/abs/2310.02588)) — the paper whose
     lesson (forward-only scoring batches trivially, no gradients needed) is why the scan is cheap
     enough to run on every training image.

  Flow: SD generates a small demo training set → PGD poisons a subset inside irregular blobs →
  the gate scans all images in one batch across both T4s → per-image figures + a
  precision/recall/IoU scorecard → kept images proceed into the VAE/scheduler training step.

- `sd_training_noise_gate.ipynb` — an **earlier draft** kept for history. Same idea, but poisons
  ImageNet photos rather than SD-generated images and scans image-by-image instead of one batched
  scan across both GPUs. Superseded by the notebook above.

## Environment

- **Kaggle: accelerator GPU T4 ×2, Internet On.** SD generation runs on one T4; the tile scan is
  split across both. Falls back to web sample images if the SD pipeline can't load, so the gate
  logic still runs end-to-end.

## Knobs

- `SIZE=512`, `GRID=4`, `TILE=128` — SD-native resolution, 16 tiles/image.
- `ATTACK_METHOD` / `EPSILON` — poison strength in the demo (`fgsm` fast, `pgd` stronger/subtler).
- `K_SENSITIVITY` — detector threshold `median + k·MAD`; lower = more sensitive.
- `REJECT_FRAC` — fraction of flagged tiles that quarantines a whole image.

## Limitation & research direction

The score keys on **high-frequency** energy (which FGSM/PGD inject); a low-frequency or smoothed
poison could slip through. Next steps toward a publishable study: a detection-vs-epsilon sweep to
find where recall collapses; a small learned per-tile classifier (or the paper's real ReciproCAM
response) compared on that sweep; and closing the loop by fine-tuning on gated vs un-gated data to
measure the downstream difference the gate makes.

# SD-Noise-Gate

A **pre-training noise/poison detection gate** for Stable Diffusion training data.

## What this folder is

The user is learning ML and studying Stable Diffusion. In `../GradCAM-Attack/` they built a
tiled, batched, gradient-free **adversarial-noise detector**: cut an image into a grid of tiles,
score each tile's high-frequency energy, and flag the tiles that carry injected noise (FGSM/PGD).

This folder takes that detector and puts it to work **as a gate in front of the Stable Diffusion
training pipeline**. Diffusion training is `image -> VAE.encode -> add diffusion noise -> UNet`.
If someone poisons the training images (adversarial / backdoor / Nightshade-Glaze style
perturbations), the model learns from corrupted data. The gate inspects every incoming image
*before* it is encoded to latents and rejects or flags the tampered ones.

## The three inputs it stitches together

1. **`../GradCAM-Attack/tiled_noise_attack_detection.ipynb`** — source of `tile_image`, the
   per-tile high-frequency-energy detector, and the FGSM/PGD attack used to make poisoned samples.
2. **The HuggingFace Stable Diffusion intro notebook** (`../Stable-Diffusion/`) — the training /
   inference pipeline (VAE, scheduler `add_noise`, UNet) that the gate protects.
3. **ViT-ReciproCAM, arXiv:2310.02588** — motivates a **gradient-free, batchable** saliency /
   inspection step: no per-image backprop means a whole batch of images (or tiles) clears in one
   forward pass, so the gate is cheap enough to run on every training image.

## Files

- `sd_training_noise_gate.ipynb` — the notebook. Runs on Kaggle/Colab **GPU T4 x2**, Internet On
  (falls back to a handful of web sample images if no dataset is mounted). Not executed locally
  (no GPU / large model downloads).

## Key design points

- **Detector is model-free** (high-frequency energy per tile) so the gate has no training of its
  own and adds almost no cost.
- **Calibrate once on a trusted clean set**, then scan incoming (possibly poisoned) images against
  that fixed threshold — this is the deployable pattern, not per-image self-calibration.
- **Verdict is per-image**: an image is rejected if too many of its tiles are flagged; flagged
  tiles also localize *where* the poison is.
- The real SD VAE step is **optional and guarded** (try/except) so the notebook runs even without
  the diffusers weights.

## Knobs

- `GRID` — tile granularity (finer = better localization, more tiles).
- `K_SENSITIVITY` (`k`) — detector threshold `median + k*MAD`; lower = more sensitive.
- `REJECT_FRAC` — fraction of flagged tiles that gets a whole image rejected.
- `ATTACK_METHOD` / `EPSILON` — poison strength used in the demo (`fgsm` fast, `pgd` stronger).

## Next steps (research log)

- Swap the HF-energy score for a small learned per-tile classifier to catch low-frequency /
  subtle poisons that high-frequency energy misses.
- Plug the paper's real ViT + ReciproCAM in as the gradient-free saliency backbone and compare.
- Wire the gate into an actual `datasets` / `DataLoader` collate step so it filters a real
  training stream.

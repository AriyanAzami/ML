# 04-Adversarial-Attacks — adversarial attacks, localization, and tiled detection

Progression of notebooks, each building on the previous one:

1. [`gradcam_fgsm_attack.ipynb`](gradcam_fgsm_attack.ipynb) — attack **one image** with FGSM
   and watch Grad-CAM's attention shift before/after.
2. [`gradcam_attack_localization.ipynb`](gradcam_attack_localization.ipynb) — extend to a
   10-animal test set and ask *where* the attack changed the model's attention.
3. [`gradcam_tiled_batch_attack_detection.ipynb`](gradcam_tiled_batch_attack_detection.ipynb) —
   cut images into a tile grid and score tiles **in one GPU batch** (2×T4), so detection
   scales to many images.
4. [`tiled_noise_attack_detection.ipynb`](tiled_noise_attack_detection.ipynb) — **the main
   result.** Attack an irregular, non-rectangular region (FGSM or PGD), then scan the image
   tile-by-tile with a model-free high-frequency-energy score calibrated on clean tiles
   (`median + k·MAD`). Grades itself with tile-level precision / recall / IoU against the
   true attacked region.

Key conventions (shared by all notebooks and by `../05-Noise-Gate/`):

- Images are `[1,3,SIZE,SIZE]` float tensors in **[0,1] pixel space**; model-specific
  normalization happens inside the model call, so noise budgets are measured in real pixels.
- `tile_image(x)` cuts `[1,3,SIZE,SIZE]` into a row-major `[GRID*GRID, 3, TILE, TILE]` stack.
- Detector threshold: `median + k·1.4826·MAD` over trusted clean tiles (default `k=3`).

Runs on **Kaggle, GPU T4 ×2, Internet on**.

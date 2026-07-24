# 07-Attack-Repair — can a cheap frequency detector find *where* an attack hit?

**Measured answer: it finds broadband attacks, misses low-frequency ones, and the obvious fix
doesn't work.** This folder tests the prediction made in [`../06-Attack-Benchmark/`](../06-Attack-Benchmark/)
about the tile gate in [`../05-Noise-Gate/`](../05-Noise-Gate/), and reports a **negative result**.

| detector | k | HF-attack IoU | LF-attack IoU | clean FP |
|---|---|---|---|---|
| HF only (the gate) | 3 | 0.32 | **0.05** | 16% |
| multi-band | 3 | 0.36 | **0.24** | **47%** |
| multi-band | 6 | 0.06 | **0.06** | 12% |

1. **The benchmark's prediction holds.** The gate localizes a broadband attack (IoU 0.32) and is
   blind to an equal-energy low-frequency one (**0.05**).
2. **Widening the HF band into a multi-band bank does not fix it.** It *looks* like a rescue
   (0.05 → 0.24) but only by flagging **47% of clean tiles**. At a comparable false-positive rate
   the advantage is gone: **0.06 vs 0.05**. The gain was bought with false positives.
3. **Why, and why it's structural.** Natural images are non-stationary. Median band energy varies
   **~6× between clean photos** (0.0048 → 0.0281) and more *within* a photo between sky and
   foliage. A bounded perturbation lifts local energy ~2.5×. **The confound is bigger than the
   signal**, so no threshold on hand-designed band energy separates them.

This is the evidence for the **learned per-tile classifier** that the
[`../05-Noise-Gate/`](../05-Noise-Gate/) research log already lists as the next step — it turns a
hunch into a measured claim. A promising untested route is a **VAE reconstruction residual**:
encode→decode projects onto SD's learned natural-image manifold, so off-manifold perturbations
shouldn't survive the round trip — a learned prior instead of a hand-picked band.

## What is solid: three verified fixes

These hold regardless of the scoring statistic, and they correct real defects:

1. **No tiling.** Blurring is translation-invariant, so blur the *whole* image once and `avg_pool`
   the residual — the same statistic, without the copy. Tiling first zero-pads every tile,
   fabricating a black neighbour; on interior tiles that **inflates the σ=8 band by 24×**. This was
   a real bug in this folder's first version, and it corrupted precisely the low-frequency bands
   the folder exists to measure.
2. **Separable convolution.** The Gaussian is an outer product, so a 49×49 2D conv becomes two 1D
   convs (~25× fewer FLOPs). Verified exact to ~1e-6.
3. **Native batching.** `[N,3,H,W]` in one pass — no `unfold`, no `ThreadPoolExecutor`, no
   multi-GPU chunking. Timed live in the notebook (on CPU: ~48 s → ~1 s for 12 images).

Bonus: because the band maps are computed at full resolution, **the grid is free**. Pool at 4×4 or
32×32 for the same cost, or use a sliding window for arbitrary shapes.

## Arbitrary shapes

Dropping tiles makes the score **per-pixel**, so nothing forces a rectangle. A sliding window plus
morphological open/close recovers any shape — blob, ring, thin scribble — and the notebook shows
it on all three. **But measured honestly it does not work**: threshold picked on a dev image and
evaluated held-out gives **IoU 0.05–0.16 at ~15% false positives**. The machinery is sound; the
statistic underneath is too weak. The failure is shown, not hidden.

## The notebook

**[`attack_localize_and_repair.ipynb`](attack_localize_and_repair.ipynb)** — Kaggle **GPU T4,
Internet On**; falls back to web sample images, and the SD step is guarded so it runs without the
weights. Sections: tile-free detector → live sanity checks (separability, speed, the 24× padding
bug) → attack zoo at equal RMS → **the money figure: IoU vs false positives** → arbitrary-shape
localization → SD inpaint repair.

Repair is demonstrated on the **ground-truth mask**, deliberately: the measured detector isn't good
enough to drive an inpainter, and wiring a 15%-FP mask into one would repaint healthy image.

### Knobs

- `IMAGE_SOURCE` — `"web"` or `"folder"` (your Kaggle dataset via **+ Add Input**).
- `RMS` — perturbation energy, held equal across attacks.
- `SIGMAS` — the blur scales defining the band bank.

## Methodology notes (learned the hard way here)

Three mistakes made in this folder, each of which produced a confidently wrong conclusion:

- **Compare at matched false-positive rate.** A fixed threshold flatters whichever detector flags
  more. This is what made multi-band look like a fix.
- **Equalize perturbation RMS across attacks.** The first low-frequency attack was 4× weaker in RMS
  than the broadband one, so "LF is harder" was partly just "LF was smaller."
- **Never pick thresholds per-image on the test case.** Oracle thresholds inflated a real IoU of
  ~0.1 into an apparent 0.47.

## References

- Attack zoo and the original prediction — [`../06-Attack-Benchmark/README.md`](../06-Attack-Benchmark/README.md).
- ViT-ReciproCAM ([arXiv:2310.02588](https://arxiv.org/abs/2310.02588)) — the gradient-free,
  forward-only scoring idea behind the cheap scan.

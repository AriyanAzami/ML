# SD-Attack-Repair — locate the hardest attack, then let Stable Diffusion fix that part

This folder closes the loop started in the other two: [`../Attack-Benchmark/`](../Attack-Benchmark/)
laid out the attack zoo and predicted *where the tile gate fails*, and
[`../SD-Noise-Gate/`](../SD-Noise-Gate/) is the gate itself (high-frequency energy per tile). Here
we take the **one attack that gate is blind to**, upgrade the detector just enough to **localize**
it, and hand the located region to **Stable Diffusion inpainting** to repair it.

```
image --> [ multi-band tile scan ] --clean--> keep
                 |
                 +--attacked tiles--> mask --> SD inpaint --> repaired image
```

## Why the low-frequency attack is the hardest one *for this detector*

The gate scores each tile by `hf_energy = mean(|tile - blur(tile)|)`. Against the zoo:

- **FGSM / PGD / Square** inject *broadband, high-frequency* noise → loud in that score → easy.
- **C&W / DeepFool / FAB** are minimal-norm but still leave some high-frequency trace → subtle.
- **Low-frequency / Nightshade-Glaze** hides its energy in *low* spatial frequencies. `|tile -
  blur(tile)|` barely moves, so the gate is **structurally blind** — not just under-tuned. That is
  the hardest case, so it is the one this folder attacks and defends against.

The notebook builds this poison by taking PGD and **low-pass filtering the perturbation at every
step**, forcing the injected energy into the low band while keeping it inside an L∞ `epsilon`
budget and confined to a random irregular blob.

## The best method to still find *where* it hit: a multi-band spectral residual

Instead of a single high-frequency band, use a small **bank of octave band-pass filters**
(differences of Gaussian blurs, fine → coarse). Calibrate a robust `median + k·MAD` baseline
**per band** on trusted-clean tiles, z-score every incoming tile per band, and flag it if **any**
band exceeds `k`.

- The gate's original HF score is exactly **band 0** of this bank — so this is a strict
  generalization, not a replacement. L∞ attacks still fire band 0; the low-frequency poison now
  fires a *lower* band.
- It stays **forward-only and calibration-only** (the ViT-ReciproCAM lesson): no gradients, no new
  training, negligible extra cost.

The flagged tiles become a mask; `StableDiffusionInpaintPipeline` regenerates only those tiles and
copies the rest through, so a clean image passes unchanged.

## The notebook

- **[`attack_localize_and_repair.ipynb`](attack_localize_and_repair.ipynb)** — one short,
  self-contained pass: tiler → HF vs. multi-band detectors → low-frequency poison → **IoU
  comparison** (HF misses, multi-band localizes) → SD inpaint repair. Kaggle **GPU T4 ×2, Internet
  On**; falls back to web sample images with no dataset, and the SD step is guarded so it runs even
  without the diffusers weights (it then just shows the mask it *would* repair).

### Knobs

- `IMAGE_SOURCE` — `"web"` (built-in samples) or `"folder"` (your Kaggle dataset via **+ Add Input**).
- `EPSILON` — L∞ budget of the low-frequency poison.
- `K_SENSITIVITY` — per-band threshold `median + k·MAD`; lower = stricter.
- `_SIGMAS` — the blur scales that define the band bank (fine → coarse).

## Honest limitation

Natural images carry lots of genuine low-frequency variation, so the coarse bands are noisier than
the HF band — expect more false positives there. That is the motivation, already noted in the
[`../SD-Noise-Gate/`](../SD-Noise-Gate/) research log, for a small **learned per-tile classifier**
as the next step beyond hand-designed frequency bands.

## References

- AutoAttack / attack zoo — see [`../Attack-Benchmark/README.md`](../Attack-Benchmark/README.md).
- ViT-ReciproCAM ([arXiv:2310.02588](https://arxiv.org/abs/2310.02588)) — gradient-free, batchable
  scoring behind the cheap forward-only scan.

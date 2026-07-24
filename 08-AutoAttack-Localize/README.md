# 08-AutoAttack-Localize — localizing AutoAttack noise on real photographs

[`../07-Attack-Repair/`](../07-Attack-Repair/) reported a negative result: a cheap frequency
detector could not localize an attack (IoU 0.05–0.16 at ~15% false positives). This folder beats
that — but **not** by the route its own first version predicted, and the gap between the two is the
most useful thing in here.

## Two wrong predictions, both corrected by measurement

**v1** was designed on synthetic 1/f images and predicted a contrast-invariant **spectral slope**
statistic would solve localization (synthetic AUC 0.55 → 0.95). On real photographs it scored
**0.55 — chance**.

**v2** diagnosed that as a ground-truth error: v1 scored against the region *drawn* for the attack,
and its RMS table (0.0201 in-region against 0.0314 saturated) suggested only ~40% of the region
carried perturbation. Predicted worth of the fix: ~0.13 AUC. **Measured worth: 0.000.** Support is
**99%** of the region, and SRM scores **0.696 either way**. APGD's δ is *graded in amplitude*, not
*absent* — a distinction a single RMS number cannot resolve.

Both errors came from reasoning about data instead of measuring it: a synthetic proxy in v1, a
summary statistic in v2. The self-checking output added in v2 is what caught v2's own mistake — the
cell written to demonstrate the ground-truth gap printed both numbers and showed there wasn't one.

### What the prototype lesson actually is

1/f images are spatially **stationary**; one median and MAD describe the whole image. Real
photographs are not — sky and foliage within one frame differ more than two photographs do. The
prototype had removed the exact confound the statistic was built to survive. **A synthetic
validation set that lacks the confound you are trying to defeat will confirm anything.**

## What survives: measured on this dataset

Per-pixel AUC against the perturbation support, pooled over 5 images:

| feature | APGD-CE / APGD-T / masked-PGD | Square | smooth L2 | low-freq |
|---|---|---|---|---|
| energy (`07`'s gate, 2-sided) | 0.39–0.40 | 0.41 | 0.44 | 0.45 |
| `07` gate as actually run (1-sided) | 0.60 | 0.59 | 0.51 | 0.50 |
| slope (v1's headline) | 0.55 | 0.56 | 0.44 | 0.45 |
| **SRM** | **0.69–0.70** | 0.45 | 0.45 | 0.45 |
| **chroma decorrelation** | 0.65–0.66 | **0.70** | **0.52** | 0.41 |

IoU at a 5% clean false-positive rate — the metric `07` reported, where it scored 0.05–0.16 **at
15%**:

| detector | APGD-CE | Square | smooth L2 | low-freq |
|---|---|---|---|---|
| `07` gate | 0.022 | 0.023 | 0.026 | 0.026 |
| SRM | **0.142** | 0.023 | 0.022 | 0.020 |
| chroma | 0.122 | **0.261** | **0.126** | 0.007 |

**6× better than `07` on L∞ and 11× on Square, at a third of the false-positive budget.** Real, and
far short of solved — nothing here is accurate enough to drive an inpainter.

### Cross-channel decorrelation is the one genuinely new thing

Attacks perturb R, G and B independently; natural fine detail is luminance-dominated. So the chroma
part of the fine residual is nearly pure attack. Two consequences, both in the output:

- It is the **only feature that catches Square Attack** (0.70 vs 0.45 for SRM). Square is
  piecewise-*constant* over patches and adds no high-frequency residual at all.
- It is the **only feature whose attack shift exceeds its between-image spread** (3.25 vs 2.57).
  Every other feature reads `confound > signal` — no single global threshold can work for it. That
  is `07`'s original diagnosis, now measured per feature.

**No detector wins everywhere**, so SRM and chroma are reported separately; `mean(SRM, chroma)` is
middling at both jobs (0.135 / 0.161) rather than better than either.

### Two things that did not work

**The fitted fusion** — a leave-one-image-out logistic regression fitted on APGD-CE — beat the best
single feature on **1 of 6** scored attacks. Kept in the notebook as a measured negative.

**FAB-T is excluded from scoring**, not because it defeats the detector but because it lands
`max|δ| = 0.02/255` — roughly 1/400 of the budget, an empty support. Scoring a localizer on an
image carrying no perturbation is meaningless; that is a fact about minimum-norm attacks.

## Still unsolved: the low-frequency poison

Nothing exceeds 0.48, and chroma is actively *worse* than chance on it (0.41). `06` predicted it,
`07` measured it, this folder fails against it too. It remains the only attack in the zoo with no
detector above chance.

The untested route from `07` still stands — a **VAE reconstruction residual**, projecting onto
Stable Diffusion's learned natural-image manifold. Chroma suggests a cheaper first try: a
**low-frequency chroma** statistic, since a smooth poison must still perturb the colour channels
independently.

## The notebook

**[`autoattack_noise_localization.ipynb`](autoattack_noise_localization.ipynb)** — Kaggle,
**GPU T4 ×2, Internet On**, ~10–15 min. 5 random images, random non-rectangular regions
(blob / ring / scribble / wedge), real `torchattacks` AutoAttack components (APGD-CE, APGD-T,
FAB-T, Square) crafted against a ResNet-50, plus a properly-confined masked PGD and RMS-matched
smooth-L2 and low-frequency controls.

Sections: images → regions → victim model and attacks → the six-feature detector → attack zoo with
**both ground truths** → invariance check (between-image spread vs measured attack shift) →
**per-pixel ROC-AUC** → **the fusion that does not transfer** → **IoU vs clean false positives** →
raw / attacked-area / detection figure → YOLO transfer damage.

### Protocol, inherited from `07`'s mistakes

- **Per-pixel ROC-AUC**, threshold-free, pooled across images.
- **IoU only at a false-positive rate calibrated on clean images.**
- **Leave-one-image-out**, fitted on APGD-CE alone; every other attack is *transfer*.
- **RMS-matched controls**, so nothing wins on raw energy.
- Baseline is the **one-sided** `median + k·MAD` gate `07` actually deployed.
- Claims in the output are **self-checking**: each closing message tests its own assertion against
  the numbers and prints the contradiction if it does not hold.

### Fixed in v2

- `torchattacks` returns tensors still attached to its graph — this crashed section 9. Now detached.
- `permute()` leaves a non-contiguous array and `ultralytics` asserts on it (`Image not
  contiguous`), which killed the YOLO cell on the first successful run. Now `ascontiguousarray`.
- FAB-T produced an empty support and turned every aggregate into `nan`. Attacks with no
  perturbation to find are now excluded by measurement, with the reason printed.
- The invariance table applied `max/min` to a **signed** feature and printed a meaningless `0.03x`.
  It now compares between-image range against the measured attack shift, in the same units.
- The fooling rate was measured in a **separate forward pass** from the clean prediction, which
  reported 5/5 fooled even for FAB-T, whose perturbation is `max|δ| = 0.02/255` — i.e. nothing.
  Clean and attacked are now scored in one pass, and the top-1 margin is printed alongside.

## Knobs

- `N_IMAGES` — 5 by default. The most valuable thing to raise.
- `EPS` — 8/255, the AutoAttack / RobustBench standard L∞ budget.
- `SIGMA_L` — sliding-window radius for every local statistic.
- `AA_NAMES` — append `"AutoAttack"` to run the full ensemble instead of its components.
- `SHOW` / `TRAIN_ON` — which attack the final figure displays, and which the fusion is fitted on.

## Measured but not shipped

**Content-conditional normalization** — bin pixels by an attack-free content proxy (coarse-band
energy) and robust-z within each bin, rather than against one global per-image median. This
directly targets the non-stationarity in finding 2 and measured a real gain at low false-positive
rates offline (SRM IoU@1%FP **0.23 → 0.52**), but was a wash inside the fusion. Left out rather
than ship complexity on mixed evidence; worth a proper test.

**Morphological open/close** on the thresholded mask changed IoU by less than 0.01. Not included.

## References

- Croce & Hein, *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse
  Parameter-free Attacks* (AutoAttack), ICML 2020 —
  [arXiv:2003.01690](https://arxiv.org/abs/2003.01690).
- Kim, *Torchattacks: A PyTorch Repository for Adversarial Attacks* —
  [arXiv:2010.01950](https://arxiv.org/abs/2010.01950).
- Liu et al., *Detecting Adversarial Examples Based on Steganalysis* — SRM residuals as
  adversarial-detection features, [arXiv:1806.09186](https://arxiv.org/abs/1806.09186).
- Lorenz et al., *Detecting AutoAttack Perturbations in the Frequency Domain*, ICML 2021 workshop —
  [arXiv:2111.08785](https://arxiv.org/abs/2111.08785). Detects *whether* an image is attacked;
  this folder asks the harder question of *where*.
- ViT-ReciproCAM — [arXiv:2310.02588](https://arxiv.org/abs/2310.02588), the forward-only,
  batchable scoring idea the detector inherits (see [`../Resources/`](../Resources/)).

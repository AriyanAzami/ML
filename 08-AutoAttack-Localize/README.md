# 08-AutoAttack-Localize — localizing AutoAttack noise on real photographs

[`../07-Attack-Repair/`](../07-Attack-Repair/) reported a negative result: a cheap frequency
detector could not localize an attack (IoU 0.05–0.16 at ~15% false positives). This folder beats
that — but **not** by the route its own first version predicted, and the gap between the two is the
most useful thing in here.

## v1 predicted the wrong fix, and running it said so

v1 was designed on synthetic 1/f images. It argued that `07` failed because energy is not
contrast-invariant, and that a **spectral slope** statistic (`e2/e1`, in which contrast cancels)
would fix it. On synthetic images that looked decisive: pooled AUC 0.55 → 0.95.

On real photographs from this dataset it scored **AUC 0.55 — chance** — and the whole zoo landed at
IoU ≤ 0.15. Taking that apart produced the three findings below.

### 1. The ground truth was wrong, and it cost the most

Every score was computed against the **region drawn** for the attack. But AutoAttack spends its
budget where the classifier is sensitive, not uniformly. The run's own table shows APGD-CE with an
in-region RMS of **0.0201** against **0.0314** for a saturated perturbation — so only ~40% of the
region carried any perturbation at all. The detector was being charged for not flagging pixels that
contain no signal.

Scoring against the **effective support** (`|δ| > ε/4`) instead, on the same images and the same
attack, moves SRM from **0.84 to 0.97**. v2 reports both, and prints the gap so the size of the
error stays visible.

### 2. The synthetic prototype removed the exact confound it was meant to survive

1/f noise images are spatially **stationary** — one median and MAD describe the whole image. Real
photographs are not: sky and foliage within one frame differ more than two photographs do. On top
of that, the fine scale of a compressed, resized photo is already noise-like (clean slope 0.65–1.75
against 3.0 for white noise), so there is little headroom for added noise to move it.

**A synthetic validation set that lacks the confound you are trying to defeat will confirm
anything.** That is the transferable lesson, and it is the same class of mistake `07` documents.

### 3. What actually works: cross-channel decorrelation

Every attack here perturbs R, G and B **independently**, while natural fine detail is
luminance-dominated and strongly channel-correlated. So the chroma component of the fine residual
is nearly pure attack. Measured on real dataset images:

| feature | L∞ (APGD/PGD) | Square | smooth L2 | low-freq |
|---|---|---|---|---|
| energy (`07`'s gate) | ~0.51 | 0.43 | 0.41 | 0.46 |
| slope (v1's headline) | 0.83 | 0.51 | 0.45 | 0.48 |
| **SRM** | **0.97** | 0.51 | 0.49 | 0.49 |
| **chroma decorrelation** | **0.96** | **0.83** | **0.90** | 0.47 |

Chroma is the only feature that catches the attacks that are **not white noise**. Square Attack is
piecewise-*constant* over patches and adds no high-frequency residual at all, which is why every
residual-based feature scores it at chance.

The recommended detector is **`mean(SRM, chroma)` with no fitting whatsoever**: AUC ~0.97 and
IoU 0.35–0.62 at a 5% clean false-positive rate on the L∞ attacks, against `07`'s 0.05–0.16 at 15%.

### The learned fusion is kept because it fails

A leave-one-image-out logistic regression fitted on APGD-CE **beat none of the single features it
is built from** on any other attack. Five images is thin, but it is a useful corrective to the
assumption that learning a combination must help — and it is why the shipped detector does no
fitting at all.

## Still unsolved: the low-frequency poison

Nothing here exceeds ~0.49 on it. `06` predicted it, `07` measured it, this folder fails against it
too — now with better instruments and a clearer idea why: a poison hiding in low frequencies is
spectrally closest to natural image content. It is now the *only* attack in the zoo with no
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

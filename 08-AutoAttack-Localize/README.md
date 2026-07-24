# 08-AutoAttack-Localize — the statistic that finally localizes the attack

[`../07-Attack-Repair/`](../07-Attack-Repair/) reported a **negative result**: a cheap frequency
detector could not localize an attack (IoU 0.05–0.16 at ~15% false positives). It also diagnosed
*why*, and that diagnosis turns out to contain the fix.

> Median band energy varies **~6× between clean photos** and more *within* one photo between sky and
> foliage, while a bounded perturbation lifts local energy only **~2.5×**. The confound is bigger
> than the signal.

**The problem was never the filter bank — it was that energy is not contrast-invariant.** So this
folder scores the **shape** of the local spectrum instead of its scale.

## The statistic

Per pixel, over a sliding window: the ratio of **second-difference energy to first-difference
energy**, `e2/e1`. Contrast appears in both numerator and denominator and cancels, so a dark sky and
a bright textured wall give the same number while additive noise moves it. For white (L∞) noise the
ratio is **exactly 3.0**; natural content sits well below.

Two design points came out of measurement rather than intuition:

1. **The score must be two-sided.** L∞ noise makes the local spectrum *flatter* than natural; smooth
   L2 noise (C&W/DeepFool-like) makes it *steeper*. A one-sided detector fitted on L∞ scores smooth
   attacks at AUC ≈ 0.34 — that is not failure, it is **inversion**. Scoring `|deviation|` from the
   image's own median catches both directions and costs almost nothing on the L∞ case.
2. **The learned fusion rediscovers the same thesis.** Free to weight five features, a 6-parameter
   logistic regression assigns large *opposite-sign* weights to SRM energy and plain band energy —
   a difference of logs, i.e. a **ratio**, a scale-invariant spectral-shape statistic it was never
   told to build. The notebook checks this rather than asserting it, and says so if a given run
   does not reproduce it.

## The notebook

**[`autoattack_noise_localization.ipynb`](autoattack_noise_localization.ipynb)** — Kaggle,
**GPU T4 ×2, Internet On**, ~10–15 min.

5 random images from my
[obstacle-detection dataset](https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset),
random non-rectangular attack regions (blob / ring / scribble / wedge), and real `torchattacks`
**AutoAttack components** — APGD-CE, APGD-T, FAB-T, Square — crafted against a ResNet-50, plus a
properly-confined masked PGD and RMS-matched smooth-L2 and low-frequency controls that AutoAttack
cannot produce.

Sections: images → regions → victim model and attacks → the five-feature detector → **the
invariance claim verified on the actual photos** → attack zoo → **per-pixel ROC-AUC table** →
**leave-one-image-out learned fusion** → **money figure: IoU vs. clean false positives** → the
raw / attacked-area / detection figure → YOLO damage from the transferred perturbation.

### Honest protocol, inherited from `07`'s mistakes

- **Per-pixel ROC-AUC** as the threshold-free primary metric, **pooled across images** — the hard
  version, since a per-image AUC hides the calibration problem that broke the energy statistic.
- **IoU only at a false-positive rate calibrated on clean images.** A fixed threshold proves
  nothing; any detector can raise IoU by flagging more pixels.
- **Leave-one-image-out**, fitted on **APGD-CE alone**. Every other attack is reported as
  *transfer*, so no number comes from training on its own test attack.
- **RMS-matched controls**, because `07` once concluded "low-frequency is harder" from an attack
  that was merely 4× weaker.
- The baseline curve is the **one-sided** `median + k·MAD` gate `07` actually deployed, not a
  two-sided variant it never ran.

### Two caveats stated in the notebook

- **AutoAttack has no masked variant.** δ is optimized over the whole image and confined to the
  region afterwards, so the noise *statistics* under test are genuine AutoAttack output but the
  printed fooling rate is a **lower bound**. `masked-PGD`, which applies the mask at every step, is
  the fair number for region-confined attack strength.
- The perturbation is crafted on ResNet-50, so its effect on YOLO is **transfer** — which is the
  realistic threat model, since an attacker poisoning a scraped dataset does not know the
  downstream model.

## Status of the numbers

**The notebook has not been run yet** — it needs the Kaggle GPU and the dataset mounted. The design
above is not guesswork: it was settled on a CPU prototype over synthetic 1/f images with a
deliberate 6× contrast spread, reproducing `07`'s confound. On that prototype, pooled AUC on L∞
noise went from **0.55** (energy) to **0.95** (slope) to **0.98** (two-sided LOIO fusion), and the
one-sided fusion's **0.34 on smooth L2** is what produced design point 1.

Those are synthetic-image numbers and are **not** claims about the dataset. Fill this table in from
the notebook's own output once it has run on Kaggle:

| detector | APGD-CE | Square | L2-smooth | low-freq | (IoU @ 5% clean FP) |
|---|---|---|---|---|---|
| 07 gate (energy, 1-sided) | | | | | |
| slope (contrast-free) | | | | | |
| learned fusion (LOIO) | | | | | |

## Knobs

- `N_IMAGES` — 5 by default; the fusion has 6 parameters, so raising this is the cheapest real
  improvement available.
- `EPS` — 8/255, the AutoAttack / RobustBench standard L∞ budget.
- `SIGMA_L` — sliding-window radius for every local statistic.
- `AA_NAMES` — append `"AutoAttack"` to also run the full ensemble instead of its components.
- `SHOW` / `TRAIN_ON` — which attack the final figure displays, and which one the fusion is
  fitted on.

## Where this leaves the research line

**The low-frequency attack is still the weak spot** — the hardest column in every table. That is
progress over `07`, not a solved problem: a poison hiding in low frequencies is spectrally closest
to natural image content, which is exactly why it is hard. The untested route from `07` still
stands — a **VAE reconstruction residual**, where encode→decode projects onto Stable Diffusion's
learned natural-image manifold, giving a learned prior over image content instead of a hand-picked
band.

Then close the loop back into [`../05-Noise-Gate/`](../05-Noise-Gate/): swap this score in for the
HF gate and re-measure quarantine precision and recall on the SD training path.

## References

- Croce & Hein, *Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse
  Parameter-free Attacks* (AutoAttack), ICML 2020 —
  [arXiv:2003.01690](https://arxiv.org/abs/2003.01690).
- Kim, *Torchattacks: A PyTorch Repository for Adversarial Attacks* —
  [arXiv:2010.01950](https://arxiv.org/abs/2010.01950).
- Liu et al., *Detecting Adversarial Examples Based on Steganalysis* — SRM residuals as
  adversarial-detection features, [arXiv:1806.09186](https://arxiv.org/abs/1806.09186).
- Lorenz et al., *Detecting AutoAttack Perturbations in the Frequency Domain*, ICML 2021 workshop —
  [arXiv:2111.08785](https://arxiv.org/abs/2111.08785). Detects *whether* an image is attacked from
  its spectrum; this folder asks the harder question of *where*.
- ViT-ReciproCAM — [arXiv:2310.02588](https://arxiv.org/abs/2310.02588), the forward-only,
  batchable scoring idea the detector inherits (see [`../Resources/`](../Resources/)).

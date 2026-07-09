# Attack-Benchmark — stress-testing the tile noise detector against an attack zoo

So far the detector in [`../GradCAM-Attack/`](../GradCAM-Attack/) and
[`../SD-Noise-Gate/`](../SD-Noise-Gate/) has only been tested against **FGSM** and **PGD** — both
L∞, gradient-based attacks that inject **broadband high-frequency noise**, which is exactly what
the `median + k·MAD` high-frequency-energy detector is built to catch. This folder tests it against
**harder and different attacks** on random images, and measures *where it starts to fail*.

## The attack ladder

| Attack | Type | Norm | Why it's harder | Expected vs. HF detector |
|---|---|---|---|---|
| **FGSM** | white-box, 1 step | L∞ | baseline | easy (loud HF noise) |
| **PGD** | white-box, iterative | L∞ | tune step/steps/restarts | easy (HF noise) |
| **DeepFool** | white-box, iterative | L2 | minimal perturbation to nearest boundary | harder (subtler) |
| **C&W** | white-box, optimization | L2 | minimizes perturbation size; smooth, low-energy | **much harder** |
| **Square Attack** | **black-box**, query-based | L∞ | no gradients; random localized squares | localized → tile-detectable |
| **FAB** | white-box | L∞/L2/L1 | minimal-norm boundary attack | hard (minimal noise) |
| **AutoAttack** | **ensemble** (below) | L∞/L2 | parameter-free worst case | the true benchmark |
| **Low-frequency / Nightshade-Glaze** | poisoning | perceptual | energy hidden in low frequencies | **evades HF detector** |

## AutoAttack vs. FGSM/PGD (the headline comparison)

AutoAttack is not one attack — it is an **ensemble of four parameter-free attacks**; an image is
"robust" only if *all four* fail:

1. **APGD-CE** — Auto-PGD on cross-entropy, with an automatic step-size schedule + momentum
   (PGD without the hand-tuning; strictly stronger than the PGD we ran).
2. **APGD-T** — targeted Auto-PGD on the DLR loss (resistant to gradient masking).
3. **FAB-T** — minimizes the *size* of the perturbation, not just crosses the boundary.
4. **Square Attack** — a **gradient-free black-box** attack; catches models that only *looked*
   robust because their gradients were broken ("gradient masking").

| | FGSM | PGD | AutoAttack |
|---|---|---|---|
| Steps | 1 | many | many, auto-scheduled |
| Hyperparameter tuning | none | you tune step/steps/restarts | **none (parameter-free)** |
| Gradient-masking robust | no | no | **yes** (includes a black-box attack) |
| Reliability of result | over-estimates robustness | decent, tuning-sensitive | **worst-case standard** (RobustBench) |
| Speed | fastest | medium | slowest |
| Strength | weakest | strong | **≥ PGD, always** |

**Why this matters for the detector:** the HF-energy score is well matched to L∞ attacks
(FGSM/PGD/APGD/Square) that inject high-frequency energy. But **L2-optimization attacks (C&W,
DeepFool, FAB)** and **low-frequency attacks** produce far less HF energy — so the detector's
recall should drop on them. Quantifying that drop is the point of this folder.

## The experiment (planned notebook)

`attack_zoo_detection.ipynb` (Kaggle, GPU T4 ×2, Internet On):

1. Pull random images (a Kaggle dataset, or web samples as fallback).
2. Run the attack zoo through the **`torchattacks`** library (one clean API for FGSM, PGD, CW,
   DeepFool, Square, FAB, AutoAttack).
3. For each attack: tile + score every image with the existing HF detector, and record
   **detection recall / ROC-AUC** and the **mean perturbation norm** (L2 and L∞).
4. **Money figure:** detection AUC vs. perturbation norm, one point per attack — a single plot
   showing which attacks the detector catches and which slip through.

This directly seeds a publishable claim: *"a cheap high-frequency tile detector reliably flags
L∞ adversarial noise but degrades against minimal-norm L2 and low-frequency attacks — motivating a
learned per-tile detector."*

## Reference

- ViT-ReciproCAM ([arXiv:2310.02588](https://arxiv.org/abs/2310.02588)) — the gradient-free,
  batchable saliency idea behind the detector (see [`../Resources/`](../Resources/)).
- AutoAttack — Croce & Hein, ICML 2020, *"Reliable Evaluation of Adversarial Robustness with an
  Ensemble of Diverse Parameter-free Attacks"* ([arXiv:2003.01690](https://arxiv.org/abs/2003.01690)).
- `torchattacks` — Kim, *"Torchattacks: A PyTorch Repository for Adversarial Attacks"*
  ([arXiv:2010.01950](https://arxiv.org/abs/2010.01950)).

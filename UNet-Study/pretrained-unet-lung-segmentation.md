# Notes — Pretrained U-Net for Lung Segmentation

**Notebook:** `pretrained-unet-lung-segmentation.ipynb`

> ⚠️ **The notebook file is currently empty (0 bytes).** It came from a placeholder
> `.xpynb` file that never had content saved. The work needs to be **re-exported
> from Kaggle** before this note can be filled in properly.

## What this was meant to cover

Lung-field segmentation from chest X-rays using a U-Net with a **pretrained
encoder backbone** (transfer learning) rather than training from scratch.

This was the natural next step after the ISIC notebooks: instead of learning all
encoder weights from random init, start from a backbone pretrained on a large
image dataset and only fine-tune — usually **faster convergence and better
results on small medical datasets**.

## To do before the meeting

- [ ] Re-download / re-export the original notebook from Kaggle.
- [ ] Replace the empty `.ipynb` and update this note with the real details
      (which backbone, dataset, and Dice/IoU scores achieved).

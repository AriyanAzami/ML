# Notes — TensorFlow U-Net (ISIC 2018), pure-Keras version

**Notebook:** `tensorflow-unet-kfold.ipynb`

> ⚠️ **The notebook file is currently empty (0 bytes).** It came from a placeholder
> `.xpynb` file that never had content saved. The work needs to be **re-exported
> from Kaggle** before this note can be filled in properly.

## What this was meant to cover

This was the **pure TensorFlow/Keras** counterpart to
[`pytorch-unet-kfold.ipynb`](pytorch-unet-kfold.md) — the same ISIC 2018
skin-lesion segmentation task and U-Net, but feeding the model with TensorFlow's
own `tf.data` pipeline instead of a bridged PyTorch `DataLoader`.

The point of having both was to **compare the two frameworks' data pipelines**
side by side on an identical task.

## To do before the meeting

- [ ] Re-download / re-export the original notebook from Kaggle.
- [ ] Replace the empty `.ipynb` and update this note with the real details.

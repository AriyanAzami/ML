# Notes — U-Net for ISIC 2018 Skin-Lesion Segmentation

**Notebook:** `pytorch-unet-kfold.ipynb`
**Task:** Binary segmentation — given a dermoscopy image, predict the lesion mask.
**Dataset:** ISIC 2018 (2,594 training image/mask pairs).
**Stack:** TensorFlow / Keras (TF 1.13) U-Net, trained on a Tesla P100 (Kaggle).

> Note on the name: despite "pytorch" in the filename, the model itself is a
> **Keras U-Net**. The "pytorch" part refers to an experiment where I fed it using
> a **PyTorch `DataLoader`** — see "Cross-framework data loading" below.

---

## What the notebook does

1. **Pairs up images and masks** by filename (`ISIC_xxxx.jpg` ↔
   `ISIC_xxxx_segmentation.png`), resized to 128×128.
2. **Defines the U-Net** — a classic encoder–decoder:
   - Encoder: repeated `Conv → Conv → MaxPool`, channels 64 → 128 → 256 → 512 → 1024.
   - Decoder: `Conv2DTranspose` upsampling, each step **concatenated with the
     matching encoder feature map** (the skip connections).
   - Final `1×1` conv with **sigmoid** → per-pixel foreground probability.
3. **Custom metrics & loss** for segmentation:
   - `dice_coef` — overlap between prediction and ground truth.
   - `iou` — intersection over union.
   - `dice_coef_loss = -dice_coef` — optimised directly.
4. **5-fold cross-validation** (`KFold`, sklearn) — train/evaluate on 5 splits,
   then report **mean ± std** of accuracy, loss, Dice, and IoU.
5. **Qualitative check** — plots original / ground-truth mask / prediction for
   random samples.

## Things I learned the hard way

- **Memory matters.** Loading all 2,594 images into one big NumPy array
  (`np.zeros([2594, 128, 128, 3])`) eats RAM fast. The fix: a **lazy dataset** that
  stores only file *paths* and reads each image in `__getitem__` — "almost 0 RAM".
- **Channels-first vs channels-last.** PyTorch wants `(C, H, W)`; Keras wants
  `(H, W, C)`. Moving the loader between frameworks meant fixing the mask's
  `expand_dims` axis (`axis=0` → `axis=-1`).
- **Dice loss vs accuracy.** Pixel accuracy is misleading when the lesion is small
  (a model predicting "all background" still scores high). Dice/IoU are the honest
  metrics for segmentation.

## Cross-framework data loading (the interesting bit)

I wanted to reuse PyTorch's convenient `Dataset`/`DataLoader` (lazy loading,
`num_workers` parallelism) but train a **Keras** model. Keras's `fit()` can't take
a PyTorch loader directly, so I wrote a small **generator bridge**:

```python
def keras_bridge(pytorch_loader):
    while True:
        for images, masks in pytorch_loader:
            yield images.numpy(), masks.numpy()  # Tensor -> NumPy for Keras
```

Then `model.fit(keras_bridge(train_loader), steps_per_epoch=len(train_loader), ...)`.

**Lesson:** framework data pipelines aren't interchangeable, but they *are*
bridgeable — both ultimately yield batches of arrays.

## Known rough edges (honest record)

- A couple of cells error on first run (`NameError: Dataset is not defined`) because
  the `torch.utils.data` import comes *after* the class that uses it — needs the
  import moved up.
- Mixes TF 1.x (`K.flatten`, `lr=` arg) — would need small changes for TF 2.x.

## Next steps

- Add data augmentation (flips/rotations) inside the dataset.
- Try a pretrained encoder backbone (see the lung-segmentation notebook).
- Compare this Keras version against a pure-PyTorch U-Net to isolate framework
  effects.

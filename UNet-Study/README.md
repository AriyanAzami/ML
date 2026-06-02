# U-Net Study

Notebooks from learning the U-Net architecture for medical image segmentation,
across both PyTorch and TensorFlow. (Originally the "TEMP Kaggle notebooks".)

| Notebook | Framework | Task |
|----------|-----------|------|
| `pytorch-unet-kfold.ipynb` | PyTorch | ISIC 2018 skin-lesion segmentation with k-fold cross-validation. |
| `tensorflow-unet-kfold.ipynb` | TensorFlow / Keras | Same ISIC 2018 task, ported to TensorFlow. |
| `pretrained-unet-lung-segmentation.ipynb` | PyTorch | Lung segmentation using a pretrained U-Net encoder. |

U-Net's encoder–decoder with skip connections is also the backbone of the diffusion
models in `../Stable-Diffusion`, so this study feeds directly into that work.

# 02-UNet-Segmentation — U-Net for medical image segmentation

Learning the U-Net encoder–decoder architecture across both PyTorch and TensorFlow.
The same architecture returns in [`../03-Diffusion-Models/`](../03-Diffusion-Models/) as the
denoising backbone of a diffusion model — predicting *noise* instead of a *mask*.

| Notebook | Framework | Task |
|----------|-----------|------|
| `pytorch-unet-kfold.ipynb` | PyTorch | ISIC 2018 skin-lesion segmentation with k-fold cross-validation. |
| `tensorflow-unet-kfold.ipynb` | TensorFlow / Keras | Same ISIC 2018 task, ported to TensorFlow. |
| `pretrained-unet-lung-segmentation.ipynb` | PyTorch | Lung segmentation using a pretrained U-Net encoder. |

U-Net's encoder–decoder with skip connections is also the backbone of the diffusion
models in `../03-Diffusion-Models`, so this study feeds directly into that work.

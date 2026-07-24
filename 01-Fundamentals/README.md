# 01-Fundamentals — TensorFlow vs PyTorch basics

Where the traineeship started: setting up both frameworks and learning where they differ.

| Notebook | What it covers |
|----------|----------------|
| [`assignment.ipynb`](assignment.ipynb) | Deep-learning fundamentals worked through in both TensorFlow and PyTorch. |

**Key takeaway:** the same model can be expressed in either framework, but the *data loading*
and *tensor layout* conventions differ — channels-first in PyTorch `(C, H, W)` vs channels-last
in Keras `(H, W, C)`. That distinction comes back in
[`../02-UNet-Segmentation/`](../02-UNet-Segmentation/), where a PyTorch `DataLoader` gets bridged
into a Keras model.

Notebooks are written for **Kaggle (GPU T4 ×2, Internet on)**.

# Resources — reference material

- [`2310.02588v1.pdf`](2310.02588v1.pdf) — **ViT-ReciproCAM: Gradient and Attention-Free
  Visual Explanations for Vision Transformer** (Byun & Lee, Intel, 2023,
  [arXiv:2310.02588](https://arxiv.org/abs/2310.02588)). Gradient-free saliency via spatial
  token masking: mask one spatial position at a time, run the tail of the network on the
  whole masked batch in a single forward pass, and read the prediction drop as the saliency
  of that position. The lesson used in this repo: **forward-only methods batch trivially**,
  which is what makes tile-level screening cheap enough to run on every training image.

- [`stable_diffusion.ipynb`](stable_diffusion.ipynb) — the Hugging Face `diffusers`
  Stable Diffusion walkthrough (pipeline usage + a hand-written inference loop exposing the
  VAE, CLIP text encoder, UNet, and scheduler). Source of the SD components that
  `../SD-Noise-Gate/` protects.

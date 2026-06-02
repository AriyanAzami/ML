# Notes — Stable Diffusion (Hugging Face `diffusers` intro)

**Notebook:** `stable_diffusion_intro.ipynb`
**Source:** the official 🤗 Diffusers Stable Diffusion walkthrough.
**Goal:** understand how text-to-image diffusion works before applying it to medical images.

---

## Part 1 — Using the pipeline (the easy way)

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               torch_dtype=torch.float16).to("cuda")
image = pipe("a photograph of an astronaut riding a horse").images[0]
```

Key knobs I learned:
- **`num_inference_steps`** — more steps = more detail (default ~50; 25 is often fine).
- **`guidance_scale`** — how strongly the image must match the prompt (7–8.5 is the sweet spot; too high = less diverse).
- **`generator` + seed** — reproducible outputs.
- **`height`/`width`** — must be multiples of 8.

## Part 2 — What Stable Diffusion actually is

It's a **latent diffusion** model: the diffusion happens in a small *compressed*
latent space, not full pixel space, which is why it's fast. Three components:

| Component | Role |
|-----------|------|
| **VAE** (autoencoder) | Compresses image ↔ latent. At inference we mainly use the **decoder**. |
| **U-Net** | Predicts the **noise** to remove at each step. Conditioned on text via cross-attention. |
| **Text encoder** (CLIP) | Turns the prompt into embeddings the U-Net can use. Frozen, not trained. |
| **Scheduler** | The denoising algorithm (PNDM, K-LMS, DPM-Solver…). |

> The VAE's 8× downsampling turns a `(3, 512, 512)` image into a `(4, 64, 64)`
> latent → ~64× less memory. That's the whole trick behind "fast on a 16GB GPU."

## Part 3 — Writing the loop by hand

The notebook also rebuilds the pipeline manually to demystify it:

1. Encode the prompt (and an empty prompt) with CLIP → `text_embeddings`.
2. Start from random latent noise.
3. **Denoising loop:** for each timestep, the U-Net predicts noise; apply
   **classifier-free guidance** (`uncond + scale·(text − uncond)`); the scheduler
   steps the latents one notch less noisy.
4. Decode the final latents with the VAE → image.

```python
noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
latents = scheduler.step(noise_pred, t, latents).prev_sample
```

## How this connects to the rest of my work

- The **U-Net here is the same architecture** I used for segmentation in
  [`../UNet-Study`](../UNet-Study/) — but it predicts *noise* instead of a *mask*.
- The **text encoder is the part medical imaging usually drops** — for CT/MRI you
  typically don't have prompts, you just learn the image distribution. That's
  exactly what the MONAI notebook does next:
  [`medical_diffusion_monai_simple.md`](medical_diffusion_monai_simple.md).

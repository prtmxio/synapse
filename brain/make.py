"""
brain/make.py — The Dream Engine

Takes a text prompt (the semantic anchor from scry.py) and generates
a new image using Stable Diffusion Turbo.

Architecture inside the pipeline:
  [Text prompt]
       ↓
  CLIP Text Encoder  →  77 × 768 token embeddings
       ↓
  UNet (denoiser)    ←  cross-attention with text tokens at every layer
       ↓  (1 step — SD-turbo is distilled to work in a single denoising step)
  Denoised latent    [64 × 64 × 4]
       ↓
  VAE Decoder        →  [512 × 512 × 3]
       ↓
  [Output image]

Why SD-turbo?
  - RTX 2050 has 4GB VRAM. Full SD 1.5 with 50 DDIM steps is slow (~30s).
  - SD-turbo is distilled via adversarial training to work in 1 step (~0.5s).
  - ~1.7GB VRAM in fp16 — leaves room alongside scry/find if needed.
  - guidance_scale=0.0 because CFG requires two UNet passes per step;
    single-step distillation bakes the guidance in — running CFG on top
    would actually degrade quality.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "../.hf_cache")
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_ID  = os.path.join(os.path.dirname(__file__), "../sd-turbo")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs")


class Maker:
    """
    Wraps SD-turbo for text-to-image generation.

    Public API (used by SynapseAgent later):
        maker.make(prompt)                    → PIL Image
        maker.make_and_save(prompt, path)     → saved file path
        maker.unload()                        → free VRAM
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        print(f"Maker loading SD-turbo on {self.device} "
              f"({self._vram_str() if self.device == 'cuda' else ''})")

        # AutoPipelineForText2Image detects the model type and loads
        # the right pipeline class. For sd-turbo it loads StableDiffusionPipeline.
        # It contains: pipe.vae, pipe.unet, pipe.text_encoder, pipe.scheduler
        #
        # torch_dtype=torch.float16:
        #   All weights stored as float16 (2 bytes per param instead of 4).
        #   SD-turbo has ~900M parameters → ~1.7GB in fp16 vs ~3.4GB in fp32.
        #   Slight precision loss is imperceptible for image generation.
        #
        # variant="fp16":
        #   Tells HuggingFace Hub to download the fp16 weight files directly
        #   instead of downloading fp32 and converting. Saves download time + disk.
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.device)

        # Disables the NSFW safety checker — it adds latency and false positives
        # on abstract/artistic outputs. Fine for a local research project.
        self.pipe.safety_checker = None

        # attention_slicing: instead of computing full attention in one shot
        # (memory spike), compute it in slices. Slight speed cost (~5%) but
        # reduces peak VRAM usage during the attention operation.
        # Important on 4GB — prevents OOM during the UNet forward pass.
        self.pipe.enable_attention_slicing()

        print(f"SD-turbo ready. VRAM in use: {self._vram_used()}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        temperature: float = 1.0,
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt:      text description — ideally the semantic anchor from scry.py
            width:       output width in pixels (must be multiple of 8)
            height:      output height in pixels (must be multiple of 8)
            seed:        set for reproducible outputs, None for random
            temperature: controls output variance (default 0.0, range 0.0–1.0)
                         0.0 → deterministic, literal interpretation of prompt
                         0.5 → moderate creative variation
                         1.0 → maximum practical variation
                         Same scale as LLM temperature: 0 = greedy, 1 = full sampling

        Returns:
            PIL Image (RGB, 512×512 by default)

        Key parameters explained:
            num_inference_steps=1:
                SD-turbo was trained with adversarial distillation to produce
                a good image in a SINGLE denoising step. Normal SD needs 50.
                At step=1, the UNet maps: pure_noise → clean_latent in one shot.

            guidance_scale=0.0:
                Classifier-Free Guidance extrapolates between conditional and
                unconditional predictions. SD-turbo was distilled WITH guidance
                baked in — running CFG on top (scale > 1) degrades quality.
                scale=0.0 means: only use the conditional prediction, no CFG.

            temperature (latent-space noise):
                After denoising, a small Gaussian perturbation is added to the
                clean latent before VAE decoding:
                  noise_std = temperature * 0.5
                  latent = latent + randn * noise_std
                This shifts the image in pixel space without touching the UNet's
                noise schedule (which would cause NaN at non-standard sigma levels).
                Range 0–1 mirrors LLM temperature: 0 = greedy, 1 = full variation.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if temperature > 0.0:
            # Run denoising normally, get clean latents back (before VAE decoding).
            # output_type="latent" stops the pipeline just before the VAE decoder.
            #
            # Why not scale input noise instead?
            #   SD-turbo was distilled to work at one specific noise level (sigma=14.6).
            #   Scaling the input beyond that pushes the UNet outside its training
            #   distribution → garbage noise predictions → NaN in fp16 → black image.
            #
            # Latent-space noise is safe because:
            #   - The denoised latent is in a well-conditioned range (~[-3, 3])
            #   - Adding small Gaussian noise here shifts the image in pixel space
            #     without breaking the UNet's arithmetic
            #   - The VAE decoder is robust to small latent perturbations
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type="latent",
                )

            latents = result.images  # shape [1, 4, H//8, W//8] — the clean latent
            noise_std = temperature * 0.5
            latents = latents + torch.randn_like(latents) * noise_std

            # Decode manually: divide by VAE scaling factor, run decoder, postprocess
            with torch.inference_mode():
                decoded = self.pipe.vae.decode(
                    latents / self.pipe.vae.config.scaling_factor
                ).sample
            image = self.pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        else:
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    width=width,
                    height=height,
                    generator=generator,
                )
            image = result.images[0]

        return image

    def make_and_save(
        self,
        prompt: str,
        output_path: str | None = None,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate and save to disk. Returns the saved file path.
        If output_path is None, saves to outputs/ with a timestamp filename.
        """
        image = self.make(prompt, width=width, height=height, seed=seed, temperature=temperature)

        if output_path is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"make_{timestamp}.png")

        image.save(output_path)
        return output_path

    def unload(self):
        """
        Free VRAM. SD-turbo holds ~1.7GB — call this before loading
        SmolVLM or running a large batch with another model.
        """
        del self.pipe
        torch.cuda.empty_cache()
        print(f"Maker unloaded. VRAM now: {self._vram_str()}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _vram_str() -> str:
        if not torch.cuda.is_available():
            return "N/A"
        free, total = torch.cuda.mem_get_info()
        return f"{(total - free) / 1e9:.1f}GB used / {total / 1e9:.1f}GB total"

    @staticmethod
    def _vram_used() -> str:
        if not torch.cuda.is_available():
            return "N/A"
        return f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Make: generate an image from a text prompt using SD-turbo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python brain/make.py --prompt "a dark coffee shop with neon signs at night"
  uv run python brain/make.py --prompt "a cat sitting on a couch" --seed 42
  uv run python brain/make.py --prompt "rainy Tokyo street" --width 768 --height 512
        """,
    )
    parser.add_argument("--prompt", required=True, help="Text description to generate from")
    parser.add_argument("--output", help="Output file path (default: outputs/make_<timestamp>.png)")
    parser.add_argument("--width",  type=int, default=512, help="Image width  (default: 512)")
    parser.add_argument("--height", type=int, default=512, help="Image height (default: 512)")
    parser.add_argument("--seed",        type=int,   default=None, help="Random seed for reproducibility")
    parser.add_argument("--temperature", type=float, default=0.0,  help="Output variance 0.0–1.0 (default 0.0 = deterministic, 1.0 = max variation).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    maker = Maker()
    saved_path = maker.make_and_save(
        prompt=args.prompt,
        output_path=args.output,
        width=args.width,
        height=args.height,
        seed=args.seed,
        temperature=args.temperature,
    )

    print(f"\n[Prompt]  {args.prompt}")
    print(f"[Saved]   {saved_path}")
    print(f"[VRAM]    {maker._vram_used()}\n")

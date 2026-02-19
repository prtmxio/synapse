"""
brain/scry.py — The Visual Cortex

Uses SmolVLM-500M (SigLIP + SmolLM2, ~1GB) to extract semantic meaning from images.

Architecture:
  [Image] → SigLIP ViT-SO400M (vision encoder)
               ↓  patch embeddings → pixel_values
          SmolLM2 language model (conditioned on vision tokens + text prompt)
               ↓
          [Text description]

Why SmolVLM-500M over LLaVA?
  - RTX 2050 has 4GB VRAM. LLaVA-7B needs ~5GB even at 4-bit. SmolVLM needs ~1GB.
  - Same conceptual architecture as BLIP-2/LLaVA: vision encoder → language model.
  - HuggingFace-native: no trust_remote_code, fully compatible with transformers 5.x.
  - 500M params on disk = ~1GB, leaves 3GB free for CLIP + SD later in the pipeline.

The key difference from what you read (BLIP-2 Q-Former):
  SmolVLM uses a pixel_shuffle (spatial merging) connector, not a Q-Former.
  Both serve the same goal: compress visual patch tokens into fewer LLM input tokens.
"""

import argparse
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "../.hf_cache")

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

CAPTION_PROMPT = (
    "Describe this image in flowing prose. Focus on the objects, setting, colors, "
    "and relationships between elements. Do not use bullet points or numbered lists. "
    "Write 3-4 complete sentences only."
)


class Scryer:
    """
    Wraps SmolVLM-500M for image-to-text inference.

    Public API (used by SynapseAgent later):
        scryer.scry(image_path)             → 50-word semantic anchor
        scryer.ask(image_path, question)    → answer to a specific question
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        print(
            f"Scryer loading on {self.device} "
            f"({self._vram_str() if self.device == 'cuda' else ''})"
        )

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,  # bfloat16: same memory as float16 but more stable range
        ).to(self.device)
        self.model.eval()

        print(f"SmolVLM-500M ready. VRAM in use: {self._vram_used()}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scry(self, image_path: str, max_tokens: int = 120) -> str:
        """
        Generate a semantic anchor — a dense caption of the image.

        Args:
            image_path: path to any image file PIL can open
            max_tokens: max new tokens to generate (120 ≈ 80-90 words)
        Returns:
            Caption string — the Semantic Anchor for downstream RAG/generation.
        """
        raw = self._infer(image_path, CAPTION_PROMPT, max_tokens)
        return self._trim_to_sentence(raw)

    def ask(self, image_path: str, question: str, max_tokens: int = 60) -> str:
        """
        Ask a specific question about the image.
        Useful for extracting structured facts: colors, counts, labels, etc.

        Example:
            scryer.ask("photo.jpg", "What objects are on the desk?")
        """
        return self._infer(image_path, question, max_tokens)

    def unload(self):
        """
        Free VRAM. Call this before loading CLIP or Stable Diffusion.
        SynapseAgent will call this to hand off VRAM between pipeline stages.
        """
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        print(f"Scryer unloaded. VRAM now: {self._vram_str()}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to_sentence(text: str) -> str:
        """Trim output to the last complete sentence so the anchor never cuts mid-phrase."""
        for end in (".", "!", "?"):
            last = text.rfind(end)
            if last != -1:
                return text[: last + 1].strip()
        return text.strip()

    def _infer(self, image_path: str, prompt: str, max_tokens: int) -> str:
        image = self._load(image_path)

        # SmolVLM uses a chat template. We build a single user turn with image + text.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # apply_chat_template converts our message list into the model's expected format
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # greedy decoding — deterministic, faster
            )

        # Trim the input tokens from the output so we only decode the new tokens
        trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.decode(trimmed[0], skip_special_tokens=True).strip()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _load(image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

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
        description="Scry: extract semantic meaning from an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python brain/scry.py brain/data/test_images/sample.jpg
  uv run python brain/scry.py photo.jpg --ask "What colors are dominant?"
  uv run python brain/scry.py photo.jpg --ask "List every object visible."
        """,
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Ask a specific question instead of generating a caption",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Max tokens to generate (default: 120 ≈ 80-90 words)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    scryer = Scryer()

    if args.ask:
        print(f"\n[Question] {args.ask}")
        answer = scryer.ask(args.image, args.ask, max_tokens=args.max_tokens)
        print(f"[Answer]   {answer}\n")
    else:
        print(f"\n[Scrying '{args.image}']")
        anchor = scryer.scry(args.image, max_tokens=args.max_tokens)
        print(f"\n--- Semantic Anchor ---\n{anchor}\n")

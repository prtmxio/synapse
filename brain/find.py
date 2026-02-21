"""
brain/find.py — The Memory Bank (Phase 2A: Similarity Scorer)

Uses CLIP (ViT-B/32) to embed images and text into a shared 512-dim vector space,
then ranks images by cosine similarity to a query.

Architecture:
                   ┌─ ViT-B/32 image encoder ─┐
  [Image / Text] → │                           │ → 512-dim L2-normalized vector
                   └─ Text Transformer         ─┘
                                   ↓
                      dot(img_vec, text_vec)   ← cosine sim (vectors are unit-length)
                                   ↓
                            similarity score

Why cosine similarity works here:
  CLIP is trained with contrastive loss to pull matching (image, text) pairs
  close together and push non-matching pairs apart — in the same 512-dim space.
  After L2 normalization, dot product == cosine similarity (range: -1 to 1).
  In practice, CLIP scores live in ~[0.15, 0.40] for real images.

Why ViT-B/32?
  - ~600MB on disk, ~300MB VRAM — leaves room for Scryer and future SD.
  - 32x32 patch size: faster but slightly less fine-grained than ViT-L/14.
  - The right tradeoff for a 4GB GPU.
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "../.hf_cache")

MODEL_ID = "openai/clip-vit-base-patch32"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class Finder:
    """
    Encodes images and text with CLIP and computes semantic similarity.

    Public API (used by SynapseAgent later):
        finder.encode_image(image_path)       → 512-dim unit vector
        finder.encode_text(text)              → 512-dim unit vector
        finder.similarity(vec_a, vec_b)       → float in [-1, 1]
        finder.rank(query, image_paths, top_k) → [(path, score), ...]
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        print(f"Finder loading CLIP on {self.device} "
              f"({self._vram_str() if self.device == 'cuda' else ''})")

        self.processor = CLIPProcessor.from_pretrained(MODEL_ID)
        self.model = CLIPModel.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
        ).to(self.device)
        self.model.eval()

        print(f"CLIP ViT-B/32 ready. VRAM in use: {self._vram_used()}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode a single image into a normalized 512-dim CLIP vector.

        The vision encoder splits the image into 32x32 patches, runs them
        through a ViT transformer, and projects the [CLS] token to 512 dims.
        L2 normalization makes the vector a unit vector on the 512-dim sphere.

        Note: transformers 5.x changed get_image_features() to return a
        BaseModelOutputWithPooling instead of a plain tensor, so we manually
        apply visual_projection on top of the pooler_output (the [CLS] token).
        """
        image = self._load(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            # vision_model gives us the ViT encoder output
            out = self.model.vision_model(**inputs)
            # pooler_output is the [CLS] token — shape [1, 768] for ViT-B/32
            # visual_projection maps 768 → 512 (the shared embedding space)
            features = self.model.visual_projection(out.pooler_output)

        return F.normalize(features, dim=-1)  # shape: [1, 512]

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode a text string into a normalized 512-dim CLIP vector.

        The text encoder is a Transformer trained jointly with the image encoder.
        Same 512-dim output space — that's what makes cross-modal search possible.

        Note: transformers 5.x changed get_text_features() to return a
        BaseModelOutputWithPooling, so we manually apply text_projection.
        """
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        with torch.inference_mode():
            # text_model gives us the Transformer encoder output
            out = self.model.text_model(**inputs)
            # pooler_output is the [EOS] token — shape [1, 512] for ViT-B/32
            # text_projection maps 512 → 512 (keeps us in the shared space)
            features = self.model.text_projection(out.pooler_output)

        return F.normalize(features, dim=-1)  # shape: [1, 512]

    def similarity(self, vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
        """
        Cosine similarity between two unit vectors = their dot product.
        Returns a plain Python float.
        """
        return (vec_a @ vec_b.T).item()

    def rank(
        self,
        query: str,
        image_paths: list[str],
        top_k: int = 5,
        query_is_image: bool = False,
    ) -> list[tuple[str, float]]:
        """
        Rank a list of images by similarity to a text or image query.

        Args:
            query:          text prompt  OR  image path (set query_is_image=True)
            image_paths:    list of image file paths to rank
            top_k:          how many top results to return
            query_is_image: if True, treat query as an image path
        Returns:
            List of (image_path, score) sorted highest-first, length = top_k.
        """
        query_vec = (
            self.encode_image(query) if query_is_image else self.encode_text(query)
        )

        scores = []
        for path in image_paths:
            try:
                img_vec = self.encode_image(path)
                score = self.similarity(query_vec, img_vec)
                scores.append((path, score))
            except Exception as e:
                print(f"  [skip] {path}: {e}")

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def unload(self):
        """Free VRAM before loading Stable Diffusion."""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        print(f"Finder unloaded. VRAM now: {self._vram_str()}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load(image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(image_path).convert("RGB")

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
# Helpers
# ------------------------------------------------------------------

def collect_images(folder: str) -> list[str]:
    """Recursively collect all image paths from a folder."""
    return [
        str(p)
        for p in Path(folder).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find: rank images by semantic similarity to a query",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score one image against a text query
  uv run python brain/find.py --query "a coffee shop at night" --image brain/data/test_images/sample.jpg

  # Rank all images in a folder
  uv run python brain/find.py --query "espresso machine" --folder brain/data/test_images/

  # Use an image as the query (image-to-image search)
  uv run python brain/find.py --query brain/data/test_images/sample.jpg --folder brain/data/test_images/ --image-query
        """,
    )
    parser.add_argument("--query", required=True,
                        help="Text prompt or image path (if --image-query)")
    parser.add_argument("--image",
                        help="Single image to score against the query")
    parser.add_argument("--folder",
                        help="Folder of images to rank against the query")
    parser.add_argument("--image-query", action="store_true",
                        help="Treat --query as an image path instead of text")
    parser.add_argument("--top-k", type=int, default=5,
                        help="How many results to show (default: 5)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.image and not args.folder:
        print("Error: provide --image or --folder")
        raise SystemExit(1)

    finder = Finder()

    if args.image:
        # Single image score
        query_vec = (
            finder.encode_image(args.query)
            if args.image_query
            else finder.encode_text(args.query)
        )
        img_vec = finder.encode_image(args.image)
        score = finder.similarity(query_vec, img_vec)
        print(f"\n[Query]  '{args.query}'")
        print(f"[Image]  {args.image}")
        print(f"[Score]  {score:.4f}  {'(strong match)' if score > 0.28 else '(weak match)'}\n")

    else:
        # Folder ranking
        images = collect_images(args.folder)
        print(f"\nRanking {len(images)} image(s) against: '{args.query}'\n")
        results = finder.rank(
            args.query, images, top_k=args.top_k, query_is_image=args.image_query
        )
        for rank, (path, score) in enumerate(results, 1):
            bar = "█" * int(score * 100)
            print(f"  {rank}. [{score:.4f}] {bar}")
            print(f"     {path}")
        print()

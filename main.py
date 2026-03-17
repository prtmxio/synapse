"""
main.py — SynapseAgent

Orchestrates the three-stage pipeline:

  [image] → Scryer → anchor (text) → Finder → similar images
                                             → Maker → generated image

Each stage runs sequentially with VRAM handoff between models:
  load Scryer → scry → unload Scryer
  load Finder → find → unload Finder
  load Maker  → make

Why sequential and not concurrent?
  RTX 2050 has 4GB VRAM. SmolVLM (1.04GB) + SD-turbo (1.7GB) = 2.74GB
  before accounting for PyTorch runtime overhead (~0.3GB), activations,
  and the OS. Loading two at once risks OOM mid-forward-pass.
  Sequential with explicit unload() is the safe pattern on constrained hardware.

Subcommands:
  run   --image <path>              full pipeline (scry → find → make)
  scry  --image <path>              describe an image
  find  --query <text> [--top-k N]  search the HNSW index
  make  --prompt <text> [--seed N]  generate an image
"""

import argparse
import os
import time

# ──────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ──────────────────────────────────────────────────────────────────────────────


def cmd_scry(args):
    from brain.scry import Scryer

    scryer = Scryer()
    anchor = scryer.scry(args.image)
    print(f"\n[Anchor]\n{anchor}\n")
    scryer.unload()


def cmd_find(args):
    from brain.find import Finder

    finder = Finder()
    results = finder.query_index(args.query, top_k=args.top_k)
    print(f"\n[Query]  '{args.query}'\n")
    for i, (path, score) in enumerate(results, 1):
        bar = "█" * int(score * 100)
        print(f"  {i}. [{score:.4f}] {bar}")
        print(f"     {path}")
    print()
    finder.unload()


def cmd_make(args):
    from brain.make import Maker

    maker = Maker()
    path = maker.make_and_save(args.prompt, seed=args.seed, temperature=args.temperature)
    print(f"\n[Prompt] {args.prompt}")
    print(f"[Saved]  {path}\n")
    maker.unload()


def _clip_truncate(text: str, max_words: int = 55) -> str:
    """
    CLIP's text encoder accepts at most 77 tokens (including BOS/EOS).
    English prose averages ~1.3 tokens/word, so 55 words ≈ 72 tokens — safely
    under the limit. We cut at the last complete sentence within that budget
    so the prompt doesn't end mid-clause.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # walk back to last sentence boundary
    for end in (".", "!", "?"):
        last = truncated.rfind(end)
        if last != -1:
            return truncated[:last + 1].strip()
    return truncated.strip()


def cmd_run(args):
    """
    Full pipeline: scry → find → make → scry generated image

    VRAM handoff sequence:
      1. Load Scryer  (~1.04GB) → produce anchor         → unload
      2. Load Finder  (~0.31GB) → retrieve images        → unload
      3. Load Maker   (~1.70GB) → generate image         → unload
      4. Load Scryer  (~1.04GB) → describe generated img → unload
    Peak VRAM never exceeds ~1.7GB because only one model is live at a time.
    """
    t_start = time.perf_counter()
    from brain.scry import Scryer
    from brain.find import Finder
    from brain.make import Maker

    # ── Stage 1: Scry input ────────────────────────────────────────────────
    print("\n── Stage 1: Scry ──────────────────────────────────")
    scryer = Scryer()
    anchor = scryer.scry(args.image)
    scryer.unload()
    print(f"[Anchor] {anchor}")

    # CLIP text encoder hard limit is 77 tokens (~300 chars of English prose).
    # The full anchor is used for retrieval (CLIP handles truncation internally),
    # but for generation we truncate to the first complete sentence that fits
    # so the UNet gets a clean, coherent prompt rather than a mid-sentence cut.
    clip_prompt = _clip_truncate(anchor)

    # ── Stage 2: Find ──────────────────────────────────────────────────────
    print("\n── Stage 2: Find ──────────────────────────────────")
    finder = Finder()
    results = finder.query_index(anchor, top_k=args.top_k)
    finder.unload()
    print(f"[Retrieved {len(results)} images]")
    for i, (path, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {os.path.abspath(path)}")

    # ── Stage 3: Make ──────────────────────────────────────────────────────
    print("\n── Stage 3: Make ──────────────────────────────────")
    if clip_prompt != anchor:
        print(f"[Prompt truncated to {len(clip_prompt.split())} words for CLIP]")
    maker = Maker()
    out_path = maker.make_and_save(clip_prompt, seed=args.seed, temperature=args.temperature)
    maker.unload()
    print(f"[Generated] {os.path.abspath(out_path)}")

    # ── Stage 4: Scry the generated image ──────────────────────────────────
    print("\n── Stage 4: Scry Generated Image ──────────────────")
    scryer = Scryer()
    generated_anchor = scryer.scry(out_path)
    scryer.unload()
    print(f"[Description] {generated_anchor}")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("\n══ Synapse Complete ═══════════════════════════════")
    print(f"  Input:       {os.path.abspath(args.image)}")
    print(f"  Anchor:      {anchor}")
    print(f"\n  Retrieved:   {len(results)} images")
    for i, (path, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.4f}] {os.path.abspath(path)}")
    print(f"\n  Generated:   {os.path.abspath(out_path)}")
    print(f"  Description: {generated_anchor}")
    print(f"\n  Time:        {elapsed:.1f}s")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────


def build_parser():
    parser = argparse.ArgumentParser(
        prog="synapse",
        description="Synapse — perceive, retrieve, generate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py run  --image data/raw_imgs/000000101420.jpg
  uv run python main.py scry --image data/raw_imgs/000000101420.jpg
  uv run python main.py find --query "a cat on a couch" --top-k 3
  uv run python main.py make --prompt "a dark coffee shop with neon signs" --seed 42
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──
    p = sub.add_parser("run", help="Full pipeline: scry → find → make")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of similar images to retrieve (default: 5)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation (default: random)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Output variance 0.0–1.0 (default: 0.0 = deterministic, 1.0 = max variation).",
    )

    # ── scry ──
    p = sub.add_parser("scry", help="Describe an image")
    p.add_argument("--image", required=True, help="Image path to describe")

    # ── find ──
    p = sub.add_parser("find", help="Search the HNSW index by text query")
    p.add_argument("--query", required=True, help="Text query")
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results (default: 5)",
    )

    # ── make ──
    p = sub.add_parser("make", help="Generate an image from a text prompt")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Output variance 0.0–1.0 (default: 0.0 = deterministic, 1.0 = max variation).",
    )

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = build_parser().parse_args()

    dispatch = {
        "run": cmd_run,
        "scry": cmd_scry,
        "find": cmd_find,
        "make": cmd_make,
    }
    dispatch[args.command](args)

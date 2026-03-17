"""
main.py — SynapseAgent

Orchestrates the three-stage pipeline:

  [image] → Scryer + Finder (co-loaded) → anchor + retrieved images
                                         → Maker → generated image
                                         → Scryer → description of output

VRAM budget (RTX 2050, 4GB):
  Stage 1+2: Scryer (1.04GB) + Finder (0.31GB) = 1.35GB  — fit together
  Stage 3:   Maker (1.70GB)                               — alone

Co-loading Scryer + Finder saves one full model load cycle (~2-3s).

Subcommands:
  run   --image <path>              full pipeline
  scry  --image <path>              describe an image
  find  --query <text> [--top-k N]  search the HNSW index
  make  --prompt <text> [--seed N]  generate an image
"""

import os
import warnings

# Suppress library noise before any imports that trigger them.
# TRANSFORMERS_VERBOSITY=error kills: weight-loading bars, CLIP load report,
#   position_ids UNEXPECTED warnings, fast-processor notices.
# DIFFUSERS_VERBOSITY=error kills: safety checker disclaimer, pipeline bars.
# filterwarnings("ignore") catches the remaining Python-level UserWarnings
#   (NVML, HF Hub auth, tokenizer length).
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import argparse
import time

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _vram() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1e9
            return f"{used:.2f}GB / {total / 1e9:.1f}GB"
    except Exception:
        pass
    return "N/A"


def _clip_truncate(text: str, max_words: int = 55) -> str:
    """Trim to ≤55 words at a sentence boundary (CLIP max is 77 tokens)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    for end in (".", "!", "?"):
        last = truncated.rfind(end)
        if last != -1:
            return truncated[: last + 1].strip()
    return truncated.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ──────────────────────────────────────────────────────────────────────────────


def cmd_scry(args):
    from brain.scry import Scryer

    scryer = Scryer()
    anchor = scryer.scry(args.image)
    scryer.unload()
    print(f"\n[Anchor]\n{anchor}\n")


def cmd_find(args):
    from brain.find import Finder

    finder = Finder()
    results = finder.query_index(args.query, top_k=args.top_k)
    finder.unload()
    print(f"\n[Query] '{args.query}'\n")
    for i, (path, score) in enumerate(results, 1):
        bar = "█" * int(score * 100)
        print(f"  {i}. [{score:.4f}] {bar}")
        print(f"     {os.path.abspath(path)}")
    print()


def cmd_make(args):
    from brain.make import Maker

    maker = Maker()
    path = maker.make_and_save(
        args.prompt, seed=args.seed, temperature=args.temperature
    )
    maker.unload()
    print(f"\n[Prompt]    {args.prompt}")
    print(f"[Generated] {os.path.abspath(path)}\n")


def cmd_run(args):
    """
    Full pipeline with optimised VRAM schedule:

      ┌─ Stage 1+2 ──────────────────────────────────────────┐
      │  Load Scryer (1.04GB) + Finder (0.31GB) = 1.35GB     │
      │  scry(input) → anchor                                 │
      │  query_index(anchor) → retrieved images               │
      │  Unload both                                          │
      └───────────────────────────────────────────────────────┘
      ┌─ Stage 3 ─────────────────────────────────────────────┐
      │  Load Maker (1.70GB)                                  │
      │  make(anchor) → generated image                       │
      │  Unload                                               │
      └───────────────────────────────────────────────────────┘
    """
    t_start = time.perf_counter()
    from brain.find import Finder
    from brain.make import Maker
    from brain.scry import Scryer

    # ── Stage 1+2: Scry + Find (co-loaded) ────────────────────────────────
    print(f"\n── Stage 1+2: Scry + Find  [VRAM: {_vram()}] ──────")
    scryer = Scryer()
    finder = Finder()

    anchor = scryer.scry(args.image)
    clip_prompt = _clip_truncate(anchor)
    results = finder.query_index(anchor, top_k=args.top_k)

    scryer.unload()
    finder.unload()

    print(f"[Anchor]    {anchor}")
    print(f"[Retrieved] {len(results)} images")
    for i, (path, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] {os.path.abspath(path)}")

    # ── Stage 3: Make ──────────────────────────────────────────────────────
    print(f"\n── Stage 3: Make  [VRAM: {_vram()}] ───────────────")
    maker = Maker()
    out_path = maker.make_and_save(
        clip_prompt, seed=args.seed, temperature=args.temperature
    )
    maker.unload()
    print(f"[Generated]\n  {os.path.abspath(out_path)}")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print("\n══ Synapse Complete ═══════════════════════════════")
    print(f"  Input:     {os.path.abspath(args.image)}")
    print(f"  Anchor:    {anchor}")
    print(f"\n  Retrieved: {len(results)} images")
    for i, (path, score) in enumerate(results, 1):
        print(f"    {i}. [{score:.4f}] {os.path.abspath(path)}")
    print(f"\n  Generated:\n    {os.path.abspath(out_path)}")
    print(f"\n  Time:      {elapsed:.1f}s")
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
  uv run python main.py make --prompt "a dark coffee shop at night" --seed 42
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
        help="Images to retrieve (default: 5)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: random)"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Output variance 0.0–1.0 (default: 0.0)",
    )

    # ── scry ──
    p = sub.add_parser("scry", help="Describe an image")
    p.add_argument("--image", required=True, help="Image path")

    # ── find ──
    p = sub.add_parser("find", help="Search HNSW index by text")
    p.add_argument("--query", required=True, help="Text query")
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results (default: 5)",
    )

    # ── make ──
    p = sub.add_parser("make", help="Generate an image from a prompt")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Output variance 0.0–1.0 (default: 0.0)",
    )

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = build_parser().parse_args()
    dispatch = {"run": cmd_run, "scry": cmd_scry, "find": cmd_find, "make": cmd_make}
    dispatch[args.command](args)

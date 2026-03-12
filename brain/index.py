"""
brain/index.py — One-time HNSW index builder

Encodes every image in a folder with CLIP once, saves the result to disk.
After this runs, find.py --db can search 5,000 images in milliseconds.

Two output files:
  brain/data/index.bin   — the HNSW graph (vectors + edges, binary)
  brain/data/index.json  — {str(int_id) → "path/to/image.jpg"}

Run once:
  uv run python brain/index.py --folder brain/data/raw_images/val2017/

Then query fast:
  uv run python brain/find.py --query "a cat on a couch" --db
"""

import argparse
import json
import os
import sys

import hnswlib
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(_HERE, "../brain/data/index.bin")
META_PATH  = os.path.join(_HERE, "../brain/data/index.json")

VECTOR_DIM = 512  # CLIP ViT-B/32 output dimension


def build(folder: str) -> None:
    from find import Finder, collect_images

    # ------------------------------------------------------------------
    # Step 1: collect image paths
    # ------------------------------------------------------------------
    print(f"Scanning '{folder}'...")
    paths = collect_images(folder)
    if not paths:
        print("No images found.")
        sys.exit(1)

    n = len(paths)
    print(f"Found {n} images.\n")

    # ------------------------------------------------------------------
    # Step 2: create the HNSW index
    #
    # space="cosine": the distance metric we use between vectors.
    #   cosine distance = 1 - cosine_similarity
    #   So distance 0.0 = identical, distance 1.0 = orthogonal (unrelated),
    #   distance 2.0 = opposite.
    #   We use cosine because our CLIP vectors are L2-normalized — their
    #   magnitude is always 1, so cosine similarity == dot product, which
    #   is exactly what we computed in find.py with the @ operator.
    #
    # dim=512: each vector has 512 dimensions (CLIP ViT-B/32).
    # ------------------------------------------------------------------
    index = hnswlib.Index(space="cosine", dim=VECTOR_DIM)

    # init_index parameters:
    #
    # max_elements=n: the hard cap on how many vectors this index holds.
    #   HNSW pre-allocates memory for this many nodes upfront. Don't set
    #   it higher than needed — it wastes RAM.
    #
    # ef_construction=200: during BUILD, when we insert a new vector, HNSW
    #   needs to find which existing nodes to connect it to. This parameter
    #   controls how many candidates it considers while searching for those
    #   neighbors. Think of it as "effort spent making good connections".
    #   Higher → better graph quality → more accurate queries later
    #          → slower indexing time
    #   200 is the standard default. For 5k images, build takes ~5 min anyway.
    #
    # M=16: the number of bidirectional connections each node has per layer.
    #   At the bottom (densest) layer, nodes have up to 2*M=32 connections.
    #   Higher → better recall (finds true nearest neighbors more reliably)
    #          → more RAM per node
    #   16 is the standard default for most datasets.
    index.init_index(max_elements=n, ef_construction=200, M=16)

    # ------------------------------------------------------------------
    # Step 3: encode every image and insert into the index
    # ------------------------------------------------------------------
    finder = Finder()
    metadata = {}  # { str(int_id) → "path/to/image.jpg" }
    failed  = 0

    for i, path in enumerate(paths):
        try:
            # encode_image returns a torch.Tensor, dtype=float16, shape [1, 512]
            # hnswlib is a C++ library — it only accepts float32 numpy arrays.
            # So we: move off GPU (.cpu()), cast to float32, convert to numpy.
            vec = finder.encode_image(path)
            vec_np = vec.cpu().to(torch.float32).numpy()  # shape [1, 512]

            # add_items(data, ids):
            #   data: numpy array of shape [n_items, dim]
            #   ids:  list of integer IDs — one per vector
            #
            # Why integer IDs? HNSW is a pure math structure — it stores floats
            # and graph edges. It has no concept of strings or file paths.
            # We assign each image a sequential integer (0, 1, 2...) and keep
            # the int→path mapping ourselves in the metadata dict.
            index.add_items(vec_np, [i])
            metadata[str(i)] = path  # str key because JSON only allows str keys

        except Exception as e:
            print(f"\n  [skip] {os.path.basename(path)}: {e}")
            failed += 1
            continue

        # progress bar
        if (i + 1) % 50 == 0 or i == n - 1:
            pct = (i + 1) / n * 100
            filled = int(pct / 5)
            bar = "█" * filled + "░" * (20 - filled)
            print(f"  [{bar}] {i+1}/{n} ({pct:.0f}%)", end="\r")

    print(f"\n\nEncoded {n - failed}/{n}  ({failed} skipped)")

    # ------------------------------------------------------------------
    # Step 4: set query-time ef
    #
    # set_ef(ef): at QUERY time, ef controls the size of the candidate
    #   pool while traversing the HNSW graph. Different from ef_construction
    #   which only affects the build phase.
    #
    #   Rule: ef must be >= k (the top_k you'll request at query time).
    #   Higher ef → more accurate results → slightly slower queries.
    #   ef=50 is generous for a 5k image dataset querying top-10.
    #   This setting is saved into the index file, so you don't need to
    #   set it again when loading.
    # ------------------------------------------------------------------
    index.set_ef(50)

    # ------------------------------------------------------------------
    # Step 5: save to disk
    #
    # index.save_index: saves the entire HNSW graph as a binary file.
    #   This includes all vectors AND all graph edges. Loading it back
    #   is instant — no rebuild needed.
    #
    # metadata JSON: maps str(int_id) → file_path.
    #   At query time: HNSW returns integer labels → we look up the path.
    #   We use str keys because JSON only supports string keys, so we
    #   convert back with str(label) on the query side.
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    index.save_index(INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    size_mb = os.path.getsize(INDEX_PATH) / 1e6
    print(f"\nIndex  → {INDEX_PATH}  ({size_mb:.1f} MB)")
    print(f"Meta   → {META_PATH}")
    print(f"\nAll done. Query with:")
    print(f'  uv run python brain/find.py --query "your text" --db')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HNSW index from image folder")
    parser.add_argument("--folder", required=True, help="Folder of images to index")
    args = parser.parse_args()
    build(args.folder)

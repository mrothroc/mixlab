#!/usr/bin/env python3
"""Download and prepare FineWeb-Edu for mixlab training.

Downloads FineWeb-Edu (10B token subset) from HuggingFace, tokenizes with
a SentencePiece BPE tokenizer at configurable vocab size, and writes binary
shards ready for mixlab.

Usage:
    python3 scripts/download_fineweb.py --output data/fineweb --vocab-size 1024
    python3 scripts/download_fineweb.py --output data/fineweb --vocab-size 8192

Requirements:
    pip install numpy tokenizers datasets

The output directory will contain:
    train_00000.bin, train_00001.bin, ...  (training shards)
    val_00000.bin                          (validation shard)
    tokenizer.json                         (BPE tokenizer)
    bytes_per_token.bin, has_leading_space.bin, is_boundary_token.bin  (BPB LUTs)
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu and prepare binary shards for mixlab"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for shards"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1024,
        help="BPE vocabulary size (default: 1024). Common choices: 1024, 4096, 8192",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of tokens for validation (default: 0.05)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Limit number of documents (0 = all, useful for testing)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=200_000_000,
        help="Approximate shard size in bytes (default: 200MB)",
    )
    args = parser.parse_args()

    # Lazy imports so --help is fast
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' package not installed.\n"
            "  Install with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import numpy  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'numpy' package not installed.\n"
            "  Install with: pip install numpy",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print(
            "ERROR: 'tokenizers' package not installed.\n"
            "  Install with: pip install tokenizers",
            file=sys.stderr,
        )
        sys.exit(1)

    # Step 1: Download FineWeb-Edu
    print(f"Downloading FineWeb-Edu from HuggingFace...")
    print(f"  This may take a while on the first run (~20GB download).")
    print(f"  Subsequent runs use the HuggingFace cache.\n")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=False,
    )

    # Step 2: Write raw text to a temporary JSONL file
    raw_dir = os.path.join(args.output, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    jsonl_path = os.path.join(raw_dir, "fineweb_edu.jsonl")

    if os.path.exists(jsonl_path):
        print(f"  [cached] {jsonl_path}")
    else:
        import json

        print(f"  Writing JSONL to {jsonl_path}...")
        count = 0
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for doc in ds:
                if args.max_docs > 0 and count >= args.max_docs:
                    break
                json.dump({"text": doc["text"]}, f, ensure_ascii=False)
                f.write("\n")
                count += 1
                if count % 100_000 == 0:
                    print(f"    {count:,} documents written...")
        print(f"  Wrote {count:,} documents to {jsonl_path}")

    # Step 3: Find prepare.py and run tokenization
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prepare_py = os.path.join(script_dir, "prepare.py")

    if not os.path.exists(prepare_py):
        print(f"ERROR: prepare.py not found at {prepare_py}", file=sys.stderr)
        sys.exit(1)

    import subprocess

    print(f"\nTokenizing with vocab_size={args.vocab_size}...")
    cmd = [
        sys.executable,
        prepare_py,
        "--input", jsonl_path,
        "--output", args.output,
        "--vocab-size", str(args.vocab_size),
        "--val-split", str(args.val_split),
        "--text-field", "text",
    ]
    print(f"  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("ERROR: prepare.py failed", file=sys.stderr)
        sys.exit(1)

    # Summary
    import glob

    train_shards = sorted(glob.glob(os.path.join(args.output, "train_*.bin")))
    val_shards = sorted(glob.glob(os.path.join(args.output, "val_*.bin")))
    total_bytes = sum(os.path.getsize(f) for f in train_shards + val_shards)
    total_tokens = total_bytes // 2  # uint16

    print(f"\nDone!")
    print(f"  Train shards: {len(train_shards)}")
    print(f"  Val shards:   {len(val_shards)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Vocab size:   {args.vocab_size}")
    print(f"  Output:       {args.output}/")
    print(f"\nTo train:")
    print(f"  mixlab -mode arch -config examples/plain_3L.json \\")
    print(f"    -train '{args.output}/train_*.bin'")


if __name__ == "__main__":
    main()

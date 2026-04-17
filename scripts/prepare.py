#!/usr/bin/env python3
"""
prepare.py — Tokenize raw text into binary shards for mixlab training.

Shard format (matching cmd/mixlab/data/loader.go):
  256 x int32 header: [magic=20240520, version=1, nTok, 0, ...]
  nTok x uint16 tokens

Usage:
  python3 prepare.py --input data.txt --output shards/ --vocab-size 1024
  python3 prepare.py --input corpus/ --output shards/ --vocab-size 1024 --val-split 0.1
  python3 prepare.py --input data.jsonl --output shards/ --vocab-size 1024 --text-field text
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
TOKENS_PER_SHARD = 1_000_000


def read_input_texts(input_path: str, text_field: str) -> list[str]:
    """Read text from a file, JSONL, or directory of text files."""
    p = Path(input_path)
    if p.is_dir():
        texts = []
        for f in sorted(p.rglob("*.txt")):
            texts.append(f.read_text(encoding="utf-8", errors="replace"))
        for f in sorted(p.rglob("*.jsonl")):
            texts.extend(_read_jsonl(f, text_field))
        if not texts:
            raise ValueError(f"No .txt or .jsonl files found in {input_path}")
        return texts

    if p.suffix == ".jsonl":
        return _read_jsonl(p, text_field)

    # Plain text file
    return [p.read_text(encoding="utf-8", errors="replace")]


def _read_jsonl(path: Path, text_field: str) -> list[str]:
    texts = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if text_field in obj:
                texts.append(obj[text_field])
    return texts


def train_bpe_tokenizer(texts: list[str], vocab_size: int, output_dir: str):
    """Train a BPE tokenizer using the HuggingFace tokenizers library."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|pad|>"],
        show_progress=True,
    )

    # Train from iterator
    tokenizer.train_from_iterator(texts, trainer=trainer)

    tok_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"Saved tokenizer to {tok_path} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(tokenizer_path: str):
    """Load a pre-trained HuggingFace tokenizer from a JSON file."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loaded tokenizer from {tokenizer_path} (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def tokenize_texts(tokenizer, texts: list[str]) -> np.ndarray:
    """Tokenize all texts and concatenate into a single uint16 array."""
    all_ids = []
    for text in texts:
        enc = tokenizer.encode(text)
        all_ids.extend(enc.ids)
    arr = np.array(all_ids, dtype=np.uint16)
    print(f"Tokenized {len(texts)} text(s) -> {len(arr):,} tokens")
    return arr


def write_shard(path: str, tokens: np.ndarray):
    """Write a single binary shard with the standard header format."""
    n_tok = len(tokens)
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = n_tok

    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def write_shards(tokens: np.ndarray, output_dir: str, prefix: str, shuffle: bool = True) -> int:
    """Split tokens into shards of TOKENS_PER_SHARD and write them out."""
    rng = np.random.default_rng(42)
    n_shards = max(1, (len(tokens) + TOKENS_PER_SHARD - 1) // TOKENS_PER_SHARD)
    written = 0

    for i in range(n_shards):
        start = i * TOKENS_PER_SHARD
        end = min(start + TOKENS_PER_SHARD, len(tokens))
        chunk = tokens[start:end].copy()

        if shuffle and len(chunk) > 2048:
            # Shuffle in 2048-token blocks (matching loader.go shuffleChunks)
            block_size = 2048
            n_blocks = len(chunk) // block_size
            blocks = chunk[: n_blocks * block_size].reshape(n_blocks, block_size)
            rng.shuffle(blocks)
            chunk[: n_blocks * block_size] = blocks.reshape(-1)

        shard_path = os.path.join(output_dir, f"{prefix}_{i:05d}.bin")
        write_shard(shard_path, chunk)
        written += 1
        print(f"  {shard_path}: {len(chunk):,} tokens")

    return written


def main():
    parser = argparse.ArgumentParser(description="Prepare binary shards for mixlab training")
    parser.add_argument("--input", required=True, help="Input text file, JSONL, or directory")
    parser.add_argument("--output", required=True, help="Output directory for shards")
    parser.add_argument("--vocab-size", type=int, default=1024, help="BPE vocabulary size (default: 1024)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of tokens for validation (default: 0.1)")
    parser.add_argument("--tokenizer-path", default="", help="Path to pre-trained tokenizer.json (skip training)")
    parser.add_argument("--text-field", default="text", help="JSON field name for text in JSONL files (default: text)")
    parser.add_argument("--tokens-per-shard", type=int, default=TOKENS_PER_SHARD, help="Tokens per shard (default: 1000000)")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable token shuffling within shards")
    args = parser.parse_args()

    global TOKENS_PER_SHARD
    TOKENS_PER_SHARD = args.tokens_per_shard

    # Read input
    print(f"Reading input from {args.input}...")
    texts = read_input_texts(args.input, args.text_field)
    print(f"Read {len(texts)} text segment(s)")

    # Train or load tokenizer
    os.makedirs(args.output, exist_ok=True)

    if args.tokenizer_path:
        tokenizer = load_tokenizer(args.tokenizer_path)
    else:
        print(f"Training BPE tokenizer with vocab_size={args.vocab_size}...")
        tokenizer = train_bpe_tokenizer(texts, args.vocab_size, args.output)

    # Tokenize
    all_tokens = tokenize_texts(tokenizer, texts)

    if len(all_tokens) == 0:
        print("ERROR: No tokens produced. Check your input.", file=sys.stderr)
        sys.exit(1)

    # Validate token range fits uint16
    max_token = int(all_tokens.max())
    if max_token >= 65536:
        print(f"ERROR: Token ID {max_token} exceeds uint16 range", file=sys.stderr)
        sys.exit(1)

    # Split train/val
    n_val = max(1, int(len(all_tokens) * args.val_split))
    n_train = len(all_tokens) - n_val

    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train:]

    print(f"\nSplit: {n_train:,} train tokens, {n_val:,} val tokens")

    # Write shards
    print("\nWriting training shards...")
    n_train_shards = write_shards(train_tokens, args.output, "train", shuffle=not args.no_shuffle)

    print("\nWriting validation shards...")
    n_val_shards = write_shards(val_tokens, args.output, "val", shuffle=False)

    print(f"\nDone! {n_train_shards} train shard(s), {n_val_shards} val shard(s) in {args.output}")
    print(f"Train pattern: {args.output}/train_*.bin")
    print(f"Val pattern:   {args.output}/val_*.bin")


if __name__ == "__main__":
    main()

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
CHAR_FEATURE_MAGIC = 20260526
CHAR_FEATURE_VERSION = 1
CHAR_FEATURE_ENCODING_BYTELEVEL = 1
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


def write_shards(tokens: np.ndarray, output_dir: str, prefix: str, shuffle: bool = True, tokens_per_shard: int = TOKENS_PER_SHARD) -> int:
    """Split tokens into shards of tokens_per_shard and write them out."""
    rng = np.random.default_rng(42)
    n_shards = max(1, (len(tokens) + tokens_per_shard - 1) // tokens_per_shard)
    written = 0

    for i in range(n_shards):
        start = i * tokens_per_shard
        end = min(start + tokens_per_shard, len(tokens))
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


def byte_level_reverse_map() -> dict[str, int]:
    """Return HuggingFace/GPT-2 ByteLevel unicode char -> byte mapping."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(0xA1, 0xAC + 1))
    bs += list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    seen = set(bs)
    n = 0
    for b in range(256):
        if b in seen:
            continue
        bs.append(b)
        cs.append(256 + n)
        n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def has_bytelevel_pretokenizer(tok_json: dict) -> bool:
    def visit(node):
        if not isinstance(node, dict):
            return False
        if node.get("type") == "ByteLevel":
            return True
        if node.get("type") == "Sequence":
            return any(visit(child) for child in node.get("pretokenizers", []))
        return False

    return visit(tok_json.get("pre_tokenizer"))


def write_char_features(tokenizer, output_dir: str, char_vocab_size: int, char_max_per_token: int):
    if char_vocab_size <= 0:
        return
    if char_vocab_size < 257:
        raise ValueError("--char-vocab-size must be 0 to disable or >= 257 when enabled")
    if char_max_per_token <= 0:
        raise ValueError("--char-max-per-token must be > 0 when char features are enabled")

    tok_json = json.loads(tokenizer.to_str())
    if tok_json.get("model", {}).get("type") != "BPE":
        raise ValueError("char features require a HuggingFace ByteLevel BPE tokenizer (model.type=BPE)")
    if not has_bytelevel_pretokenizer(tok_json):
        raise ValueError("char features require a HuggingFace ByteLevel BPE tokenizer (pre_tokenizer=ByteLevel)")

    vocab = tok_json.get("model", {}).get("vocab", {})
    if not vocab:
        raise ValueError("tokenizer JSON has no model.vocab")
    vocab_size = tokenizer.get_vocab_size()
    rows = np.zeros((vocab_size, char_max_per_token), dtype=np.uint16)

    special_ids = set()
    for added in tok_json.get("added_tokens", []):
        if added.get("special"):
            idx = int(added.get("id", -1))
            if 0 <= idx < vocab_size:
                special_ids.add(idx)

    reverse = byte_level_reverse_map()
    for token, idx in vocab.items():
        idx = int(idx)
        if idx < 0 or idx >= vocab_size or idx in special_ids:
            continue
        seen = set()
        ids = []
        for ch in token:
            if ch not in reverse:
                raise ValueError(f"token {token!r} contains non-ByteLevel character U+{ord(ch):04X}")
            b = reverse[ch]
            if b in seen:
                continue
            seen.add(b)
            ids.append(b + 1)
            if len(ids) >= char_max_per_token:
                break
        if ids:
            rows[idx, : len(ids)] = np.array(ids, dtype=np.uint16)

    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = CHAR_FEATURE_MAGIC
    header[1] = CHAR_FEATURE_VERSION
    header[2] = vocab_size
    header[3] = char_vocab_size
    header[4] = char_max_per_token
    header[5] = CHAR_FEATURE_ENCODING_BYTELEVEL

    path = os.path.join(output_dir, "char_features.bin")
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(rows.tobytes())
    print(
        f"Saved char features to {path} "
        f"(vocab_size={vocab_size}, char_vocab_size={char_vocab_size}, char_max_per_token={char_max_per_token})"
    )


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
    parser.add_argument("--char-vocab-size", type=int, default=0, help="Write tokenizer-level char_features.bin; 0 disables")
    parser.add_argument("--char-max-per-token", type=int, default=16, help="Fixed char feature slots per token")
    args = parser.parse_args()

    tokens_per_shard = args.tokens_per_shard

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

    write_char_features(tokenizer, args.output, args.char_vocab_size, args.char_max_per_token)

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
    n_train_shards = write_shards(train_tokens, args.output, "train", shuffle=not args.no_shuffle, tokens_per_shard=tokens_per_shard)

    print("\nWriting validation shards...")
    n_val_shards = write_shards(val_tokens, args.output, "val", shuffle=False, tokens_per_shard=tokens_per_shard)

    print(f"\nDone! {n_train_shards} train shard(s), {n_val_shards} val shard(s) in {args.output}")
    print(f"Train pattern: {args.output}/train_*.bin")
    print(f"Val pattern:   {args.output}/val_*.bin")


if __name__ == "__main__":
    main()

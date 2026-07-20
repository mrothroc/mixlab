"""Record-preserving text preparation for Mixlab sequence shards."""

import json
import os

import numpy as np


def tokenizer_special_token_ids(tokenizer) -> dict[str, int]:
    """Return inspectable tokenizer special-token IDs keyed by token text."""
    tokenizer_doc = json.loads(tokenizer.to_str())
    special_token_ids = {}
    for entry in tokenizer_doc.get("added_tokens") or []:
        if not entry.get("special", False):
            continue
        token = str(entry.get("content", ""))
        token_id = entry.get("id")
        if token and isinstance(token_id, int):
            special_token_ids[token] = token_id
    return special_token_ids


def split_text_records(texts: list[str], val_split: float) -> tuple[list[str], list[str]]:
    if val_split < 0 or val_split >= 1:
        raise ValueError("--val-split must be in [0,1) for per-record framing")
    if val_split > 0 and len(texts) < 2:
        raise ValueError("per-record framing with --val-split > 0 requires at least two input records")
    n_val = 0 if val_split == 0 else min(len(texts) - 1, max(1, int(len(texts) * val_split)))
    split = len(texts) - n_val
    return texts[:split], texts[split:]


def tokenize_text_records(tokenizer, texts: list[str], seq_len: int, overflow: str, split_name: str):
    capacity = seq_len - 2
    records = []
    dropped = 0
    truncated = 0
    for index, text in enumerate(texts):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            raise ValueError(f"{split_name} record {index} produced no tokens")
        if len(ids) > capacity:
            if overflow == "error":
                raise ValueError(
                    f"{split_name} record {index} has {len(ids)} tokens but --record-seq-len={seq_len} "
                    f"permits {capacity}; use --record-overflow=drop or truncate"
                )
            if overflow == "drop":
                dropped += 1
                continue
            ids = ids[:capacity]
            truncated += 1
        records.append((f"{split_name}_{index:08d}", np.asarray(ids, dtype=np.uint16)))
    if texts and not records:
        raise ValueError(f"all {len(texts)} {split_name} records were dropped by the overflow policy")
    lengths = [len(record) for _, record in records]
    stats = {
        "input": len(texts),
        "written": len(records),
        "dropped": dropped,
        "truncated": truncated,
        "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "max": max(lengths, default=0),
    }
    return records, stats


def _write_text_record_manifest(tokenizer, output_dir: str, seq_len: int, pad_id: int, bos_id: int, eos_id: int,
                                train_records, val_records, train_stats: dict, val_stats: dict,
                                n_train_shards: int, n_val_shards: int):
    def split_entry(records, stats, pattern, shards):
        return {
            "pattern": pattern,
            "tokens": sum(len(record) for _, record in records),
            "shards": shards,
            "sequences": len(records),
            "dropped_sequences": stats["dropped"],
            "truncated_sequences": stats["truncated"],
            "mean_sequence_tokens": stats["mean"],
            "max_sequence_tokens": stats["max"],
        }

    manifest = {
        "format": "mixlab.dataset",
        "version": 1,
        "representation": "discrete_tokens",
        "modality": "text",
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "token_dtype": "uint16",
        "shard_format": "mixlab_sequence_shard_v1",
        "sequence_layout": "one_record_per_row",
        "record_seq_len": seq_len,
        "special_token_ids": {"pad": pad_id, "bos": bos_id, "eos": eos_id},
        "artifacts": {"tokenizer": "tokenizer.json"},
        "splits": {
            "train": split_entry(train_records, train_stats, "train_*.bin", n_train_shards),
            "val": split_entry(val_records, val_stats, "val_*.bin", n_val_shards),
        },
    }
    path = os.path.join(output_dir, "mixlab.dataset.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Saved per-record dataset manifest to {path}")


def prepare_text_records(args, tokenizer, texts: list[str], write_sequence_shards):
    if args.record_seq_len < 3:
        raise ValueError("--frame-per-record requires --record-seq-len >= 3")
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size > 65536:
        raise ValueError(f"tokenizer vocabulary size {vocab_size} exceeds uint16 record shard capacity")
    token_ids = [args.record_pad_id, args.record_bos_id, args.record_eos_id]
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise ValueError(f"record pad/bos/eos IDs must each be in [0,{vocab_size})")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("record pad/bos/eos IDs must be distinct")

    train_texts, val_texts = split_text_records(texts, args.val_split)
    train_records, train_stats = tokenize_text_records(tokenizer, train_texts, args.record_seq_len, args.record_overflow, "train")
    val_records, val_stats = tokenize_text_records(tokenizer, val_texts, args.record_seq_len, args.record_overflow, "val")
    print(f"Per-record split: {len(train_records):,} train / {len(val_records):,} val records")
    for name, stats in (("train", train_stats), ("val", val_stats)):
        drop_rate = stats["dropped"] / stats["input"] if stats["input"] else 0.0
        print(
            f"  {name}: input={stats['input']:,} written={stats['written']:,} dropped={stats['dropped']:,} "
            f"drop_rate={drop_rate:.2%} truncated={stats['truncated']:,} "
            f"mean_tokens={stats['mean']:.2f} max_tokens={stats['max']:,}"
        )
    n_train_shards = write_sequence_shards(train_records, args.output, "train", args.tokens_per_shard)
    n_val_shards = write_sequence_shards(val_records, args.output, "val", args.tokens_per_shard)
    _write_text_record_manifest(
        tokenizer, args.output, args.record_seq_len,
        args.record_pad_id, args.record_bos_id, args.record_eos_id,
        train_records, val_records, train_stats, val_stats, n_train_shards, n_val_shards,
    )
    print(f"Done! Train pattern: {args.output}/train_*.bin")

"""Record-preserving text preparation for Mixlab sequence shards."""

import json
import os
from collections import Counter

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


def validate_label_space(records) -> int:
    labels = sorted({int(record[2]) for record in records})
    if any(isinstance(record[2], bool) or int(record[2]) < 0 for record in records):
        raise ValueError("classification labels must be non-negative integer class IDs")
    if len(labels) < 2:
        raise ValueError("classification preparation requires at least two labels")
    expected = list(range(labels[-1] + 1))
    if labels != expected:
        raise ValueError(f"classification labels must be contiguous IDs 0..N-1; got {labels}")
    return len(labels)


def split_labeled_records(records, val_split: float):
    if val_split < 0 or val_split >= 1:
        raise ValueError("--val-split must be in [0,1) for labeled records")
    if val_split == 0:
        return records[:], []
    by_label = {}
    for index, record in enumerate(records):
        by_label.setdefault(int(record[2]), []).append(index)
    val_indexes = set()
    for label, indexes in sorted(by_label.items()):
        if len(indexes) < 2:
            continue
        count = min(len(indexes) - 1, max(1, int(len(indexes) * val_split)))
        val_indexes.update(indexes[-count:])
    if not val_indexes:
        raise ValueError("--val-split > 0 requires at least one class with two labeled records")
    train = [record for i, record in enumerate(records) if i not in val_indexes]
    val = [record for i, record in enumerate(records) if i in val_indexes]
    return train, val


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


def tokenize_labeled_text_records(tokenizer, records, seq_len: int, overflow: str, split_name: str):
    capacity = seq_len - 2
    encoded = []
    dropped = 0
    truncated = 0
    for index, (record_id, text, label) in enumerate(records):
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            raise ValueError(f"{split_name} record {record_id!r} produced no tokens")
        if len(ids) > capacity:
            if overflow == "error":
                raise ValueError(
                    f"{split_name} record {record_id!r} has {len(ids)} tokens but --record-seq-len={seq_len} "
                    f"permits {capacity}; use --record-overflow=drop or truncate"
                )
            if overflow == "drop":
                dropped += 1
                continue
            ids = ids[:capacity]
            truncated += 1
        encoded.append((str(record_id or f"{split_name}_{index:08d}"), np.asarray(ids, dtype=np.uint16), int(label)))
    if records and not encoded:
        raise ValueError(f"all {len(records)} {split_name} records were dropped by the overflow policy")
    lengths = [len(record) for _, record, _ in encoded]
    stats = {
        "input": len(records),
        "written": len(encoded),
        "dropped": dropped,
        "truncated": truncated,
        "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "max": max(lengths, default=0),
    }
    return encoded, stats


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


def prepare_labeled_text_records(args, tokenizer, records, write_labeled_sequence_shards):
    if args.record_seq_len < 3:
        raise ValueError("classification preparation requires --record-seq-len >= 3")
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if vocab_size > 65536:
        raise ValueError(f"tokenizer vocabulary size {vocab_size} exceeds uint16 record shard capacity")
    token_ids = [args.record_pad_id, args.record_bos_id, args.record_eos_id]
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise ValueError(f"record pad/bos/eos IDs must each be in [0,{vocab_size})")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("record pad/bos/eos IDs must be distinct")
    num_labels = validate_label_space(records)
    train_raw, val_raw = split_labeled_records(records, args.val_split)
    train_records, train_stats = tokenize_labeled_text_records(
        tokenizer, train_raw, args.record_seq_len, args.record_overflow, "train"
    )
    val_records, val_stats = tokenize_labeled_text_records(
        tokenizer, val_raw, args.record_seq_len, args.record_overflow, "val"
    )
    if sorted({label for _, _, label in train_records}) != list(range(num_labels)):
        raise ValueError("every class must retain at least one training record after overflow handling")
    n_train_shards = write_labeled_sequence_shards(
        train_records, args.output, "train", args.tokens_per_shard
    )
    n_val_shards = write_labeled_sequence_shards(
        val_records, args.output, "val", args.tokens_per_shard
    )

    def split_entry(encoded, stats, pattern, shards):
        counts = Counter(label for _, _, label in encoded)
        return {
            "pattern": pattern,
            "tokens": sum(len(record) for _, record, _ in encoded),
            "shards": shards,
            "sequences": len(encoded),
            "dropped_sequences": stats["dropped"],
            "truncated_sequences": stats["truncated"],
            "mean_sequence_tokens": stats["mean"],
            "max_sequence_tokens": stats["max"],
            "class_counts": {str(label): int(counts.get(label, 0)) for label in range(num_labels)},
        }

    manifest = {
        "format": "mixlab.dataset",
        "version": 1,
        "representation": "discrete_tokens",
        "modality": "text",
        "vocab_size": vocab_size,
        "token_dtype": "uint16",
        "shard_format": "mixlab_labeled_sequence_shard_v1",
        "sequence_layout": "one_record_per_row",
        "record_seq_len": args.record_seq_len,
        "special_token_ids": {
            "pad": args.record_pad_id,
            "bos": args.record_bos_id,
            "eos": args.record_eos_id,
        },
        "artifacts": {"tokenizer": "tokenizer.json"},
        "task": {"type": "single_label_classification", "num_labels": num_labels},
        "splits": {
            "train": split_entry(train_records, train_stats, "train_*.bin", n_train_shards),
            "val": split_entry(val_records, val_stats, "val_*.bin", n_val_shards),
        },
    }
    path = os.path.join(args.output, "mixlab.dataset.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"Prepared labeled text records: train={len(train_records):,} val={len(val_records):,} "
        f"num_labels={num_labels}"
    )
    print(f"Saved classification dataset manifest to {path}")

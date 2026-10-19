"""Pinned, document-disjoint FineWeb-Edu experiment data (not a general CLI)."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

DATASET = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def partition(text):
    digest = hashlib.sha256(text.encode()).hexdigest()
    return digest, "validation" if int(digest[:8], 16) % 20 == 0 else "train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs", type=int, default=20000)
    parser.add_argument("--vocab-size", type=int, default=8192)
    args = parser.parse_args()
    if args.docs < 100 or not 257 <= args.vocab_size <= 65536:
        parser.error("require docs>=100 and vocab-size in [257,65536]")
    args.output.mkdir(parents=True, exist_ok=False)
    from datasets import load_dataset
    import datasets
    import tokenizers
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from prepare import train_bpe_tokenizer, write_shards

    ds = load_dataset(DATASET, name="sample-10BT", split="train", revision=REVISION, streaming=True, token=False)
    splits = {"train": [], "validation": []}
    seen = set()
    records = []
    for ordinal, doc in enumerate(itertools.islice(ds, args.docs)):
        text = doc["text"]
        digest, split = partition(text)
        if digest in seen:
            continue
        seen.add(digest)
        splits[split].append(text)
        records.append({"ordinal": ordinal, "id": doc["id"], "text_sha256": digest, "split": split})
    if not all(splits.values()):
        raise ValueError("empty document split")
    tokenizer = train_bpe_tokenizer(splits["train"], args.vocab_size, str(args.output))
    counts = {}
    for split, texts in splits.items():
        target = args.output / split
        target.mkdir()
        # No token-level shuffle or split: documents are assigned before fitting.
        ids = [x for text in texts for x in tokenizer.encode(text).ids]
        counts[split] = len(ids)
        write_shards(np.asarray(ids, dtype="<u2"), str(target), "val" if split == "validation" else "train", shuffle=False)
    (args.output / "documents.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    hashes = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.rglob("*")) if p.is_file()}
    provenance = {"dataset": DATASET, "revision": REVISION, "subset": "sample-10BT", "source_split": "train", "documents_requested": args.docs, "documents_unique": len(records), "split_rule": "SHA256(text) first 32 bits modulo 20 == 0 is validation", "tokenizer_fit": "train documents only", "tokens": counts, "versions": {"datasets": datasets.__version__, "tokenizers": tokenizers.__version__, "numpy": np.__version__}, "command": sys.argv, "sha256": hashes}
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"documents": len(records), "tokens": counts}), flush=True)


if __name__ == "__main__":
    main()

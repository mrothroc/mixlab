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
from collections import Counter
import json
import math
import os
import random
import re
import shutil
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
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


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


def train_bpe_tokenizer(texts: list[str], vocab_size: int, output_dir: str, wwm_compatible: bool = False):
    """Train a BPE tokenizer using the HuggingFace tokenizers library."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=wwm_compatible)

    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"] if wwm_compatible else ["<|pad|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
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


def copy_tokenizer_json(tokenizer_path: str, output_dir: str) -> str:
    """Keep the exact external tokenizer artifact next to generated shards."""
    destination = os.path.join(output_dir, "tokenizer.json")
    if os.path.abspath(tokenizer_path) != os.path.abspath(destination):
        shutil.copyfile(tokenizer_path, destination)
    return destination


def _find_tokenizer_node(value, node_type: str):
    if isinstance(value, dict):
        if value.get("type") == node_type:
            return value
        for child in value.values():
            found = _find_tokenizer_node(child, node_type)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_tokenizer_node(child, node_type)
            if found is not None:
                return found
    return None


def detect_wwm_boundary_scheme(tokenizer) -> str:
    """Validate tokenizer-level word starts used by Mixlab whole-word masking."""
    doc = json.loads(tokenizer.to_str())
    model = doc.get("model") or {}
    model_type = str(model.get("type", "")).lower()
    pre = doc.get("pre_tokenizer")
    metaspace = _find_tokenizer_node(pre, "Metaspace")
    if metaspace is not None:
        legacy_prefix = bool(metaspace.get("add_prefix_space", False))
        if str(metaspace.get("prepend_scheme", "")).lower() != "always" and not legacy_prefix:
            raise ValueError("WWM Metaspace tokenizer requires prepend_scheme='always'")
        if model_type not in {"bpe", "unigram"}:
            raise ValueError(f"WWM Metaspace tokenizer has unsupported model.type={model_type!r}")
        return "sentencepiece"
    bytelevel = _find_tokenizer_node(pre, "ByteLevel")
    if bytelevel is not None:
        if model_type != "bpe":
            raise ValueError(f"WWM ByteLevel tokenizer has unsupported model.type={model_type!r}")
        prepend = _find_tokenizer_node(doc.get("normalizer"), "Prepend")
        prepends_space = bool(prepend and str(prepend.get("prepend", prepend.get("content", ""))).startswith(" "))
        if not bool(bytelevel.get("add_prefix_space", False)) and not prepends_space:
            raise ValueError("WWM ByteLevel tokenizer requires add_prefix_space=true")
        return "bytelevel"
    if model_type == "wordpiece":
        return "wordpiece"
    raise ValueError("tokenizer has no supported whole-word boundary convention")


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


def split_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def join_words(words: list[str]) -> str:
    out = ""
    for tok in words:
        if not out:
            out = tok
        elif re.match(r"^[^\w\s]$", tok):
            out += tok
        elif out.endswith(("(", "[", "{", "“", '"', "'")):
            out += tok
        else:
            out += " " + tok
    return out


AGREEMENT_FLIPS = {
    "is": "are",
    "are": "is",
    "was": "were",
    "were": "was",
    "has": "have",
    "have": "has",
    "does": "do",
    "do": "does",
}


def induce_morphology(texts: list[str]) -> dict[str, object]:
    freq: Counter[str] = Counter()
    for text in texts:
        for tok in split_words(text):
            if tok.isalpha():
                freq[tok.lower()] += 1
    flips = dict(AGREEMENT_FLIPS)
    for word in list(freq):
        if word.endswith("s") and len(word) > 3:
            stem = word[:-1]
            if stem in freq and stem not in flips and word not in flips:
                flips[stem] = word
                flips[word] = stem
    return {"word_freq": freq, "agreement_flips": flips}


def corrupt_word_order(words: list[str], rng: random.Random, _morphology=None) -> tuple[list[str], str] | None:
    candidates = [i for i in range(len(words) - 1) if words[i].isalnum() and words[i + 1].isalnum()]
    if not candidates:
        return None
    i = rng.choice(candidates)
    out = words[:]
    out[i], out[i + 1] = out[i + 1], out[i]
    return out, "word_order"


def corrupt_agreement(words: list[str], rng: random.Random, morphology=None) -> tuple[list[str], str] | None:
    flips = AGREEMENT_FLIPS
    if morphology and isinstance(morphology.get("agreement_flips"), dict):
        flips = morphology["agreement_flips"]
    candidates = [i for i, w in enumerate(words) if w.lower() in flips]
    if not candidates:
        # Fallback: flip a simple present-tense suffix on a later alphabetic token.
        candidates = [i for i, w in enumerate(words) if i > 0 and w.isalpha() and len(w) > 3]
        if not candidates:
            return None
        i = rng.choice(candidates)
        out = words[:]
        if out[i].lower().endswith("s"):
            out[i] = out[i][:-1]
        else:
            out[i] = out[i] + "s"
        return out, "agreement"
    i = rng.choice(candidates)
    out = words[:]
    repl = flips[out[i].lower()]
    out[i] = repl.capitalize() if out[i][:1].isupper() else repl
    return out, "agreement"


def corrupt_attractor(words: list[str], rng: random.Random, morphology=None) -> tuple[list[str], str] | None:
    out = corrupt_agreement(words, rng, morphology)
    if out is None:
        return None
    corrupted, _ = out
    return corrupted, "attractor"


def corrupt_npi_licensor(words: list[str], rng: random.Random, _morphology=None) -> tuple[list[str], str] | None:
    lower = [w.lower() for w in words]
    if any(w in {"any", "ever", "anyone", "anything", "anybody"} for w in lower):
        licensors = [i for i, w in enumerate(lower) if w in {"not", "n't", "never", "no"}]
        if licensors:
            out = [w for i, w in enumerate(words) if i != rng.choice(licensors)]
            return out, "npi_licensor"
    verbs = [i for i, w in enumerate(lower) if w in {"is", "are", "was", "were", "has", "have", "do", "does", "did"}]
    if verbs:
        i = rng.choice(verbs)
        return words[:i] + ["not"] + words[i:], "npi_licensor"
    return None


def corrupt_quantifier_scope(words: list[str], rng: random.Random, _morphology=None) -> tuple[list[str], str] | None:
    flips = {"all": "some", "some": "all", "every": "some", "each": "some", "no": "some", "many": "few", "few": "many"}
    candidates = [i for i, w in enumerate(words) if w.lower() in flips]
    if not candidates:
        return None
    i = rng.choice(candidates)
    out = words[:]
    repl = flips[out[i].lower()]
    out[i] = repl.capitalize() if out[i][:1].isupper() else repl
    return out, "quantifier_scope"


def corrupt_filler_gap(words: list[str], rng: random.Random, morphology=None) -> tuple[list[str], str] | None:
    lower = [w.lower() for w in words]
    wh = [i for i, w in enumerate(lower) if w in {"who", "what", "which", "where", "when"}]
    if wh and len(words) > 4:
        i = rng.choice(wh)
        candidates = [j for j, w in enumerate(words) if j != i and w.isalpha()]
        if candidates:
            j = rng.choice(candidates)
            out = words[:]
            out[i], out[j] = out[j], out[i]
            return out, "filler_gap"
    return corrupt_word_order(words, rng, morphology)


CORRUPTORS = {
    "agreement": corrupt_agreement,
    "attractor": corrupt_attractor,
    "word_order": corrupt_word_order,
    "npi_licensor": corrupt_npi_licensor,
    "quantifier_scope": corrupt_quantifier_scope,
    "filler_gap": corrupt_filler_gap,
}


def sentence_candidates(texts: list[str]) -> list[str]:
    out = []
    for text in texts:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
            sent = sent.strip()
            if sent:
                out.append(sent)
    return out


def parse_minimal_pair_weights(weights: str, families: list[str]) -> dict[str, float]:
    if not weights.strip():
        return {family: 1.0 for family in families}
    parsed: dict[str, float] = {}
    if weights.strip().startswith("{"):
        raw = json.loads(weights)
        if not isinstance(raw, dict):
            raise ValueError("--minimal-pair-weights JSON must be an object")
        for key, value in raw.items():
            parsed[str(key).strip()] = float(value)
    else:
        for part in weights.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                key, value = part.split("=", 1)
            elif ":" in part:
                key, value = part.split(":", 1)
            else:
                raise ValueError("--minimal-pair-weights entries must be family=value")
            parsed[key.strip()] = float(value)
    unknown = [key for key in parsed if key not in families]
    if unknown:
        raise ValueError(f"minimal-pair weights reference disabled families: {', '.join(unknown)}")
    out = {family: parsed.get(family, 1.0) for family in families}
    bad = [family for family, value in out.items() if value < 0 or not math.isfinite(value)]
    if bad:
        raise ValueError(f"minimal-pair weights must be finite and >= 0 for: {', '.join(bad)}")
    if sum(out.values()) <= 0:
        raise ValueError("--minimal-pair-weights must leave at least one positive family weight")
    return out


def weighted_family_choice(rng: random.Random, families: list[str], weights: dict[str, float]) -> str:
    total = sum(weights[family] for family in families)
    draw = rng.random() * total
    acc = 0.0
    for family in families:
        acc += weights[family]
        if draw <= acc:
            return family
    return families[-1]


def write_minimal_pair_report(path: str, report: dict):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


def write_minimal_pair_samples(path: str, samples: list[dict]):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")


def write_minimal_pairs(
    tokenizer,
    texts: list[str],
    output_path: str,
    corruptions: str,
    weights: str,
    morphology: str,
    max_pairs: int,
    seed: int,
    report_out: str,
    sample_out: str,
    sample_count: int,
):
    if not output_path:
        return
    families = [c.strip() for c in corruptions.split(",") if c.strip()]
    if not families:
        raise ValueError("--minimal-pair-corruptions must include at least one family")
    unknown = [c for c in families if c not in CORRUPTORS]
    if unknown:
        raise ValueError(f"unsupported minimal-pair corruptions: {', '.join(unknown)}")
    if sample_count < 0:
        raise ValueError("--minimal-pair-sample-count must be >= 0")
    morphology = morphology.strip().lower()
    if morphology != "induced":
        raise ValueError("--minimal-pair-morphology currently supports only 'induced'")
    family_weights = parse_minimal_pair_weights(weights, families)
    morphology_tables = induce_morphology(texts)
    rng = random.Random(seed)
    candidates = sentence_candidates(texts)
    rng.shuffle(candidates)
    if max_pairs <= 0:
        max_pairs = min(max(1, len(candidates) * 2), 10000)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    counts = {family: 0 for family in families}
    rejects = {family: 0 for family in families}
    reject_reasons: dict[str, dict[str, int]] = {family: {} for family in families}
    samples: list[dict] = []

    def reject(family: str, reason: str):
        rejects[family] = rejects.get(family, 0) + 1
        family_reasons = reject_reasons.setdefault(family, {})
        family_reasons[reason] = family_reasons.get(reason, 0) + 1

    written = 0
    attempts = 0
    max_attempts = max_pairs * 50
    with open(output_path, "w", encoding="utf-8") as f:
        while written < max_pairs and attempts < max_attempts and candidates:
            attempts += 1
            clean_text = candidates[attempts % len(candidates)]
            words = split_words(clean_text)
            family = weighted_family_choice(rng, families, family_weights)
            if len([w for w in words if w.isalnum()]) < 3:
                reject(family, "too_short")
                continue
            result = CORRUPTORS[family](words, rng, morphology_tables)
            if result is None:
                reject(family, "no_candidate")
                continue
            corrupt_words, actual_family = result
            corrupt_text = join_words(corrupt_words)
            if corrupt_text == clean_text:
                reject(actual_family, "unchanged")
                continue
            clean_ids = tokenizer.encode(clean_text).ids
            corrupt_ids = tokenizer.encode(corrupt_text).ids
            if not clean_ids or not corrupt_ids:
                reject(actual_family, "empty_tokens")
                continue
            key = (tuple(clean_ids), tuple(corrupt_ids))
            if key in seen:
                reject(actual_family, "duplicate")
                continue
            seen.add(key)
            rec_id = f"mp_{written:08d}"
            rec = {
                "id": rec_id,
                "clean": clean_ids,
                "corrupt": corrupt_ids,
                "family": actual_family,
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            counts[actual_family] = counts.get(actual_family, 0) + 1
            if len(samples) < sample_count:
                samples.append(
                    {
                        "id": rec_id,
                        "family": actual_family,
                        "clean_text": clean_text,
                        "corrupt_text": corrupt_text,
                        "clean": clean_ids,
                        "corrupt": corrupt_ids,
                    }
                )
            written += 1
    report = {
        "output_path": output_path,
        "seed": seed,
        "morphology": morphology,
        "max_pairs": max_pairs,
        "attempts": attempts,
        "written": written,
        "family_weights": family_weights,
        "accepted_by_family": counts,
        "rejected_by_family": rejects,
        "rejection_reasons": reject_reasons,
        "sample_count": len(samples),
    }
    write_minimal_pair_report(report_out, report)
    write_minimal_pair_samples(sample_out, samples)
    print(f"Saved {written} minimal pairs to {output_path}")
    print(f"Minimal-pair accepted by family: {counts}")
    print(f"Minimal-pair rejected by family: {rejects}")
    if report_out:
        print(f"Minimal-pair report: {report_out}")
    if sample_out:
        print(f"Minimal-pair audit samples: {sample_out}")


def main():
    parser = argparse.ArgumentParser(description="Prepare binary shards for mixlab training")
    parser.add_argument("--input", required=True, help="Input text file, JSONL, or directory")
    parser.add_argument("--output", required=True, help="Output directory for shards")
    parser.add_argument("--vocab-size", type=int, default=1024, help="BPE vocabulary size (default: 1024)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of tokens for validation (default: 0.1)")
    parser.add_argument("--tokenizer-path", default="", help="Path to pre-trained tokenizer.json (skip training)")
    parser.add_argument("--wwm-compatible-tokenizer", action="store_true", help="Train or validate tokenizer metadata for whole-word masking")
    parser.add_argument("--text-field", default="text", help="JSON field name for text in JSONL files (default: text)")
    parser.add_argument("--tokens-per-shard", type=int, default=TOKENS_PER_SHARD, help="Tokens per shard (default: 1000000)")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable token shuffling within shards")
    parser.add_argument("--char-vocab-size", type=int, default=0, help="Write tokenizer-level char_features.bin; 0 disables")
    parser.add_argument("--char-max-per-token", type=int, default=16, help="Fixed char feature slots per token")
    parser.add_argument("--minimal-pair-out", default="", help="Write corpus-derived minimal-pair JSONL")
    parser.add_argument(
        "--minimal-pair-corruptions",
        default="agreement,attractor,word_order",
        help="Comma-separated minimal-pair corruption families",
    )
    parser.add_argument(
        "--minimal-pair-weights",
        default="",
        help="Family weights as JSON object or comma-separated family=value entries",
    )
    parser.add_argument(
        "--minimal-pair-morphology",
        default="induced",
        help="Morphology source for generated pairs; currently 'induced'",
    )
    parser.add_argument("--minimal-pair-max-pairs", type=int, default=0, help="Maximum generated minimal pairs; 0 auto-selects")
    parser.add_argument("--minimal-pair-seed", type=int, default=1234, help="Deterministic minimal-pair seed")
    parser.add_argument("--minimal-pair-report-out", default="", help="Write minimal-pair generation report JSON")
    parser.add_argument("--minimal-pair-sample-out", default="", help="Write auditable minimal-pair sample JSONL")
    parser.add_argument("--minimal-pair-sample-count", type=int, default=20, help="Maximum audit samples to write")
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
        copied_path = copy_tokenizer_json(args.tokenizer_path, args.output)
        print(f"Saved tokenizer artifact to {copied_path}")
    else:
        print(f"Training BPE tokenizer with vocab_size={args.vocab_size}...")
        tokenizer = train_bpe_tokenizer(texts, args.vocab_size, args.output, args.wwm_compatible_tokenizer)

    if args.wwm_compatible_tokenizer:
        scheme = detect_wwm_boundary_scheme(tokenizer)
        print(f"Whole-word masking boundary scheme: {scheme}")

    write_char_features(tokenizer, args.output, args.char_vocab_size, args.char_max_per_token)
    write_minimal_pairs(
        tokenizer,
        texts,
        args.minimal_pair_out,
        args.minimal_pair_corruptions,
        args.minimal_pair_weights,
        args.minimal_pair_morphology,
        args.minimal_pair_max_pairs,
        args.minimal_pair_seed,
        args.minimal_pair_report_out,
        args.minimal_pair_sample_out,
        args.minimal_pair_sample_count,
    )

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

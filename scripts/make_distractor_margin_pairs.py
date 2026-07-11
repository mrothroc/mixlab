#!/usr/bin/env python3
"""Create corpus-derived annotated span-PLL margin pairs for Mixlab.

The generator is deliberately conservative: it only emits pairs when a simple
surface pattern identifies a subject, an opposite-number distractor noun, and
an agreeing auxiliary. It swaps that auxiliary to create the contrast view,
then uses tokenizer offset mappings to annotate the unchanged distractor span.
No parser, tagger, pretrained model, or benchmark template is used.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from prepare import AGREEMENT_FLIPS, induce_morphology, load_tokenizer, read_input_texts, sentence_candidates, split_words


SINGULAR_AUX = {"is", "was", "has", "does"}
PLURAL_AUX = {"are", "were", "have", "do"}
RELATIONAL_NOUNS = {
    "author", "captain", "committee", "director", "editor", "group", "key",
    "list", "member", "owner", "picture", "report", "set", "student", "teacher",
}
RELATIVE_MARKERS = {"that", "which", "who"}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "our", "their"}
IRREGULAR_PLURAL = {"children", "men", "women", "people", "mice", "geese", "teeth", "feet"}
IRREGULAR_SINGULAR = {"news", "series", "species"}
WORD = re.compile(r"^[A-Za-z]+$")


def render_words(words: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Match prepare.py spacing while retaining a character span per token."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    previous = ""
    for token in words:
        space = bool(parts) and not re.match(r"^[^\w\s]$", token) and not previous.endswith(("(", "[", "{", "\"", "'"))
        if space:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        spans.append((start, cursor))
        previous = token
    return "".join(parts), spans


def noun_number(token: str) -> str | None:
    lower = token.lower()
    if not WORD.match(token):
        return None
    if lower in IRREGULAR_PLURAL:
        return "plural"
    if lower in IRREGULAR_SINGULAR:
        return "singular"
    if len(lower) > 2 and lower.endswith("s") and not lower.endswith("ss"):
        return "plural"
    return "singular"


def previous_word(words: list[str], start: int) -> int | None:
    for index in range(start - 1, -1, -1):
        if WORD.match(words[index]):
            return index
    return None


def next_word(words: list[str], start: int, end: int) -> int | None:
    for index in range(start + 1, end):
        if WORD.match(words[index]):
            return index
    return None


def next_content_word(words: list[str], start: int, end: int) -> int | None:
    for index in range(start + 1, end):
        if WORD.match(words[index]) and words[index].lower() not in DETERMINERS:
            return index
    return None


def find_structural_candidate(words: list[str]) -> tuple[str, int, int, int] | None:
    """Return (family, subject_index, distractor_index, auxiliary_index)."""
    lower = [word.lower() for word in words]
    for auxiliary_index, auxiliary in enumerate(lower):
        if auxiliary not in AGREEMENT_FLIPS:
            continue
        aux_number = "singular" if auxiliary in SINGULAR_AUX else "plural" if auxiliary in PLURAL_AUX else None
        if aux_number is None:
            continue

        of_indices = [index for index in range(auxiliary_index) if lower[index] == "of"]
        if of_indices:
            of_index = of_indices[-1]
            subject_index = previous_word(words, of_index)
            distractor_index = next_content_word(words, of_index, auxiliary_index)
            if subject_index is not None and distractor_index is not None:
                subject = noun_number(words[subject_index])
                distractor = noun_number(words[distractor_index])
                if subject == aux_number and distractor is not None and distractor != subject:
                    family = "relational_noun" if lower[subject_index] in RELATIONAL_NOUNS else "attractor_agreement"
                    return family, subject_index, distractor_index, auxiliary_index

        marker_indices = [index for index in range(auxiliary_index) if lower[index] in RELATIVE_MARKERS]
        if marker_indices:
            marker_index = marker_indices[-1]
            subject_index = previous_word(words, marker_index)
            distractor_index = previous_word(words, auxiliary_index)
            if subject_index is not None and distractor_index is not None and subject_index != distractor_index:
                subject = noun_number(words[subject_index])
                distractor = noun_number(words[distractor_index])
                if subject == aux_number and distractor is not None and distractor != subject:
                    return "relative_clause", subject_index, distractor_index, auxiliary_index
    return None


def token_positions_for_span(tokenizer, text: str, span: tuple[int, int]) -> tuple[list[int], list[int]]:
    encoding = tokenizer.encode(text)
    positions = [
        index for index, (start, end) in enumerate(encoding.offsets)
        if start < span[1] and end > span[0]
    ]
    return encoding.ids, positions


def normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def load_contamination_guard(path: str) -> set[str]:
    if not path:
        return set()
    guarded: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            guarded.add(normalized_text(line))
            continue
        if isinstance(record, dict):
            for key in ("sentence", "text", "clean_text", "view_pos_text"):
                value = record.get(key)
                if isinstance(value, str):
                    guarded.add(normalized_text(value))
    return guarded


def make_pairs(tokenizer, texts: list[str], max_pairs: int, seed: int, contamination_guard: set[str]) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    candidates = sentence_candidates(texts)
    rng.shuffle(candidates)
    if max_pairs <= 0:
        max_pairs = min(max(1, len(candidates)), 100_000)
    induced = induce_morphology(texts)
    allowed_swaps = induced.get("agreement_flips", AGREEMENT_FLIPS)
    accepted: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    for source_text in candidates:
        if sum(len(bucket) for bucket in accepted.values()) >= max_pairs * 4:
            break
        words = split_words(source_text)
        candidate = find_structural_candidate(words)
        if candidate is None:
            rejected["no_structural_candidate"] += 1
            continue
        family, subject_index, distractor_index, auxiliary_index = candidate
        pos_text, pos_spans = render_words(words)
        if normalized_text(pos_text) in contamination_guard:
            rejected["contamination_guard"] += 1
            continue
        auxiliary = words[auxiliary_index].lower()
        replacement = allowed_swaps.get(auxiliary)
        if not replacement:
            rejected["no_attested_aux_swap"] += 1
            continue
        neg_words = words[:]
        neg_words[auxiliary_index] = replacement.capitalize() if words[auxiliary_index][:1].isupper() else replacement
        neg_text, neg_spans = render_words(neg_words)
        if normalized_text(neg_text) in contamination_guard:
            rejected["contamination_guard"] += 1
            continue
        pos_ids, pos_positions = token_positions_for_span(tokenizer, pos_text, pos_spans[distractor_index])
        neg_ids, neg_positions = token_positions_for_span(tokenizer, neg_text, neg_spans[distractor_index])
        if not pos_positions or not neg_positions:
            rejected["unmapped_distractor"] += 1
            continue
        target_ids = [pos_ids[index] for index in pos_positions]
        if target_ids != [neg_ids[index] for index in neg_positions]:
            rejected["target_tokenization_changed"] += 1
            continue
        key = (tuple(pos_ids), tuple(neg_ids))
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        direction = noun_number(words[subject_index]) or "unknown"
        accepted[(family, direction)].append(
            {
                "family": family,
                "view_pos": pos_ids,
                "target_pos_positions": pos_positions,
                "view_neg": neg_ids,
                "target_neg_positions": neg_positions,
                "target_ids": target_ids,
                "view_pos_text": pos_text,
                "view_neg_text": neg_text,
                "subject": words[subject_index],
                "distractor": words[distractor_index],
            }
        )

    accepted_counts = {f"{family}:{number}": len(records) for (family, number), records in accepted.items()}
    buckets = [accepted[key] for key in sorted(accepted)]
    for bucket in buckets:
        rng.shuffle(bucket)
    output: list[dict] = []
    cursor = 0
    while buckets and len(output) < max_pairs:
        bucket = buckets[cursor % len(buckets)]
        if bucket:
            record = bucket.pop()
            record["id"] = f"pll_margin_{len(output):08d}"
            output.append(record)
        buckets = [candidate for candidate in buckets if candidate]
        cursor += 1
    report = {
        "seed": seed,
        "max_pairs": max_pairs,
        "written": len(output),
        "accepted_by_family_and_subject_number": accepted_counts,
        "rejected": dict(sorted(rejected.items())),
        "contamination_guard_size": len(contamination_guard),
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create corpus-derived annotated PLL-margin distractor pairs")
    parser.add_argument("--input", required=True, help="Text, JSONL, or directory corpus input")
    parser.add_argument("--tokenizer-path", required=True, help="Existing tokenizer.json")
    parser.add_argument("--output", required=True, help="Output pair JSONL")
    parser.add_argument("--text-field", default="text", help="JSONL text field")
    parser.add_argument("--max-pairs", type=int, default=0, help="Maximum records; 0 auto-selects")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic sampling seed")
    parser.add_argument("--report-out", default="", help="Optional generation report JSON")
    parser.add_argument("--sample-out", default="", help="Optional audit JSONL including source text")
    parser.add_argument("--contamination-guard", default="", help="Optional text/JSONL guard file; matching source sentences are rejected")
    args = parser.parse_args()

    texts = read_input_texts(args.input, args.text_field)
    tokenizer = load_tokenizer(args.tokenizer_path)
    pairs, report = make_pairs(tokenizer, texts, args.max_pairs, args.seed, load_contamination_guard(args.contamination_guard))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for pair in pairs:
            persisted = {key: value for key, value in pair.items() if key not in {"view_pos_text", "view_neg_text", "subject", "distractor"}}
            handle.write(json.dumps(persisted, separators=(",", ":")) + "\n")
    if args.sample_out:
        Path(args.sample_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.sample_out, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, separators=(",", ":")) + "\n")
    report["output_path"] = args.output
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {len(pairs)} PLL-margin pairs to {args.output}")
    print(f"PLL-margin rejected: {report['rejected']}")


if __name__ == "__main__":
    main()

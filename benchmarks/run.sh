#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$ROOT_DIR/benchmarks"
DATA_DIR="$BENCH_DIR/data/shakespeare_char"
RESULTS_DIR="$BENCH_DIR/results"
CONFIG="$BENCH_DIR/gpt2_small.json"
MIXLAB_BIN="${MIXLAB_BIN:-$ROOT_DIR/mixlab}"
BASELINE_VAL_LOSS="${BASELINE_VAL_LOSS:-1.4697}"

TRAIN_GLOB="$DATA_DIR/train_*.bin"
LOG_FILE="$RESULTS_DIR/shakespeare_train.log"
CSV_FILE="$RESULTS_DIR/shakespeare_loss.csv"

mkdir -p "$DATA_DIR" "$RESULTS_DIR"

if [[ ! -x "$MIXLAB_BIN" ]]; then
  echo "mixlab binary not found at $MIXLAB_BIN; building with make build"
  (cd "$ROOT_DIR" && make build)
fi

# Download Tiny Shakespeare if not cached
INPUT_TXT="$DATA_DIR/input.txt"
if [[ ! -f "$INPUT_TXT" ]]; then
  echo "Downloading Tiny Shakespeare"
  curl -fsSL \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
    -o "$INPUT_TXT"
fi

echo "Writing byte-level mixlab shards"
python3 - "$INPUT_TXT" "$DATA_DIR" <<'PY'
from __future__ import annotations

import sys
import struct
from pathlib import Path
from array import array

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def write_shard(path: Path, token_bytes: bytes) -> None:
    header = [0] * HEADER_INTS
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(token_bytes)

    tokens = array("H", (token_bytes[i] for i in range(len(token_bytes))))
    if sys.byteorder != "little":
        tokens.byteswap()

    with path.open("wb") as f:
        f.write(struct.pack(f"<{HEADER_INTS}i", *header))
        tokens.tofile(f)


def main() -> None:
    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    raw = input_path.read_bytes()

    split = int(len(raw) * 0.9)
    train = raw[:split]
    val = raw[split:]

    write_shard(output_dir / "train_00000.bin", train)
    write_shard(output_dir / "val_00000.bin", val)

    print(f"dataset bytes: {len(raw):,}")
    print(f"train tokens:  {len(train):,}")
    print(f"val tokens:    {len(val):,}")
    print("vocab:         256 byte IDs")


if __name__ == "__main__":
    main()
PY

WEIGHTS_FILE="$RESULTS_DIR/shakespeare_weights.st"
SAMPLE_FILE="$RESULTS_DIR/shakespeare_sample.txt"

echo "Training mixlab"
"$MIXLAB_BIN" -mode arch -config "$CONFIG" -train "$TRAIN_GLOB" \
  -safetensors "$WEIGHTS_FILE" 2>&1 | tee "$LOG_FILE"

echo "Generating sample text"
# Prompt with "T" (byte 84) "h" (byte 104) "e" (byte 101) = "The"
"$MIXLAB_BIN" -mode generate -config "$CONFIG" \
  -safetensors-load "$WEIGHTS_FILE" \
  -prompt token_ids:84,104,101 \
  -max-tokens 256 -temperature 0.8 2>&1 | tee "$SAMPLE_FILE"

echo "Extracting loss CSV"
python3 - "$LOG_FILE" "$CSV_FILE" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])

patterns = [
    re.compile(r"step\s+(\d+)\s*\|\s*loss\s+([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"step\s+(\d+)(?:/\d+)?\s+loss=([0-9]*\.?[0-9]+)", re.IGNORECASE),
]
val_pattern = re.compile(r"\bval=([0-9]*\.?[0-9]+)", re.IGNORECASE)

rows: list[tuple[int, float, float | None]] = []
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    match = None
    for pattern in patterns:
        match = pattern.search(line)
        if match:
            break
    if not match:
        continue
    val_match = val_pattern.search(line)
    rows.append((
        int(match.group(1)),
        float(match.group(2)),
        float(val_match.group(1)) if val_match else None,
    ))

if not rows:
    raise SystemExit(f"no step/loss rows found in {log_path}")

with csv_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "train_loss", "val_loss"])
    writer.writerows(rows)

last_step, last_train, last_val = rows[-1]
print(f"wrote {csv_path} ({len(rows)} rows)")
print(f"final step: {last_step}")
print(f"final train loss: {last_train:.4f}")
if last_val is not None:
    print(f"final sampled val loss: {last_val:.4f}")
PY

echo
python3 - "$CSV_FILE" "$BASELINE_VAL_LOSS" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
baseline = float(sys.argv[2])

with csv_path.open(newline="") as f:
    rows = list(csv.DictReader(f))

last = rows[-1]
last_train = float(last["train_loss"])
last_val = float(last["val_loss"]) if last["val_loss"] else None

# Find best sampled val loss
val_rows = [(r["step"], float(r["val_loss"])) for r in rows if r.get("val_loss")]
best_step, best_val = min(val_rows, key=lambda x: x[1]) if val_rows else (None, None)

# Parse elapsed time and throughput from log
import re as _re
log_text = (csv_path.parent / "shakespeare_train.log").read_text(errors="replace")
elapsed_match = _re.search(r"\((\d+)m([\d.]+)s\)\s*$", log_text, _re.MULTILINE)
elapsed_sec = None
if elapsed_match:
    elapsed_sec = int(elapsed_match.group(1)) * 60 + float(elapsed_match.group(2))

# Parse batch_tokens from config line
bt_match = _re.search(r"batch_tokens=(\d+)", log_text)
batch_tokens = int(bt_match.group(1)) if bt_match else None
steps_match = _re.search(r"steps=(\d+)", log_text)
total_steps = int(steps_match.group(1)) if steps_match else None

print("Benchmark summary")
print("-----------------")
print(f"nanoGPT published best val loss:  {baseline:.4f}")
print(f"mixlab final train loss:          {last_train:.4f}")
if best_val is not None:
    delta = best_val - baseline
    print(f"mixlab best sampled val loss:     {best_val:.4f} at step {best_step} ({delta:+.4f} vs nanoGPT)")
if last_val is not None and last_val != best_val:
    print(f"mixlab final sampled val loss:    {last_val:.4f}")
if elapsed_sec and batch_tokens and total_steps:
    tok_per_sec = batch_tokens * total_steps / elapsed_sec
    print(f"throughput:                       {tok_per_sec:,.0f} tokens/sec")
    print(f"total time:                       {elapsed_sec:.0f}s ({elapsed_sec/60:.1f} min)")
print(f"log: {csv_path.parent / 'shakespeare_train.log'}")
print(f"csv: {csv_path}")
PY

if python3 -c "import matplotlib" >/dev/null 2>&1; then
  python3 "$BENCH_DIR/plot.py" "$CSV_FILE"
else
  echo "matplotlib not installed; skipping plot. Install with: pip install matplotlib"
fi

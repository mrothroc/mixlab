#!/usr/bin/env bash
# Eval a trained model and export both per-token NLLs and per-token target
# ranks in a single GPU pass. The two files are aligned position-by-position
# so downstream Hit@K / MRR / rank-conditional calibration analyses can use
# them together.
#
# Usage:
#   examples/eval-with-ranks.sh <config> <weights.safetensors> <train_glob> [lut_dir]
#
# Example:
#   examples/eval-with-ranks.sh \
#     examples/plain_3L.json \
#     model_cache/plain_3L/weights.safetensors \
#     'data/example/train_*.bin'

set -euo pipefail

config=${1:?"usage: $0 <config> <weights> <train_glob> [lut_dir]"}
weights=${2:?"usage: $0 <config> <weights> <train_glob> [lut_dir]"}
train_glob=${3:?"usage: $0 <config> <weights> <train_glob> [lut_dir]"}
lut_dir=${4:-data}

./mixlab -mode eval \
  -config       "$config" \
  -safetensors-load "$weights" \
  -train        "$train_glob" \
  -lut-dir      "$lut_dir" \
  -logprobs-out logprobs.bin \
  -ranks-out    ranks.bin

echo "wrote logprobs.bin and ranks.bin"
echo "compute Hit@K with the snippet in the README eval section"

#!/usr/bin/env bash
# Eval a trained model and export per-token NLLs, target ranks, and
# candidate-side uncertainty metrics in a single GPU pass. The three files are
# aligned position-by-position.
#
# Usage:
#   examples/eval-with-uncertainty.sh <config> <weights.safetensors> <train_glob> [lut_dir]
#
# Example:
#   examples/eval-with-uncertainty.sh \
#     examples/plain_3L.json \
#     model_cache/plain_3L/weights.safetensors \
#     'data/example/train_*.bin'

set -euo pipefail

config=${1:?"usage: $0 <config> <weights> <train_glob> [lut_dir]"}
weights=${2:?"usage: $0 <config> <weights> <train_glob> [lut_dir]"}
train_glob=${3:?"usage: $0 <config> <weights> <train_glob> [lut_dir]"}
lut_dir=${4:-data}

./mixlab -mode eval \
  -config          "$config" \
  -safetensors-load "$weights" \
  -train           "$train_glob" \
  -lut-dir         "$lut_dir" \
  -logprobs-out    logprobs.bin \
  -ranks-out       ranks.bin \
  -uncertainty-out uncertainty.bin

echo "wrote logprobs.bin, ranks.bin, and uncertainty.bin"
echo "compute Hit@K and selective-prediction metrics with the snippets in the README eval section"

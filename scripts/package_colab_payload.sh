#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-/home/seyominaoto/archimedes-data/colab/archimedes_math_colab_payload.tar.gz}
ROOT=/home/seyominaoto/archimedes
DATA=/home/seyominaoto/archimedes-data

mkdir -p "$(dirname "$OUT")"

tar -czf "$OUT" \
  -C "$ROOT" \
  README.md configs scripts src \
  -C "$DATA" \
  tokenizers/archimedes_math_bpe_32768 \
  shards/math_tokens_v1

du -h "$OUT"


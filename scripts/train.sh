#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 MODEL_PATH DATASET_PATH OUTPUT_DIR [additional train.py arguments...]" >&2
  exit 2
fi

MODEL_PATH=$1
DATASET_PATH=$2
OUTPUT_DIR=$3
shift 3

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
NUM_PROCESSES=${NUM_PROCESSES:-4}

accelerate launch \
  --config_file "$REPO_ROOT/configs/accelerate_zero2.yaml" \
  --num_processes "$NUM_PROCESSES" \
  "$REPO_ROOT/src/train.py" \
  --model-name-or-path "$MODEL_PATH" \
  --dataset-name "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

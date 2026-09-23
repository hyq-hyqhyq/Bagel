#!/usr/bin/env bash
set -Eeuo pipefail

# Simple entrypoint for the full perspective evaluation.
# It evaluates train20 + test20 (40 groups), with both good and bad inputs,
# and writes 80 predictions plus Dice summaries.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_perspective_checkpoint.sh \
    --checkpoint /path/to/checkpoints/0006000 \
    --dataset 1

Only two arguments are required:
  --checkpoint  A numbered checkpoint directory, or its model.safetensors.
  --dataset     1 or 2.

Fixed defaults:
  GPUs: 0,1,2,3
  seed: 52

Dataset aliases:
  1: perspective_strong train20 + test20
  2: perspective_combined SAM-postprocessed train20 + test20
EOF
}

CHECKPOINT=""
DATASET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT=${2:?"--checkpoint requires a value"}; shift 2 ;;
    --dataset) DATASET=${2:?"--dataset requires 1 or 2"}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z $CHECKPOINT || -z $DATASET ]]; then
  echo "Both --checkpoint and --dataset are required." >&2
  usage >&2
  exit 2
fi

if [[ $DATASET != 1 && $DATASET != 2 ]]; then
  echo "--dataset must be 1 or 2: $DATASET" >&2
  exit 2
fi

CHECKPOINT=${CHECKPOINT%/}
if [[ $(basename -- "$CHECKPOINT") == model.safetensors ]]; then
  CHECKPOINT=$(dirname -- "$CHECKPOINT")
fi

STEP_DIR=$(basename -- "$CHECKPOINT")
if [[ ! $STEP_DIR =~ ^[0-9]+$ ]]; then
  echo "Checkpoint directory must have a numeric name, e.g. 0006000: $CHECKPOINT" >&2
  exit 2
fi

if [[ $(basename -- "$(dirname -- "$CHECKPOINT")") != checkpoints ]]; then
  echo "Expected checkpoint path like <run>/checkpoints/0006000: $CHECKPOINT" >&2
  exit 2
fi

MODEL_FILE=$CHECKPOINT/model.safetensors
if [[ ! -s $MODEL_FILE ]]; then
  echo "Missing or empty checkpoint model: $MODEL_FILE" >&2
  exit 2
fi

RUN_DIR=$(dirname -- "$(dirname -- "$CHECKPOINT")")
STEP=$((10#$STEP_DIR))
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$SCRIPT_DIR/eval_perspective_train20_test20.sh" \
  --run-dir "$RUN_DIR" \
  --step "$STEP" \
  --seed 52 \
  --dataset "$DATASET" \
  --gpus 0,1,2,3

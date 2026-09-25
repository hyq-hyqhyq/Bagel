#!/usr/bin/env bash
set -euo pipefail

# Teacher-forced tokenwise recipe for visual Inspection/Conclusion data.
# The base launcher owns the model, FSDP and W&B defaults.  Options supplied
# by the caller are appended last, so paths, GPUs, steps and weights remain
# easy to override without editing either script.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "${SCRIPT_DIR}/train_perspective_sam_judge_4gpu.sh" \
  --data-root /data/bagel/data/perspective_combined_train1400_sam_postprocessed_20260920_judge_inspect_v1 \
  --gpus 0,1,2,3 \
  --freeze-vae False \
  --freeze-vit False \
  --text-dropout 0.0 \
  --vae-dropout 0.0 \
  --vit-dropout 0.0 \
  --tokenwise-judgment True \
  --judgment-rollout-probability 0.0 \
  --reason-base-ce-weight 1.0 \
  --inspection-ce-weight 4.0 \
  --check-ce-weight 8.0 \
  --conclusion-ce-weight 4.0 \
  --global-ce-weight 8.0 \
  --run-tag inspect-tokenwise \
  "$@"

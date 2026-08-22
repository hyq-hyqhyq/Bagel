#!/usr/bin/env bash

set -euo pipefail

BAGEL_ROOT=${BAGEL_ROOT:-/root/autodl-tmp/bagel}
REPO=${REPO:-$BAGEL_ROOT/repo/Bagel}
PYTHON=${PYTHON:-$BAGEL_ROOT/envs/bagel/bin/python}
WATCHER_ROOT=${WATCHER_ROOT:-$BAGEL_ROOT/outputs/hf_stage3_3000_raw_checkpoint_watcher}

RUN_EXPLANATION_ROOT=${RUN_EXPLANATION_ROOT:-$BAGEL_ROOT/outputs/stage3_pass_train3000_explanation_heatmap_full_30k_a800_gpu4567/checkpoints}
RUN_HEATMAP_ONLY_ROOT=${RUN_HEATMAP_ONLY_ROOT:-$BAGEL_ROOT/outputs/stage3_pass_train3000_heatmap_only_full_30k_a800_gpu0123/checkpoints}

export HF_HOME=${HF_HOME:-$BAGEL_ROOT/hf_cache}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$WATCHER_ROOT" "$HF_HOME"

test -x "$PYTHON" || {
  echo "Missing Python environment: $PYTHON" >&2
  exit 1
}
test -s "$REPO/scripts/hf_raw_model_checkpoint_watcher.py" || {
  echo "Missing watcher script under $REPO" >&2
  exit 1
}

exec "$PYTHON" -u "$REPO/scripts/hf_raw_model_checkpoint_watcher.py" \
  --watch \
    stage3_explanation_heatmap \
    "$RUN_EXPLANATION_ROOT" \
    checkpoint/stage3_pass_train3000_explanation_heatmap_full_finetune_bf16 \
  --watch \
    stage3_heatmap_only \
    "$RUN_HEATMAP_ONLY_ROOT" \
    checkpoint/stage3_pass_train3000_heatmap_only_full_finetune_bf16 \
  --repo-id dfgfhdhhhghg/bagel \
  --repo-type dataset \
  --optimizer-shards 4 \
  --min-step 8000 \
  --max-step 30000 \
  --step-multiple 2000 \
  --poll-seconds 30 \
  --stable-seconds 180 \
  --stop-after-step 30000 \
  --keep-latest-local 2 \
  --state-file "$WATCHER_ROOT/state.json" \
  --log-file "$WATCHER_ROOT/watcher.log"

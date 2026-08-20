#!/usr/bin/env bash

set -euo pipefail

BAGEL_ROOT=${BAGEL_ROOT:-/root/autodl-tmp/bagel}
REPO=${REPO:-$BAGEL_ROOT/repo/Bagel}
PYTHON=${PYTHON:-$BAGEL_ROOT/envs/bagel/bin/python}
WATCHER_ROOT=${WATCHER_ROOT:-$BAGEL_ROOT/outputs/hf_raw_checkpoint_watcher}

RUN_BINARY_ROOT=${RUN_BINARY_ROOT:-$BAGEL_ROOT/outputs/sanity_patch_binary_prompt_ce05_mse10_full_30k_bf16_4gpu/checkpoints}
RUN_SENSENOVA_ROOT=${RUN_SENSENOVA_ROOT:-$BAGEL_ROOT/outputs/sanity_patch_sensenova_tok24576_gpu4567_20260820_130921/checkpoints}

export HF_HOME=${HF_HOME:-$BAGEL_ROOT/hf_cache}
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
    binary_prompt_ce05_mse10 \
    "$RUN_BINARY_ROOT" \
    checkpoint/sanity_patch_binary_prompt_ce0p5_mse10_full_finetune_bf16 \
  --watch \
    sensenova_tok24576 \
    "$RUN_SENSENOVA_ROOT" \
    checkpoint/sanity_patch_sensenova_tok24576_gpu4567_full_finetune_bf16 \
  --repo-id dfgfhdhhhghg/bagel \
  --repo-type dataset \
  --optimizer-shards 4 \
  --min-step 2000 \
  --max-step 30000 \
  --step-multiple 2000 \
  --poll-seconds 30 \
  --stable-seconds 180 \
  --stop-after-step 30000 \
  --state-file "$WATCHER_ROOT/state.json" \
  --log-file "$WATCHER_ROOT/watcher.log"

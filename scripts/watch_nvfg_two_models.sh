#!/usr/bin/env bash
set -u

cd /data/bagel/repo/Bagel
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

# inference imports sanity_patch even though this experiment does not use it.
SANITY_ROOT=/tmp/bagel_sanity_patch_runtime
if [ ! -f "$SANITY_ROOT/sanity_patch/settings.py" ]; then
    rm -rf "$SANITY_ROOT"
    mkdir -p "$SANITY_ROOT"
    git archive HEAD sanity_patch | tar -x -C "$SANITY_ROOT"
fi
export PYTHONPATH="$SANITY_ROOT:$PWD:${PYTHONPATH:-}"

MODEL_PATH=/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT
DATA_ROOT=/data/bagel/runs/perspective_strong/test20_fresh_score_package
META_PATH=$DATA_ROOT/metadata/test.jsonl
BASELINE_DIR=/data/bagel/repo/Bagel/results/nvfg_8g_long50k_4x2_v2
CURRENT_DIR=/data/bagel/repo/Bagel/results/perspective_strong_nvfg_4gpu_30k_v2
OUT_ROOT=/data/bagel/repo/Bagel/results/eval_watch_nvfg_two_models_seed1_23_52_64_321
CSV_PATH=$OUT_ROOT/dice_summary.csv
LOCK_DIR=$OUT_ROOT/.locks
SEEDS=(1 23 52 64 321)
START_STEP=8000
POLL_SECONDS=60

mkdir -p "$OUT_ROOT/logs" "$LOCK_DIR"
[ -f "$CSV_PATH" ] || echo 'model,step,seed,good_dice,bad_dice,total_single_dice' > "$CSV_PATH"

dice_mean() {
    /data/bagel/conda/envs/bagel/bin/python - "$1" <<'PY'
import sys
from pathlib import Path
from PIL import Image
import numpy as np
root = Path(sys.argv[1]); vals = []
for pred in sorted(root.rglob("prediction.png")):
    target = pred.parent / "target.png"
    if not target.exists(): continue
    a = np.asarray(Image.open(pred).convert("L")) >= 128
    b = np.asarray(Image.open(target).convert("L")) >= 128
    den = int(a.sum() + b.sum())
    vals.append(1.0 if den == 0 else float(2 * (a & b).sum() / den))
print("NA" if not vals else f"{sum(vals)/len(vals):.6f}")
PY
}

run_eval() {
    local gpu=$1 tag=$2 step=$3 seed=$4 task=$5 ckpt=$6
    local out="$OUT_ROOT/$tag/step$(printf '%07d' "$step")/seed${seed}/${task}"
    local log="$OUT_ROOT/logs/${tag}_step${step}_seed${seed}_${task}.log"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" python inference_reason_heatmap.py \
      --model_path "$MODEL_PATH" --checkpoint_path "$ckpt" \
      --data_dir "$DATA_ROOT" --metadata_path "$META_PATH" \
      --row_index 0 --num_samples 20 --sample_type "$task" \
      --prompt_domain perspective --prompt_recipe single_pair_refine \
      --output_dir "$out" --seed "$seed" --num_timesteps 50 \
      --timestep_shift 4.0 --cfg_text_scale 4.0 --cfg_img_scale 1.0 \
      --binary_threshold 127 > "$log" 2>&1
}

evaluate() {
    local tag=$1 base=$2 step=$3
    local ckpt="$base/checkpoints/$(printf '%07d' "$step")/model.safetensors"
    local stepout="$OUT_ROOT/$tag/step$(printf '%07d' "$step")"
    local lock="$LOCK_DIR/${tag}_step$(printf '%07d' "$step").done"
    [ -f "$ckpt" ] && [ ! -f "$lock" ] || return 0
    echo "[$(date '+%F %T')] START $tag step=$step"
    local i=0; local -a pids=()
    for seed in "${SEEDS[@]}"; do
        for task in good bad; do
            run_eval $((i % 4)) "$tag" "$step" "$seed" "$task" "$ckpt" &
            pids+=("$!"); i=$((i+1))
            if [ "${#pids[@]}" -eq 4 ]; then
                for p in "${pids[@]}"; do wait "$p"; done
                pids=()
            fi
        done
    done
    for p in "${pids[@]}"; do wait "$p"; done

    local rows=0
    for seed in "${SEEDS[@]}"; do
        local good bad total
        good=$(dice_mean "$stepout/seed${seed}/good")
        bad=$(dice_mean "$stepout/seed${seed}/bad")
        if [ "$good" != NA ] && [ "$bad" != NA ]; then
            total=$(awk -v g="$good" -v b="$bad" 'BEGIN { printf "%.6f", (g+b)/2 }')
            echo "$tag,$step,$seed,$good,$bad,$total" >> "$CSV_PATH"
            printf '%s step=%s seed=%s good=%s bad=%s total=%s\n' "$tag" "$step" "$seed" "$good" "$bad" "$total"
            rows=$((rows+1))
        fi
    done
    [ "$rows" -eq "${#SEEDS[@]}" ] && touch "$lock"
}

scan() {
    local tag=$1 base=$2
    [ -d "$base/checkpoints" ] || return 0
    find "$base/checkpoints" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9][0-9][0-9][0-9][0-9]' -printf '%f\n' 2>/dev/null | sort -n |
    while read -r name; do
        local step=$((10#$name))
        [ "$step" -ge "$START_STEP" ] && [ $((step % 2000)) -eq 0 ] && evaluate "$tag" "$base" "$step"
    done
}

echo "Watching baseline=$BASELINE_DIR current=$CURRENT_DIR seeds=${SEEDS[*]}"
while true; do
    scan baseline "$BASELINE_DIR"
    scan current "$CURRENT_DIR"
    sleep "$POLL_SECONDS"
done

#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate one full-model checkpoint on the fixed perspective train20/test20
# suites. Each group is evaluated twice: single-good heatmap and single-bad
# heatmap. Dataset aliases are defined in resolve_dataset().

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval_perspective_train20_test20.sh \
    --run-dir RUN_DIR \
    --step STEP \
    --seed SEED \
    --dataset {1|2} \
    --gpus GPU_LIST \
    [--output-dir OUTPUT_DIR]

Required arguments:
  --run-dir     Training result directory containing checkpoints/.
  --step        Checkpoint step, for example 14000.
  --seed        Inference seed, for example 52.
  --dataset     1 or 2 (see dataset aliases below).
  --gpus        Comma-separated GPU indexes, for example 0,1,2,3.

Optional:
  --output-dir  Evaluation output directory. If omitted, a deterministic
                directory name is generated under <repo>/results/.

Dataset aliases:
  1: /data/bagel/runs/perspective_strong/
     perspective_strong_train20_test20/perspective_strong_train20_test20
  2: /data/bagel/data/
     perspective_combined_train1400_sam_postprocessed_20260920

Both aliases evaluate rows 0-19 from train.jsonl and all 20 rows from
test.jsonl, for both good and bad tasks (80 inference outputs in total).
EOF
}

RUN_DIR=""
STEP=""
SEED=""
DATASET_ALIAS=""
GPU_SPEC=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR=${2:?"--run-dir requires a value"}
            shift 2
            ;;
        --step)
            STEP=${2:?"--step requires a value"}
            shift 2
            ;;
        --seed)
            SEED=${2:?"--seed requires a value"}
            shift 2
            ;;
        --dataset)
            DATASET_ALIAS=${2:?"--dataset requires 1 or 2"}
            shift 2
            ;;
        --gpus)
            GPU_SPEC=${2:?"--gpus requires a value"}
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=${2:?"--output-dir requires a value"}
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for required_name in RUN_DIR STEP SEED DATASET_ALIAS GPU_SPEC; do
    if [[ -z ${!required_name} ]]; then
        echo "Missing required argument: $required_name" >&2
        usage >&2
        exit 2
    fi
done

if [[ ! $STEP =~ ^[0-9]+$ ]] || (( STEP < 0 )); then
    echo "--step must be a non-negative integer: $STEP" >&2
    exit 2
fi

if [[ ! $SEED =~ ^[0-9]+$ ]]; then
    echo "--seed must be a non-negative integer: $SEED" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODEL_PATH=/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT
SAMPLES_PER_SPLIT=20

RUN_DIR=${RUN_DIR%/}
if [[ $(basename -- "$RUN_DIR") == checkpoints ]]; then
    RUN_DIR=$(dirname -- "$RUN_DIR")
fi

resolve_dataset() {
    case "$DATASET_ALIAS" in
        1)
            DATA_ROOT=/data/bagel/runs/perspective_strong/perspective_strong_train20_test20/perspective_strong_train20_test20
            TRAIN_META=$DATA_ROOT/metadata/train.jsonl
            TEST_META=$DATA_ROOT/metadata/test.jsonl
            ;;
        2)
            DATA_ROOT=/data/bagel/data/perspective_combined_train1400_sam_postprocessed_20260920
            TRAIN_META=$DATA_ROOT/train.jsonl
            TEST_META=$DATA_ROOT/test.jsonl
            ;;
        *)
            echo "--dataset must be 1 or 2: $DATASET_ALIAS" >&2
            exit 2
            ;;
    esac
}

resolve_dataset

STEP_NUMBER=$((10#$STEP))
STEP_PAD=$(printf '%07d' "$STEP_NUMBER")
CHECKPOINT=$RUN_DIR/checkpoints/$STEP_PAD/model.safetensors

GPU_SPEC=${GPU_SPEC//,/ }
read -r -a GPUS <<< "$GPU_SPEC"

if (( ${#GPUS[@]} == 0 )); then
    echo "No GPUs were provided." >&2
    exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if [[ ! $gpu =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index: $gpu" >&2
        exit 2
    fi
    if [[ -n ${SEEN_GPUS[$gpu]:-} ]]; then
        echo "Duplicate GPU index: $gpu" >&2
        exit 2
    fi
    SEEN_GPUS[$gpu]=1
done

if (( ${#GPUS[@]} > 80 )); then
    echo "At most 80 GPUs are supported for this 80-output evaluation." >&2
    exit 2
fi

RUN_TAG=$(basename -- "$RUN_DIR")
GPU_TAG=$(IFS=_; echo "${GPUS[*]}")

if [[ -z $OUTPUT_DIR ]]; then
    OUTPUT_DIR=$ROOT/results/eval_${RUN_TAG}_step${STEP_NUMBER}_dataset${DATASET_ALIAS}_train20_test20_seed${SEED}_gpu${GPU_TAG}
fi

OUTPUT_DIR=${OUTPUT_DIR%/}
PREDICTION_ROOT=$OUTPUT_DIR/predictions/step$STEP_PAD/seed$SEED
ATTEMPT_ID=$(date '+%Y%m%d_%H%M%S')_$$
LOG_DIR=$OUTPUT_DIR/logs/$ATTEMPT_ID
STATUS_DIR=$OUTPUT_DIR/status/$ATTEMPT_ID

for required_file in "$CHECKPOINT" "$TRAIN_META" "$TEST_META"; do
    if [[ ! -s $required_file ]]; then
        echo "Missing or empty required file: $required_file" >&2
        exit 2
    fi
done

train_rows=$(wc -l < "$TRAIN_META")
test_rows=$(wc -l < "$TEST_META")
if (( train_rows < SAMPLES_PER_SPLIT || test_rows < SAMPLES_PER_SPLIT )); then
    echo "Dataset $DATASET_ALIAS does not contain 20 train and 20 test rows." >&2
    echo "train_rows=$train_rows test_rows=$test_rows" >&2
    exit 2
fi

cd "$ROOT"
source /data/bagel/conda/etc/profile.d/conda.sh
conda activate bagel

SANITY_ROOT=/tmp/bagel_sanity_patch_runtime
mkdir -p "$SANITY_ROOT"
if [[ ! -f $SANITY_ROOT/sanity_patch/settings.py ]]; then
    if [[ -f $ROOT/sanity_patch/settings.py ]]; then
        cp -a "$ROOT/sanity_patch" "$SANITY_ROOT/"
    else
        git archive HEAD sanity_patch | tar -x -C "$SANITY_ROOT"
    fi
fi

export PYTHONPATH="$SANITY_ROOT:$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1

python -c "from sanity_patch.settings import SANITY_PATCH_PROMPT; from sanity_patch.mask_utils import to_binary_mask; print('sanity_patch import OK')"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
for split in train test; do
    mkdir -p "$PREDICTION_ROOT/$split/good" "$PREDICTION_ROOT/$split/bad"
done

cat <<EOF
===== CONFIGURATION =====
run_dir=$RUN_DIR
checkpoint=$CHECKPOINT
step=$STEP_NUMBER
seed=$SEED
dataset=$DATASET_ALIAS
data_root=$DATA_ROOT
train_metadata=$TRAIN_META
test_metadata=$TEST_META
gpus=${GPUS[*]}
output=$OUTPUT_DIR
attempt=$ATTEMPT_ID
EOF

run_job() {
    local gpu=$1
    local split=$2
    local task=$3
    local row_start=$4
    local sample_count=$5
    local shard=$6
    local metadata
    local job_name

    if [[ $split == train ]]; then
        metadata=$TRAIN_META
    else
        metadata=$TEST_META
    fi

    job_name=gpu${gpu}_${split}_${task}_rows${row_start}_$((row_start + sample_count - 1))_shard${shard}
    echo "START GPU=$gpu SPLIT=$split TASK=$task ROWS=$row_start-$((row_start + sample_count - 1))"

    set +e
    CUDA_VISIBLE_DEVICES=$gpu python inference_reason_heatmap.py \
        --model_path "$MODEL_PATH" \
        --checkpoint_path "$CHECKPOINT" \
        --data_dir "$DATA_ROOT" \
        --metadata_path "$metadata" \
        --row_index "$row_start" \
        --num_samples "$sample_count" \
        --sample_type "$task" \
        --prompt_domain perspective \
        --prompt_recipe single_pair_refine \
        --output_dir "$PREDICTION_ROOT/$split/$task" \
        --seed "$SEED" \
        --num_timesteps 50 \
        --timestep_shift 4.0 \
        --cfg_text_scale 4.0 \
        --cfg_img_scale 1.0 \
        --binary_threshold 127 \
        > "$LOG_DIR/$job_name.log" 2>&1
    local rc=$?
    set -e

    echo "$rc" > "$STATUS_DIR/$job_name.status"
    echo "FINISHED GPU=$gpu SPLIT=$split TASK=$task STATUS=$rc"
    return "$rc"
}

wait_for_jobs() {
    local failed=0
    local pid
    for pid in "$@"; do
        wait "$pid" || failed=1
    done
    return "$failed"
}

SPLITS=(train train test test)
TASKS=(good bad good bad)
GPU_COUNT=${#GPUS[@]}
failed=0

if (( GPU_COUNT <= 4 )); then
    # With four GPUs this maps exactly to train-good, train-bad,
    # test-good, test-bad. With fewer GPUs it runs in safe batches.
    for ((batch_start = 0; batch_start < 4; batch_start += GPU_COUNT)); do
        pids=()
        for ((slot = 0; slot < GPU_COUNT; slot++)); do
            category=$((batch_start + slot))
            (( category < 4 )) || break
            run_job "${GPUS[$slot]}" "${SPLITS[$category]}" "${TASKS[$category]}" 0 "$SAMPLES_PER_SPLIT" 0 &
            pids+=("$!")
        done
        wait_for_jobs "${pids[@]}" || failed=1
    done
else
    # With more than four GPUs, give every category at least one GPU and
    # distribute the remaining GPUs as extra contiguous row shards.
    base_workers=$((GPU_COUNT / 4))
    extra_workers=$((GPU_COUNT % 4))
    gpu_cursor=0
    pids=()

    for ((category = 0; category < 4; category++)); do
        workers=$base_workers
        (( category < extra_workers )) && workers=$((workers + 1))
        if (( workers > SAMPLES_PER_SPLIT )); then
            echo "Too many GPUs assigned to one 20-row category." >&2
            exit 2
        fi

        base_count=$((SAMPLES_PER_SPLIT / workers))
        extra_rows=$((SAMPLES_PER_SPLIT % workers))
        row_start=0

        for ((shard = 0; shard < workers; shard++)); do
            sample_count=$base_count
            (( shard < extra_rows )) && sample_count=$((sample_count + 1))
            run_job "${GPUS[$gpu_cursor]}" "${SPLITS[$category]}" "${TASKS[$category]}" "$row_start" "$sample_count" "$shard" &
            pids+=("$!")
            row_start=$((row_start + sample_count))
            gpu_cursor=$((gpu_cursor + 1))
        done
    done

    wait_for_jobs "${pids[@]}" || failed=1
fi

echo "===== JOB STATUS ====="
cat "$STATUS_DIR"/*.status

if (( failed != 0 )) || grep -qv '^0$' "$STATUS_DIR"/*.status; then
    echo "At least one inference job failed. Logs: $LOG_DIR" >&2
    exit 1
fi

export EVAL_OUTPUT_DIR=$OUTPUT_DIR
export EVAL_PREDICTION_ROOT=$PREDICTION_ROOT
export EVAL_STEP=$STEP_NUMBER
export EVAL_SEED=$SEED

python - <<'PY'
import csv
import glob
import os

import numpy as np
from PIL import Image

output_dir = os.environ["EVAL_OUTPUT_DIR"]
prediction_root = os.environ["EVAL_PREDICTION_ROOT"]
step = int(os.environ["EVAL_STEP"])
seed = int(os.environ["EVAL_SEED"])


def mask(path):
    return np.asarray(Image.open(path).convert("L")) >= 127


def dice(prediction, target):
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(prediction, target).sum())
    return 2.0 * intersection / denominator


per_sample = []
means = {}

for split in ("train", "test"):
    for task in ("good", "bad"):
        paths = sorted(
            glob.glob(
                os.path.join(
                    prediction_root,
                    split,
                    task,
                    "*",
                    "prediction.png",
                )
            )
        )
        if len(paths) != 20:
            raise RuntimeError(
                f"Expected 20 {split}/{task} predictions, found {len(paths)}"
            )

        values = []
        for prediction_path in paths:
            sample_dir = os.path.dirname(prediction_path)
            target_path = os.path.join(sample_dir, "target.png")
            reason_path = os.path.join(sample_dir, "reason.txt")
            score_path = os.path.join(sample_dir, "score.txt")

            for required_path in (target_path, reason_path, score_path):
                if not os.path.isfile(required_path):
                    raise FileNotFoundError(required_path)

            value = dice(mask(prediction_path), mask(target_path))
            values.append(value)
            per_sample.append(
                {
                    "split": split,
                    "task": task,
                    "sample": os.path.basename(sample_dir),
                    "dice": value,
                }
            )

        means[(split, task)] = float(np.mean(values))

per_sample_path = os.path.join(output_dir, "dice_per_sample.csv")
with open(per_sample_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["split", "task", "sample", "dice"],
    )
    writer.writeheader()
    writer.writerows(per_sample)

summary_rows = []
for split in ("train", "test"):
    good = means[(split, "good")]
    bad = means[(split, "bad")]
    summary_rows.append(
        {
            "split": split,
            "step": step,
            "seed": seed,
            "good_dice": good,
            "bad_dice": bad,
            "mean": (good + bad) / 2.0,
            "good_count": 20,
            "bad_count": 20,
        }
    )

overall_good = (means[("train", "good")] + means[("test", "good")]) / 2.0
overall_bad = (means[("train", "bad")] + means[("test", "bad")]) / 2.0
summary_rows.append(
    {
        "split": "overall",
        "step": step,
        "seed": seed,
        "good_dice": overall_good,
        "bad_dice": overall_bad,
        "mean": (overall_good + overall_bad) / 2.0,
        "good_count": 40,
        "bad_count": 40,
    }
)

summary_path = os.path.join(output_dir, "dice_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print("===== DICE SUMMARY =====")
for row in summary_rows:
    print(
        f"{row['split']:7s} "
        f"good={row['good_dice']:.6f} "
        f"bad={row['bad_dice']:.6f} "
        f"mean={row['mean']:.6f}"
    )
print(f"Saved: {summary_path}")
print(f"Saved: {per_sample_path}")
PY

echo "===== COMPLETENESS ====="
echo "prediction: $(find "$PREDICTION_ROOT" -type f -name prediction.png | wc -l)"
echo "reason:     $(find "$PREDICTION_ROOT" -type f -name reason.txt | wc -l)"
echo "score:      $(find "$PREDICTION_ROOT" -type f -name score.txt | wc -l)"
echo "Evaluation complete: $OUTPUT_DIR"

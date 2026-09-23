#!/usr/bin/env bash
set -euo pipefail

cd /data/bagel/repo/Bagel
source /data/bagel/conda/etc/profile.d/conda.sh
conda activate bagel

# Make the repository modules and the local sanity patch visible to every
# torchrun worker.  This avoids ``ModuleNotFoundError: data`` and the
# intermittent sanity_patch import issue after a fresh shell/login.
export PYTHONPATH=/data/bagel/repo/Bagel:/tmp/bagel_sanity_patch_runtime:${PYTHONPATH:-}
mkdir -p /tmp/bagel_sanity_patch_runtime
if [ ! -f /tmp/bagel_sanity_patch_runtime/sanity_patch/settings.py ]; then
  if [ -f /data/bagel/repo/Bagel/sanity_patch/settings.py ]; then
    cp -a /data/bagel/repo/Bagel/sanity_patch /tmp/bagel_sanity_patch_runtime/
  else
    git archive HEAD sanity_patch | tar -x -C /tmp/bagel_sanity_patch_runtime
  fi
fi
python -c "from sanity_patch.settings import SANITY_PATCH_PROMPT; from sanity_patch.mask_utils import to_binary_mask; import data; print('sanity_patch/data import OK')"

# ===== Edit only this block for a new run =====
GPU_LIST=0,1,2,3
DATA_ROOT=/data/bagel/data/perspective_combined_train1400_sam_postprocessed_20260920
METADATA_PATH=""
FREEZE_VAE=False
FREEZE_VIT=False
FREEZE_LLM=False
FREEZE_UND=False
TEXT_DROPOUT=0.0
VAE_DROPOUT=0.0
VIT_DROPOUT=0.0
TOTAL_STEPS=30000
SAVE_EVERY=2000
LR=2e-5
GLOBAL_SEED=4396
DATA_SEED=42
RUN_TAG=judge
# =============================================

# Optional command-line overrides. Example:
#   bash scripts/train_perspective_sam_judge_4gpu.sh \
#     --data-root /data/bagel/data/myset --total-steps 14000 --run-tag ablation
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --metadata-path) METADATA_PATH="$2"; METADATA_PATH_EXPLICIT=1; shift 2 ;;
    --freeze-vae) FREEZE_VAE="$2"; shift 2 ;;
    --freeze-vit) FREEZE_VIT="$2"; shift 2 ;;
    --freeze-llm) FREEZE_LLM="$2"; shift 2 ;;
    --freeze-und) FREEZE_UND="$2"; shift 2 ;;
    --text-dropout) TEXT_DROPOUT="$2"; shift 2 ;;
    --vae-dropout) VAE_DROPOUT="$2"; shift 2 ;;
    --vit-dropout) VIT_DROPOUT="$2"; shift 2 ;;
    --total-steps) TOTAL_STEPS="$2"; shift 2 ;;
    --save-every) SAVE_EVERY="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --global-seed) GLOBAL_SEED="$2"; shift 2 ;;
    --data-seed) DATA_SEED="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    -h|--help)
      sed -n '/^# Optional command-line overrides/,/^while /p' "$0"
      echo "Options: --gpus --data-root --metadata-path --freeze-vae --freeze-vit --freeze-llm --freeze-und"
      echo "         --text-dropout --vae-dropout --vit-dropout --total-steps --save-every --lr"
      echo "         --global-seed --data-seed --run-tag"
      exit 0
      ;;
    *) echo "Unknown argument: $1 (use --help)" >&2; exit 2 ;;
  esac
done

# If the data root is overridden but metadata is not, use its sibling judge
# directory automatically. Explicit --metadata-path always wins.
if [[ -z "${METADATA_PATH}" ]]; then
  METADATA_PATH="${DATA_ROOT}_judge/train.jsonl"
fi

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export OMP_NUM_THREADS=1
export BAGEL_REASON_HEATMAP_DATA_DIR=${DATA_ROOT}
export BAGEL_REASON_HEATMAP_METADATA_PATH=${METADATA_PATH}
export BAGEL_PERSPECTIVE_MULTITASK_REASON=1
export BAGEL_PERSPECTIVE_JUDGMENT=1

GPU_COUNT=$(awk -F, '{print NF}' <<< "${GPU_LIST}")
VAE_TAG=$([[ "${FREEZE_VAE}" == "True" ]] && echo vaefreeze || echo vaeopen)
VIT_TAG=$([[ "${FREEZE_VIT}" == "True" ]] && echo vitfreeze || echo vitopen)
# Always make each launch a distinct W&B run, even when all hyperparameters
# are unchanged.  The timestamp is also part of RESULTS_DIR so checkpoints
# from separate launches cannot be mixed accidentally.
RUN_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
# Keep the W&B name independent of long dataset paths.
RUN_NAME="perspective-${RUN_TAG}-${VIT_TAG}-${VAE_TAG}-${GPU_COUNT}gpu-${TOTAL_STEPS}step-${RUN_TIMESTAMP}"
# Keep a generous margin below W&B's 128-character Name limit.  All source
# components are normalized to ASCII, so character count is byte-safe.
# Keep a conservative limit well below W&B's 128-character limit.
# The current components fit in this limit while retaining the timestamp.
RUN_NAME=$(printf '%s' "${RUN_NAME}" | cut -c1-64)
# pretrain_unified_navit.py builds the final W&B id as
#   f"{wandb_name}-run{wandb_runid}"
# Keep runid short; using the timestamp avoids doubling the full name past
# W&B's 128-character limit.
RUN_ID="${RUN_TIMESTAMP}"
echo "W&B name (${#RUN_NAME} chars): ${RUN_NAME}"
echo "W&B run id (${#RUN_ID} chars): ${RUN_ID}"
RESULTS_DIR=/data/bagel/repo/Bagel/results/${RUN_NAME}
mkdir -p "${RESULTS_DIR}/checkpoints"

torchrun \
  --nproc_per_node=4 \
  --master_port=29542 \
  train/finetune_reason_heatmap_multitask.py \
  --dataset_config_file ./data/configs/perspective_single_pair_refine.yaml \
  --model_path /data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --resume_from /data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT \
  --resume_model_only True \
  --finetune_from_ema True \
  --auto_resume False \
  --sequential_checkpoint_load True \
  --model_init_dtype bfloat16 \
  --visual_gen True \
  --visual_und True \
  --score_head True \
  --score_weight 1.0 \
  --split_gen_adapter_by_task True \
  --gen_task_filter joint \
  --freeze_vae "${FREEZE_VAE}" \
  --freeze_vit "${FREEZE_VIT}" \
  --freeze_llm "${FREEZE_LLM}" \
  --freeze_und "${FREEZE_UND}" \
  --text_cond_dropout_prob "${TEXT_DROPOUT}" \
  --vae_cond_dropout_prob "${VAE_DROPOUT}" \
  --vit_cond_dropout_prob "${VIT_DROPOUT}" \
  --timestep_shift 4.0 \
  --ce_weight 0.25 \
  --judgment_ce_weight 1.0 \
  --mse_weight 10 \
  --repair_mse_weight 1 \
  --heatmap_mse_weight 10 \
  --foreground_balanced_heatmap_mse False \
  --use_flex True \
  --num_shard 4 \
  --num_replicate 1 \
  --sharding_strategy HYBRID_SHARD \
  --expected_num_tokens 24576 \
  --max_num_tokens 27648 \
  --max_num_tokens_per_sample 16384 \
  --gradient_accumulation_steps 1 \
  --num_workers 2 \
  --prefetch_factor 4 \
  --global_seed "${GLOBAL_SEED}" \
  --data_seed "${DATA_SEED}" \
  --lr "${LR}" \
  --lr_scheduler constant \
  --warmup_steps 500 \
  --total_steps "${TOTAL_STEPS}" \
  --save_every "${SAVE_EVERY}" \
  --log_every 1 \
  --wandb_offline False \
  --wandb_project bagel \
  --wandb_name "${RUN_NAME}" \
  --wandb_runid "${RUN_ID}" \
  --wandb_resume allow \
  --checkpoint_dir "${RESULTS_DIR}/checkpoints" \
  --results_dir "${RESULTS_DIR}"

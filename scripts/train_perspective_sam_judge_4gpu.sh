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
METADATA_PATH="${DATA_ROOT}_judge/train.jsonl"
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

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export OMP_NUM_THREADS=1
export BAGEL_REASON_HEATMAP_DATA_DIR=${DATA_ROOT}
export BAGEL_REASON_HEATMAP_METADATA_PATH=${METADATA_PATH}
export BAGEL_PERSPECTIVE_MULTITASK_REASON=1
export BAGEL_PERSPECTIVE_JUDGMENT=1

GPU_COUNT=$(awk -F, '{print NF}' <<< "${GPU_LIST}")
DATA_TAG=$(basename "${DATA_ROOT}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')
VAE_TAG=$([[ "${FREEZE_VAE}" == "True" ]] && echo vaefreeze || echo vaeopen)
VIT_TAG=$([[ "${FREEZE_VIT}" == "True" ]] && echo vitfreeze || echo vitopen)
RUN_NAME="perspective-${DATA_TAG}-${RUN_TAG}-${VIT_TAG}-${VAE_TAG}-${GPU_COUNT}gpu-${TOTAL_STEPS}step"
RUN_NAME=${RUN_NAME:0:120}
RUN_ID=$(echo "${RUN_NAME}" | tr '_' '-')
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

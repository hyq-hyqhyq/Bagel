#!/usr/bin/env bash
set -euo pipefail

cd /data/bagel/repo/Bagel
source /data/bagel/conda/etc/profile.d/conda.sh
conda activate bagel

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export BAGEL_REASON_HEATMAP_DATA_DIR=/data/bagel/data/perspective_combined_train1400_sam_postprocessed_20260920
export BAGEL_REASON_HEATMAP_METADATA_PATH=/data/bagel/data/perspective_combined_train1400_sam_postprocessed_20260920_judge/train.jsonl
export BAGEL_PERSPECTIVE_MULTITASK_REASON=1
export BAGEL_PERSPECTIVE_JUDGMENT=1

RUN_NAME=perspective_combined1400_sam_judge_vitopen_vaeopen_4gpu_30k_v1
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
  --freeze_vae False \
  --freeze_vit False \
  --freeze_llm False \
  --freeze_und False \
  --text_cond_dropout_prob 0.0 \
  --vae_cond_dropout_prob 0.0 \
  --vit_cond_dropout_prob 0.0 \
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
  --global_seed 4396 \
  --data_seed 42 \
  --lr 2e-5 \
  --lr_scheduler constant \
  --warmup_steps 500 \
  --total_steps 30000 \
  --save_every 2000 \
  --log_every 1 \
  --wandb_offline False \
  --wandb_project bagel \
  --wandb_name "${RUN_NAME}" \
  --wandb_runid perspective-combined1400-sam-judge-vitopen-vaeopen-4gpu-30k-v1 \
  --wandb_resume allow \
  --checkpoint_dir "${RESULTS_DIR}/checkpoints" \
  --results_dir "${RESULTS_DIR}"

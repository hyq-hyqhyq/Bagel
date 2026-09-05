#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATA_PATH=${DATA_PATH:-/data/bagel/data/perspective_ultra_pass_train3000_test20_20260828}
MODEL_PATH=${MODEL_PATH:-/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT}
JOINT_CHECKPOINT=${JOINT_CHECKPOINT:?Set JOINT_CHECKPOINT to the trained joint repair/heatmap checkpoint}
RUN_NAME=${RUN_NAME:-perspective_e2e_k4_30k_4gpu}
RESULTS_DIR=${RESULTS_DIR:-/data/bagel/repo/Bagel/results/$RUN_NAME}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$RESULTS_DIR/checkpoints}
DATASET_CONFIG=${DATASET_CONFIG:-./data/configs/perspective_e2e.yaml}
BACKWARD_PREFETCH=${BACKWARD_PREFETCH:-BACKWARD_PRE}
FSDP_FINE_GRAINED_MOT=${FSDP_FINE_GRAINED_MOT:-False}
VAE_DECODER_CHECKPOINT=${VAE_DECODER_CHECKPOINT:-False}
GRADIENT_DENOISE_STEPS=${GRADIENT_DENOISE_STEPS:-4}

test -s "$DATA_PATH/train.jsonl" || {
  echo "Missing training metadata: $DATA_PATH/train.jsonl" >&2
  exit 1
}
test -d "$JOINT_CHECKPOINT" || {
  echo "Missing joint checkpoint: $JOINT_CHECKPOINT" >&2
  exit 1
}

export BAGEL_REASON_HEATMAP_DATA_DIR="$DATA_PATH"
export BAGEL_REASON_HEATMAP_METADATA_PATH="$DATA_PATH/train.jsonl"
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} nohup torchrun \
  --nnodes=1 --node_rank=0 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=${MASTER_PORT:-29507} \
  train/finetune_reason_heatmap_e2e.py \
  --dataset_config_file "$DATASET_CONFIG" \
  --model_path "$MODEL_PATH" \
  --finetune_from_hf True --resume_from "$JOINT_CHECKPOINT" \
  --resume_model_only True --finetune_from_ema True \
  --auto_resume False --sequential_checkpoint_load True \
  --model_init_dtype bfloat16 --visual_gen True --visual_und True \
  --score_head True --split_gen_adapter_by_task True --gen_task_filter joint \
  --freeze_vae True --freeze_llm False --freeze_vit False --freeze_und False \
  --ema_enabled False \
  --num_timesteps 50 --gradient_denoise_steps "$GRADIENT_DENOISE_STEPS" \
  --max_reason_tokens 1000 \
  --vae_decoder_checkpoint "$VAE_DECODER_CHECKPOINT" \
  --cfg_text_scale 4.0 --cfg_img_scale 1.0 --timestep_shift 4.0 \
  --heatmap_flow_weight 10.0 --repair_flow_weight 1.0 \
  --heatmap_score_weight 1.0 --repair_score_weight 0.2 \
  --heatmap_reason_weight 0.25 --repair_reason_weight 0.05 \
  --lr 2e-5 --lr_scheduler constant --warmup_steps 500 --total_steps 30000 \
  --save_every 2000 --log_every 1 --gradient_accumulation_steps 1 \
  --num_shard 4 --num_replicate 1 --sharding_strategy HYBRID_SHARD \
  --backward_prefetch "$BACKWARD_PREFETCH" \
  --fsdp_fine_grained_mot "$FSDP_FINE_GRAINED_MOT" \
  --max_latent_size 64 --num_workers 1 --prefetch_factor 2 \
  --wandb_offline False --wandb_project bagel --wandb_name "$RUN_NAME" \
  --wandb_runid "${WANDB_RUNID:-perspective-e2e-k4-v1}" --wandb_resume allow \
  --checkpoint_dir "$CHECKPOINT_DIR" --results_dir "$RESULTS_DIR" \
  > "$RESULTS_DIR/train.log" 2>&1 &

echo "launcher PID: $!"
echo "log: $RESULTS_DIR/train.log"
tail -f "$RESULTS_DIR/train.log"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

DATA_PATH=${DATA_PATH:-/data/bagel/data/perspective_ultra_pass_train3000_test20_20260828}
MODEL_PATH=${MODEL_PATH:-/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT}
RUN_NAME=${RUN_NAME:-perspective_single_pair_refine_2_1_1_30k_4gpu}
RESULTS_DIR=${RESULTS_DIR:-/data/bagel/repo/Bagel/results/$RUN_NAME}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$RESULTS_DIR/checkpoints}
DATASET_CONFIG=${DATASET_CONFIG:-./data/configs/perspective_single_pair_refine.yaml}
WANDB_NAME=${WANDB_NAME:-$RUN_NAME}
WANDB_RUNID=${WANDB_RUNID:-perspective-single-pair-refine-2-1-1-v1}

test -s "$DATA_PATH/train.jsonl" || {
  echo "Missing training metadata: $DATA_PATH/train.jsonl" >&2
  exit 1
}

export BAGEL_REASON_HEATMAP_DATA_DIR="$DATA_PATH"
export BAGEL_REASON_HEATMAP_METADATA_PATH="$DATA_PATH/train.jsonl"
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} nohup torchrun \
  --nnodes=1 --node_rank=0 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=${MASTER_PORT:-29506} \
  train/finetune_reason_heatmap_multitask.py \
  --dataset_config_file "$DATASET_CONFIG" \
  --model_path "$MODEL_PATH" \
  --finetune_from_hf True --resume_from "$MODEL_PATH" \
  --resume_model_only True --finetune_from_ema True \
  --auto_resume False --sequential_checkpoint_load True \
  --model_init_dtype bfloat16 --visual_gen True --visual_und True \
  --score_head True --score_weight 1.0 \
  --split_gen_adapter_by_task True --gen_task_filter joint \
  --freeze_vae True --freeze_llm False --freeze_vit False --freeze_und False \
  --ce_weight 0.25 --mse_weight 10 \
  --text_cond_dropout_prob 0.05 --vae_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 --timestep_shift 4.0 \
  --lr 2e-5 --lr_scheduler constant --warmup_steps 500 --total_steps 30000 \
  --save_every 2000 --log_every 1 --gradient_accumulation_steps 1 \
  --num_shard 4 --num_replicate 1 --sharding_strategy HYBRID_SHARD \
  --max_latent_size 64 --expected_num_tokens 24576 \
  --max_num_tokens 27648 --max_num_tokens_per_sample 16384 \
  --num_workers 1 --prefetch_factor 2 --use_flex True \
  --wandb_offline False --wandb_project bagel --wandb_name "$WANDB_NAME" \
  --wandb_runid "$WANDB_RUNID" --wandb_resume allow \
  --checkpoint_dir "$CHECKPOINT_DIR" --results_dir "$RESULTS_DIR" \
  > "$RESULTS_DIR/train.log" 2>&1 &

echo "launcher PID: $!"
echo "log: $RESULTS_DIR/train.log"
tail -f "$RESULTS_DIR/train.log"

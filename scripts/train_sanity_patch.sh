#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

num_nodes=${num_nodes:-1}
node_rank=${node_rank:-0}
nproc_per_node=${nproc_per_node:-4}
master_addr=${master_addr:-127.0.0.1}
master_port=${master_port:-29503}
model_path=${model_path:-/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT}
data_path=${BAGEL_SANITY_PATCH_DATA_DIR:-/data/bagel/repo/Bagel/sanity_patch_data}
run_name=${run_name:-sanity_patch_two_stage_score_split_gen_mse4_full_30k_a800_4gpu}
output_path=${output_path:-/data/bagel/repo/Bagel/results/$run_name}
ckpt_path=${ckpt_path:-$output_path/checkpoints}
wandb_name=${wandb_name:-$run_name}
wandb_runid=${wandb_runid:-sanity-patch-two-stage-score-split-gen-mse4-a800-v1}
wandb_offline=${wandb_offline:-False}
total_steps=${total_steps:-30000}
dataset_config_file=${dataset_config_file:-./data/configs/sanity_patch.yaml}

test -s "$data_path/metadata/train.jsonl" || {
  echo "Missing training metadata: $data_path/metadata/train.jsonl" >&2
  exit 1
}

python sanity_patch/prepare_training_metadata.py \
  --data-root "$data_path" \
  --train-size 3960 \
  --test-size 40

export BAGEL_SANITY_PATCH_DATA_DIR="$data_path"
export BAGEL_SANITY_PATCH_METADATA_PATH="$data_path/metadata/train_3960.jsonl"
export MODEL_PATH="$model_path"
export RUN_NAME="$run_name"
export RESULTS_DIR="$output_path"
export CHECKPOINT_DIR="$ckpt_path"
mkdir -p "$output_path" "$ckpt_path"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} nohup torchrun \
  --nnodes="$num_nodes" \
  --node_rank="$node_rank" \
  --nproc_per_node="$nproc_per_node" \
  --master_addr="$master_addr" \
  --master_port="$master_port" \
  train/finetune_reason_heatmap.py \
  --dataset_config_file "$dataset_config_file" \
  --model_path "$model_path" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --resume_from "$model_path" \
  --resume_model_only True \
  --finetune_from_ema True \
  --auto_resume False \
  --sequential_checkpoint_load True \
  --model_init_dtype bfloat16 \
  --visual_gen True \
  --visual_und True \
  --score_head True \
  --split_gen_adapter_by_task True \
  --gen_task_filter joint \
  --freeze_vae True \
  --freeze_vit False \
  --freeze_llm False \
  --freeze_und False \
  --text_cond_dropout_prob 0.05 \
  --vae_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 \
  --timestep_shift 4.0 \
  --ce_weight 0.25 \
  --mse_weight 10 \
  --repair_mse_weight 1 \
  --heatmap_mse_weight 10 \
  --score_weight 1.0 \
  --use_flex True \
  --num_shard 4 \
  --num_replicate 1 \
  --sharding_strategy HYBRID_SHARD \
  --expected_num_tokens 24576 \
  --max_num_tokens 27648 \
  --wandb_project bagel \
  --wandb_resume allow \
  --results_dir "$output_path" \
  --checkpoint_dir "$ckpt_path" \
  --wandb_name "$wandb_name" \
  --wandb_runid "$wandb_runid" \
  --wandb_offline "$wandb_offline" \
  --num_workers 1 \
  --log_every 1 \
  --save_every 2000 \
  --total_steps "$total_steps" \
  --warmup_steps 500 \
  --lr 2e-5 \
  --max_num_tokens_per_sample 11520 \
  > "$output_path/train.log" 2>&1 &

echo "launcher PID: $!"
echo "log: $output_path/train.log"
tail -f "$output_path/train.log"

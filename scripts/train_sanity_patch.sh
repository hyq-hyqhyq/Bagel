#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

num_nodes=${num_nodes:-1}
node_rank=${node_rank:-0}
nproc_per_node=${nproc_per_node:-8}
master_addr=${master_addr:-127.0.0.1}
master_port=${master_port:-29500}
model_path=${model_path:-/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT}
data_path=${BAGEL_SANITY_PATCH_DATA_DIR:-/data/bagel/repo/Bagel/sanity_patch_data}
output_path=${output_path:-results/sanity_patch_full_8gpu_score}
ckpt_path=${ckpt_path:-$output_path/checkpoints}
wandb_name=${wandb_name:-sanity_patch_full_8gpu_score}
wandb_runid=${wandb_runid:-0}
wandb_offline=${wandb_offline:-True}
total_steps=${total_steps:-10000}
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
mkdir -p "$output_path" "$ckpt_path"

torchrun \
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
  --auto_resume True \
  --visual_gen True \
  --visual_und True \
  --score_head True \
  --score_weight 1.0 \
  --freeze_vae True \
  --freeze_vit False \
  --freeze_llm False \
  --freeze_und False \
  --text_cond_dropout_prob 0.05 \
  --vae_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 \
  --timestep_shift 4.0 \
  --ce_weight 1.0 \
  --mse_weight 1.0 \
  --use_flex True \
  --results_dir "$output_path" \
  --checkpoint_dir "$ckpt_path" \
  --wandb_name "$wandb_name" \
  --wandb_runid "$wandb_runid" \
  --wandb_offline "$wandb_offline" \
  --num_workers 1 \
  --log_every 1 \
  --save_every 1000 \
  --total_steps "$total_steps" \
  --warmup_steps 500 \
  --lr 2e-5 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 11520

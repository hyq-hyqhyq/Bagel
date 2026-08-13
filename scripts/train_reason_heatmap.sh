#!/usr/bin/env bash

# Replace the variables with your own.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

num_nodes=${num_nodes:-1}
node_rank=${node_rank:-0}
nproc_per_node=${nproc_per_node:-8}
master_addr=${master_addr:-127.0.0.1}
master_port=${master_port:-29500}
model_path=${model_path:-models/BAGEL-7B-MoT}
data_path=${BAGEL_REASON_HEATMAP_DATA_DIR:-/data/bagel/data/perspective_5k/canonical_5k_clean_4402}
output_path=${output_path:-results/reason_heatmap}
ckpt_path=${ckpt_path:-$output_path/checkpoints}

export BAGEL_REASON_HEATMAP_DATA_DIR="$data_path"

torchrun \
  --nnodes="$num_nodes" \
  --node_rank="$node_rank" \
  --nproc_per_node="$nproc_per_node" \
  --master_addr="$master_addr" \
  --master_port="$master_port" \
  train/finetune_reason_heatmap.py \
  --dataset_config_file ./data/configs/reason_heatmap.yaml \
  --model_path "$model_path" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --resume_from "$model_path" \
  --resume_model_only True \
  --finetune_from_ema True \
  --visual_gen True \
  --visual_und True \
  --freeze_vae True \
  --freeze_vit False \
  --freeze_llm False \
  --freeze_und False \
  --use_flex True \
  --results_dir "$output_path" \
  --checkpoint_dir "$ckpt_path" \
  --num_workers 1 \
  --log_every 1 \
  --save_every 2000 \
  --lr 2e-5 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240

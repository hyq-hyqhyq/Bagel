#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

num_nodes=${num_nodes:-1}
node_rank=${node_rank:-0}
nproc_per_node=${nproc_per_node:-4}
master_addr=${master_addr:-127.0.0.1}
master_port=${master_port:-29503}
model_path=${model_path:-/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT}
data_path=${data_path:-/data/bagel/repo/Bagel/sanity_patch_data}
metadata_path=${metadata_path:-$data_path/metadata/train.jsonl}
dataset_config_file=${dataset_config_file:-./data/configs/sanity_patch.yaml}
run_name=${run_name:-multitask_reason_heatmap_30k_4gpu}
output_path=${output_path:-/data/bagel/repo/Bagel/results/$run_name}
ckpt_path=${ckpt_path:-$output_path/checkpoints}
wandb_name=${wandb_name:-$run_name}
wandb_runid=${wandb_runid:-multitask-reason-heatmap-v1}
wandb_offline=${wandb_offline:-False}
total_steps=${total_steps:-30000}

test -s "$metadata_path" || {
  echo "Missing training metadata: $metadata_path" >&2
  exit 1
}

export BAGEL_REASON_HEATMAP_DATA_DIR="$data_path"
export BAGEL_REASON_HEATMAP_METADATA_PATH="$metadata_path"
export BAGEL_SANITY_PATCH_DATA_DIR="$data_path"
export BAGEL_SANITY_PATCH_METADATA_PATH="$metadata_path"
mkdir -p "$output_path" "$ckpt_path"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} nohup torchrun \
  --nnodes="$num_nodes" --node_rank="$node_rank" \
  --nproc_per_node="$nproc_per_node" --master_addr="$master_addr" \
  --master_port="$master_port" train/finetune_reason_heatmap_multitask.py \
  --dataset_config_file "$dataset_config_file" \
  --model_path "$model_path" --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 --finetune_from_hf True \
  --resume_from "$model_path" --resume_model_only True \
  --finetune_from_ema True --auto_resume False \
  --sequential_checkpoint_load True --model_init_dtype bfloat16 \
  --visual_gen True --visual_und True --score_head True \
  --split_gen_adapter_by_task True --gen_task_filter joint \
  --freeze_vae True --freeze_vit False --freeze_llm False --freeze_und False \
  --text_cond_dropout_prob 0.05 --vae_cond_dropout_prob 0.1 \
  --vit_cond_dropout_prob 0.1 --timestep_shift 4.0 \
  --ce_weight 0.25 --mse_weight 10 --repair_mse_weight 1 \
  --heatmap_mse_weight 10 --score_weight 1.0 --use_flex True \
  --num_shard 4 --num_replicate 1 --sharding_strategy HYBRID_SHARD \
  --expected_num_tokens 24576 --max_num_tokens 27648 \
  --max_num_tokens_per_sample 11520 --num_workers 1 --prefetch_factor 2 \
  --wandb_offline "$wandb_offline" --wandb_project bagel \
  --wandb_name "$wandb_name" --wandb_runid "$wandb_runid" \
  --wandb_resume allow --checkpoint_dir "$ckpt_path" \
  --results_dir "$output_path" --log_every 1 --save_every 2000 \
  --total_steps "$total_steps" --warmup_steps 500 --lr 2e-5 \
  > "$output_path/train.log" 2>&1 &

echo "launcher PID: $!"
echo "log: $output_path/train.log"
tail -f "$output_path/train.log"

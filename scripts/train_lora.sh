export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

model_path=${model_path:-/data/bagel/repo/agent/bpipe/models/BAGEL-7B-MoT}
data_path=${BAGEL_REASON_HEATMAP_DATA_DIR:-/data/bagel/data/perspective_5k/canonical_5k_clean_4402}
output_path=${output_path:-./results/reason_heatmap_lora}
ckpt_path=${ckpt_path:-$output_path/checkpoints}
nproc_per_node=${nproc_per_node:-8}
master_port=${master_port:-29500}
total_steps=${total_steps:-20000}
save_every=${save_every:-200}
stop_after_step=${stop_after_step:-0}
mse_weight=${mse_weight:-50}
ce_weight=${ce_weight:-0.1}
wandb_name=${wandb_name:-reason_heatmap_lora_$(date +%Y%m%d_%H%M%S)}
wandb_runid=${wandb_runid:-0}
wandb_offline=${wandb_offline:-False}

export BAGEL_REASON_HEATMAP_DATA_DIR="$data_path"

torchrun \
  --nnodes=1 \
  --nproc_per_node=$nproc_per_node \
  --master_port="$master_port" \
  train/pretrain_unified_navit_lora.py \
  --dataset_config_file ./data/configs/reason_heatmap.yaml \
  --warmup_steps 200 \
  --total_steps "$total_steps" \
  --model_path $model_path \
  --wandb_offline "$wandb_offline" \
  --wandb_name "$wandb_name" \
  --wandb_runid "$wandb_runid" \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume_from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune_from_ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_workers 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 11520 \
  --num_shard $nproc_per_node \
  --cpu_offload False \
  --text_cond_dropout_prob 0. \
  --vae_cond_dropout_prob 0. \
  --vit_cond_dropout_prob 0. \
  --mse_weight "$mse_weight" \
  --ce_weight "$ce_weight" \
  --save_every "$save_every" \
  --stop_after_step "$stop_after_step" \
  --lora_rank 256 \
  --lora_alpha 512 \
  --results_dir "$output_path" \
  --checkpoint_dir "$ckpt_path"

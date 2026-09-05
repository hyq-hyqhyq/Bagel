# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Truncated end-to-end perspective repair -> heatmap fine-tuning."""

from dataclasses import dataclass, field

from data import dataset_base
from data.reason_heatmap_dataset_info import DATASET_INFO, DATASET_REGISTRY

dataset_base.DATASET_INFO = DATASET_INFO
dataset_base.DATASET_REGISTRY = DATASET_REGISTRY

from train.pretrain_unified_navit import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    main,
)


@dataclass
class E2EModelArguments(ModelArguments):
    text_cond_dropout_prob: float = field(default=0.0)
    vae_cond_dropout_prob: float = field(default=0.0)
    vit_cond_dropout_prob: float = field(default=0.0)


@dataclass
class E2ETrainingArguments(TrainingArguments):
    e2e_enabled: bool = field(default=True)
    ema_enabled: bool = field(default=False)
    split_gen_adapter_by_task: bool = field(default=True)
    gen_task_filter: str = field(default="joint")

    num_timesteps: int = field(default=50)
    gradient_denoise_steps: int = field(default=4)
    max_reason_tokens: int = field(default=1000)
    vae_decoder_checkpoint: bool = field(default=False)
    e2e_empty_cache: bool = field(default=False)
    e2e_cpu_gradient_staging: bool = field(default=False)
    e2e_activation_cpu_offload: bool = field(default=False)
    cfg_text_scale: float = field(default=4.0)
    cfg_img_scale: float = field(default=1.0)
    timestep_shift: float = field(default=4.0)

    heatmap_flow_weight: float = field(default=10.0)
    repair_flow_weight: float = field(default=1.0)
    heatmap_score_weight: float = field(default=1.0)
    repair_score_weight: float = field(default=0.2)
    heatmap_reason_weight: float = field(default=0.25)
    repair_reason_weight: float = field(default=0.05)


if __name__ == "__main__":
    main(
        model_arguments=E2EModelArguments,
        data_arguments=DataArguments,
        training_arguments=E2ETrainingArguments,
    )

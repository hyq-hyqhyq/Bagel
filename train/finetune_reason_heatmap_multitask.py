# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Multitask reason/heatmap training entrypoint.

The single-task entrypoint remains ``finetune_reason_heatmap.py`` and uses the
legacy ``vae2llm``/``llm2vae`` generation head.  This entrypoint explicitly
enables the task-specific repair and heatmap adapters.
"""

from dataclasses import dataclass, field

from data import dataset_base
from data.reason_heatmap_dataset_info import DATASET_INFO, DATASET_REGISTRY
from sanity_patch.settings import (
    TEXT_COND_DROPOUT_PROB,
    TIMESTEP_SHIFT,
    VAE_COND_DROPOUT_PROB,
    VIT_COND_DROPOUT_PROB,
)

dataset_base.DATASET_INFO = DATASET_INFO
dataset_base.DATASET_REGISTRY = DATASET_REGISTRY

from train.pretrain_unified_navit import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    main,
)


@dataclass
class MultitaskModelArguments(ModelArguments):
    text_cond_dropout_prob: float = field(default=TEXT_COND_DROPOUT_PROB)
    vae_cond_dropout_prob: float = field(default=VAE_COND_DROPOUT_PROB)
    vit_cond_dropout_prob: float = field(default=VIT_COND_DROPOUT_PROB)


@dataclass
class MultitaskTrainingArguments(TrainingArguments):
    split_gen_adapter_by_task: bool = field(default=True)
    gen_task_filter: str = field(default="joint")
    timestep_shift: float = field(default=TIMESTEP_SHIFT)


if __name__ == "__main__":
    main(
        model_arguments=MultitaskModelArguments,
        data_arguments=DataArguments,
        training_arguments=MultitaskTrainingArguments,
    )

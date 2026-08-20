# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

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
class ReasonHeatmapModelArguments(ModelArguments):
    text_cond_dropout_prob: float = field(
        default=TEXT_COND_DROPOUT_PROB,
        metadata={"help": "Probability of dropping text embeddings during training."},
    )
    vae_cond_dropout_prob: float = field(
        default=VAE_COND_DROPOUT_PROB,
        metadata={"help": "Probability of dropping VAE latent inputs during training."},
    )
    vit_cond_dropout_prob: float = field(
        default=VIT_COND_DROPOUT_PROB,
        metadata={"help": "Probability of dropping ViT visual features during training."},
    )


@dataclass
class ReasonHeatmapTrainingArguments(TrainingArguments):
    timestep_shift: float = field(
        default=TIMESTEP_SHIFT,
        metadata={
            "help": "Shift applied to diffusion timestep indices (for latent prediction)."
        },
    )


if __name__ == "__main__":
    main(
        model_arguments=ReasonHeatmapModelArguments,
        data_arguments=DataArguments,
        training_arguments=ReasonHeatmapTrainingArguments,
    )

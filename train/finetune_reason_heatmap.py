# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from data import dataset_base
from data.reason_heatmap_dataset_info import DATASET_INFO, DATASET_REGISTRY


dataset_base.DATASET_INFO = DATASET_INFO
dataset_base.DATASET_REGISTRY = DATASET_REGISTRY

from train.pretrain_unified_navit import main


if __name__ == "__main__":
    main()

# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os

from .interleave_datasets.reason_heatmap_dataset import ReasonHeatmapIterableDataset
from .interleave_datasets.sanity_patch_dataset import SanityPatchIterableDataset


def dataset_entry(env_name):
    data_dir = os.environ.get(env_name)
    return {
        "data_dir": data_dir,
        "jsonl_path": os.path.join(data_dir, "metadata/train.jsonl")
        if data_dir
        else None,
    }


DATASET_REGISTRY = {
    "reason_heatmap": ReasonHeatmapIterableDataset,
    "sanity_patch": SanityPatchIterableDataset,
}


DATASET_INFO = {
    "reason_heatmap": {
        "perspective": dataset_entry("BAGEL_REASON_HEATMAP_DATA_DIR"),
    },
    "sanity_patch": {
        "patch": dataset_entry("BAGEL_SANITY_PATCH_DATA_DIR"),
    },
}

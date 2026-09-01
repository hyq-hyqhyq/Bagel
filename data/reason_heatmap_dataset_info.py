# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os

from .interleave_datasets.reason_heatmap_dataset import ReasonHeatmapIterableDataset
from .interleave_datasets.perspective_single_pair_refine_dataset import (
    PerspectiveSinglePairRefineIterableDataset,
)
from .interleave_datasets.perspective_e2e_dataset import (
    PerspectiveE2EIterableDataset,
)
from .interleave_datasets.sanity_patch_dataset import SanityPatchIterableDataset


def dataset_entry(env_name, metadata_env_name=None):
    data_dir = os.environ.get(env_name)
    metadata_path = os.environ.get(metadata_env_name) if metadata_env_name else None
    return {
        "data_dir": data_dir,
        "jsonl_path": metadata_path
        or (os.path.join(data_dir, "metadata/train.jsonl") if data_dir else None),
    }


DATASET_REGISTRY = {
    "reason_heatmap": ReasonHeatmapIterableDataset,
    "perspective_single_pair_refine": (
        PerspectiveSinglePairRefineIterableDataset
    ),
    "perspective_e2e": PerspectiveE2EIterableDataset,
    "sanity_patch": SanityPatchIterableDataset,
}


DATASET_INFO = {
    "reason_heatmap": {
        "perspective": dataset_entry(
            "BAGEL_REASON_HEATMAP_DATA_DIR",
            "BAGEL_REASON_HEATMAP_METADATA_PATH",
        ),
    },
    "perspective_single_pair_refine": {
        "perspective": dataset_entry(
            "BAGEL_REASON_HEATMAP_DATA_DIR",
            "BAGEL_REASON_HEATMAP_METADATA_PATH",
        ),
    },
    "perspective_e2e": {
        "perspective": dataset_entry(
            "BAGEL_REASON_HEATMAP_DATA_DIR",
            "BAGEL_REASON_HEATMAP_METADATA_PATH",
        ),
    },
    "sanity_patch": {
        "patch": dataset_entry(
            "BAGEL_SANITY_PATCH_DATA_DIR",
            "BAGEL_SANITY_PATCH_METADATA_PATH",
        ),
    },
}

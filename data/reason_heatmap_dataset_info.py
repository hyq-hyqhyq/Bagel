# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os

from .interleave_datasets.reason_heatmap_dataset import ReasonHeatmapIterableDataset


DATASET_REGISTRY = {
    "reason_heatmap": ReasonHeatmapIterableDataset,
}


DATASET_INFO = {
    "reason_heatmap": {
        "perspective": {
            "data_dir": os.environ.get(
                "BAGEL_REASON_HEATMAP_DATA_DIR",
            ),
            "jsonl_path": os.path.join(
                os.environ.get("BAGEL_REASON_HEATMAP_DATA_DIR"),
                "metadata/train.jsonl",
            ),
        },
    },
}

# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Isolated perspective single/pair/refine multitask training dataset."""

import os

from PIL import Image

from .reason_heatmap_dataset import ReasonHeatmapIterableDataset
from ..reason_heatmap_prompts import (
    PAIR_HEATMAP_PROMPT,
    REFINE_PROMPT,
    SINGLE_HEATMAP_PROMPT,
)


class PerspectiveSinglePairRefineIterableDataset(ReasonHeatmapIterableDataset):
    """Emit an exact 2:1:1 single-heatmap/pair-heatmap/refine task mix.

    Each source row expands to eight samples. Single-good and single-bad are
    emitted twice each, while pair-good, pair-bad, good-refine, and bad-refine
    are emitted once each. This gives a 25/25/12.5/12.5/12.5/12.5 percent
    sample mix without changing the legacy reason-heatmap dataset behavior.
    """

    def __init__(self, *args, **kwargs):
        if "heatmap_only" in kwargs:
            raise ValueError(
                "PerspectiveSinglePairRefineIterableDataset does not support "
                "heatmap_only; its task recipe is fixed."
            )
        super().__init__(*args, heatmap_only=False, **kwargs)

    def parse_row(self, row, data_dir):
        good_image = self._read_image(os.path.join(data_dir, row["good_image"]))
        bad_image = self._read_image(os.path.join(data_dir, row["bad_image"]))
        bad_heatmap = self._read_image(
            os.path.join(data_dir, row["bad_heatmap"])
        )
        black_heatmap = Image.new("RGB", good_image.size)

        task_specs = [
            # Duplicate both single-image variants to give the single task a
            # total weight of two relative to pair and refine.
            (
                "single_good_heatmap",
                "heatmap",
                "good",
                [good_image],
                SINGLE_HEATMAP_PROMPT,
                black_heatmap,
                row.get("good_score", 1.0),
            ),
            (
                "single_bad_heatmap",
                "heatmap",
                "bad",
                [bad_image],
                SINGLE_HEATMAP_PROMPT,
                bad_heatmap,
                row.get("bad_score", 0.0),
            ),
            (
                "single_good_heatmap",
                "heatmap",
                "good",
                [good_image],
                SINGLE_HEATMAP_PROMPT,
                black_heatmap,
                row.get("good_score", 1.0),
            ),
            (
                "single_bad_heatmap",
                "heatmap",
                "bad",
                [bad_image],
                SINGLE_HEATMAP_PROMPT,
                bad_heatmap,
                row.get("bad_score", 0.0),
            ),
            # For pair tasks the first image is always the original and the
            # second image is always its refined reference. The target mask is
            # aligned with the first image.
            (
                "pair_good_heatmap",
                "heatmap",
                "good",
                [good_image, good_image],
                PAIR_HEATMAP_PROMPT,
                black_heatmap,
                row.get("good_score", 1.0),
            ),
            (
                "pair_bad_heatmap",
                "heatmap",
                "bad",
                [bad_image, good_image],
                PAIR_HEATMAP_PROMPT,
                bad_heatmap,
                row.get("bad_score", 0.0),
            ),
            (
                "good_refine",
                "repair",
                "good",
                [good_image],
                REFINE_PROMPT,
                good_image,
                row.get("good_score", 1.0),
            ),
            (
                "bad_refine",
                "repair",
                "bad",
                [bad_image],
                REFINE_PROMPT,
                good_image,
                row.get("bad_score", 0.0),
            ),
        ]

        samples = []
        for (
            task_name,
            gen_task,
            quality,
            input_images,
            prompt,
            target_image,
            score_label,
        ) in task_specs:
            data = self._init_data()
            for input_image in input_images:
                data = self._add_image(
                    data,
                    input_image,
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )
            data = self._add_text(
                data,
                prompt,
                need_loss=False,
                enable_cfg=False,
            )
            data = self._add_image(
                data,
                target_image,
                need_loss=True,
                need_vae=False,
                need_vit=False,
            )
            if score_label is not None:
                data["score_label"] = float(score_label)
            data["gen_task"] = gen_task
            data["gen_quality"] = quality
            data["task_name"] = task_name
            samples.append(data)

        return samples

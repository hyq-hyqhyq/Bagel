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
    """Emit a configurable single-heatmap/pair-heatmap/refine task mix.

    The legacy behavior remains a 2:1:1 mix without reason supervision. New
    experiments can set ``BAGEL_PERSPECTIVE_MULTITASK_RATIO`` (for example,
    ``4:1:1``) and enable supervised reason text with
    ``BAGEL_PERSPECTIVE_MULTITASK_REASON=1``.
    """

    _RATIO_ENV = "BAGEL_PERSPECTIVE_MULTITASK_RATIO"
    _REASON_ENV = "BAGEL_PERSPECTIVE_MULTITASK_REASON"

    def __init__(self, *args, **kwargs):
        if "heatmap_only" in kwargs:
            raise ValueError(
                "PerspectiveSinglePairRefineIterableDataset does not support "
                "heatmap_only; its task recipe is fixed."
            )
        super().__init__(*args, heatmap_only=False, **kwargs)
        ratio = os.environ.get(self._RATIO_ENV, "2:1:1")
        try:
            task_ratio = tuple(int(value) for value in ratio.split(":"))
        except ValueError as exc:
            raise ValueError(
                f"{self._RATIO_ENV} must contain three integers, got {ratio!r}"
            ) from exc
        if len(task_ratio) != 3 or any(value < 0 for value in task_ratio):
            raise ValueError(
                f"{self._RATIO_ENV} must be SINGLE:PAIR:REFINE with three "
                f"non-negative integers, got {ratio!r}"
            )
        if sum(task_ratio) == 0:
            raise ValueError(f"{self._RATIO_ENV} cannot be 0:0:0")
        self.task_ratio = task_ratio

        reason_flag = os.environ.get(self._REASON_ENV, "0").strip().lower()
        if reason_flag not in {"0", "1", "false", "true", "no", "yes"}:
            raise ValueError(
                f"{self._REASON_ENV} must be a boolean, got {reason_flag!r}"
            )
        self.include_reason = reason_flag in {"1", "true", "yes"}
        print(
            f"dataset-{self.dataset_name}: multitask_ratio="
            f"{':'.join(str(value) for value in self.task_ratio)}, "
            f"reason_supervision={self.include_reason}"
        )

    def parse_row(self, row, data_dir):
        good_image = self._read_image(os.path.join(data_dir, row["good_image"]))
        bad_image = self._read_image(os.path.join(data_dir, row["bad_image"]))
        bad_heatmap = self._read_image(
            os.path.join(data_dir, row["bad_heatmap"])
        )
        black_heatmap = Image.new("RGB", good_image.size)

        single_specs = [
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
        ]
        # For pair tasks the first image is always the original and the second
        # image is always its refined reference. The target mask is aligned
        # with the first image.
        pair_specs = [
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
        ]
        refine_specs = [
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
        single_weight, pair_weight, refine_weight = self.task_ratio
        task_specs = (
            single_specs * single_weight
            + pair_specs * pair_weight
            + refine_specs * refine_weight
        )

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
            if self.include_reason:
                reason = row[f"{quality}_reason"]
                data = self._add_text(
                    data,
                    f"<think>{reason}</think>",
                    need_loss=True,
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

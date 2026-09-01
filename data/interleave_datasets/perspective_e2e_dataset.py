# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Row-coupled perspective data for truncated two-stage E2E training."""

import os

from PIL import Image
from torchvision.transforms import functional as tvf
from torchvision.transforms import InterpolationMode

from .reason_heatmap_dataset import ReasonHeatmapIterableDataset
from ..reason_heatmap_prompts import (
    GEN_THINK_SYSTEM_PROMPT,
    PAIR_HEATMAP_PROMPT,
    REFINE_PROMPT,
)


class PerspectiveE2EIterableDataset(ReasonHeatmapIterableDataset):
    """Emit good and bad cascades from the same source row at a 1:1 ratio.

    Unlike the packed multitask dataset, one item keeps the original, repair
    target, heatmap target, and both reasons together.  This is required to
    build ``original -> predicted repair -> heatmap`` inside one autograd graph.
    """

    def __init__(self, *args, **kwargs):
        if "heatmap_only" in kwargs:
            raise ValueError("PerspectiveE2EIterableDataset has a fixed recipe")
        super().__init__(*args, heatmap_only=False, **kwargs)

    @staticmethod
    def _resize_exact(image, height, width):
        return tvf.resize(
            image,
            [height, width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

    def _make_sample(self, row, quality, original, refined, heatmap):
        # The inference path first applies the VAE resize and then feeds that
        # resized image to both VAE and ViT.  Keep the same order here.
        original = self.transform.resize_transform(original)
        width, height = original.size
        refined = self._resize_exact(refined, height, width)
        heatmap = self._resize_exact(heatmap, height, width)

        reason = row[f"{quality}_reason"]
        return {
            "original_vae_image": self.transform(original),
            "original_vit_image": self.vit_transform(original),
            "refined_vae_image": self.transform(refined),
            "heatmap_vae_image": self.transform(heatmap),
            "system_ids": self.tokenizer.encode(GEN_THINK_SYSTEM_PROMPT),
            "repair_prompt_ids": self.tokenizer.encode(REFINE_PROMPT),
            "heatmap_prompt_ids": self.tokenizer.encode(PAIR_HEATMAP_PROMPT),
            "repair_reason_ids": self.tokenizer.encode(
                f"<think>{reason}</think>"
            ),
            "heatmap_reason_ids": self.tokenizer.encode(
                f"<think>{reason}</think>"
            ),
            "score_label": float(row.get(f"{quality}_score", quality == "good")),
            "quality": quality,
            "image_size": (height, width),
        }

    def parse_row(self, row, data_dir):
        good = self._read_image(os.path.join(data_dir, row["good_image"]))
        bad = self._read_image(os.path.join(data_dir, row["bad_image"]))
        bad_heatmap = self._read_image(
            os.path.join(data_dir, row["bad_heatmap"])
        )
        black_heatmap = Image.new("RGB", good.size)
        return [
            self._make_sample(
                row, "good", good, good.copy(), black_heatmap
            ),
            self._make_sample(row, "bad", bad, good, bad_heatmap),
        ]

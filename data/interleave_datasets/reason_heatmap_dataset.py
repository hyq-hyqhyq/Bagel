# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os

from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset
from ..data_utils import pil_img2rgb
from ..reason_heatmap_prompts import (
    PERSPECTIVE_REFINE_PROMPT,
    PERSPECTIVE_VERIFY_PROMPT,
)


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class ReasonHeatmapIterableDataset(InterleavedBaseIterableDataset):

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        vit_transform,
        jsonl_path_list,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        heatmap_only=False,
    ):
        super().__init__(
            dataset_name=dataset_name,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
        )
        self.transform = transform
        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.data_status = data_status
        self.heatmap_only = heatmap_only
        self.data_paths = self.get_data_paths(
            jsonl_path_list, data_dir_list, num_used_data
        )
        self.set_epoch()

    def get_data_paths(self, jsonl_path_list, data_dir_list, num_used_data):
        data_paths = []
        for jsonl_path, data_dir, num_data_point in zip(
            jsonl_path_list, data_dir_list, num_used_data
        ):
            with open(jsonl_path, "r") as f:
                for row_idx, line in enumerate(f):
                    if row_idx >= num_data_point:
                        break
                    data_paths.append((row_idx, data_dir, line))
        return data_paths

    @staticmethod
    def _read_image(image_path):
        return pil_img2rgb(Image.open(image_path))

    def parse_row(self, row, data_dir):
        good_image = self._read_image(os.path.join(data_dir, row["good_image"]))
        bad_image = self._read_image(os.path.join(data_dir, row["bad_image"]))
        bad_heatmap = self._read_image(os.path.join(data_dir, row["bad_heatmap"]))
        black_heatmap = Image.new("RGB", good_image.size)

        if self.heatmap_only:
            task_specs = [
                (
                    [good_image],
                    "Analyze the perspective and projection realism of this image.",
                    row["good_reason"],
                    black_heatmap,
                    row.get("good_score", 1.0),
                ),
                (
                    [bad_image],
                    "Analyze the perspective and projection realism of this image.",
                    row["bad_reason"],
                    bad_heatmap,
                    row.get("bad_score", 0.0),
                ),
                (
                    [good_image, bad_image],
                    "Compare the two images and explain the perspective and projection realism difference.",
                    row["pair_reason"],
                    bad_heatmap,
                    row.get("pair_score"),
                ),
            ]
        else:
            task_specs = [
                (
                    [good_image],
                    PERSPECTIVE_REFINE_PROMPT,
                    row["good_reason"],
                    good_image,
                    row.get("good_score", 1.0),
                ),
                (
                    [bad_image],
                    PERSPECTIVE_REFINE_PROMPT,
                    row["bad_reason"],
                    good_image,
                    row.get("bad_score", 0.0),
                ),
                (
                    [good_image, good_image],
                    PERSPECTIVE_VERIFY_PROMPT,
                    row["good_reason"],
                    black_heatmap,
                    row.get("good_score", 1.0),
                ),
                (
                    [bad_image, good_image],
                    PERSPECTIVE_VERIFY_PROMPT,
                    row["bad_reason"],
                    bad_heatmap,
                    row.get("bad_score", 0.0),
                ),
            ]

        samples = []
        for input_images, prompt, reason, target_image, score_label in task_specs:
            data = self._init_data()
            for input_image in input_images:
                data = self._add_image(
                    data,
                    input_image,
                    need_loss=False,
                    need_vae=True,
                    need_vit=True,
                )
            data = self._add_text(data, prompt, need_loss=False)
            if not self.heatmap_only:
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
            samples.append(data)
        return samples

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id]["data_indexes"] + 1
        else:
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            for row_idx, data_dir, line in data_paths_per_worker[row_start_id:]:
                try:
                    samples = self.parse_row(json.loads(line), data_dir)
                    for data in samples:
                        data['data_indexes'] = {
                            "data_indexes": row_idx,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                        yield data
                except Exception as e:
                    print(f"Error {e} at row#{row_idx}")
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")

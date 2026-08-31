# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import Counter
import sys
import types
from pathlib import Path

import torch
from PIL import Image

# This unit test never opens parquet data, but importing the shared interleave
# base loads the optional pyarrow modules. Stub only those imports so the test
# can run in lightweight developer environments.
if "pyarrow" not in sys.modules:
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    sys.modules["pyarrow"] = pyarrow
    sys.modules["pyarrow.parquet"] = types.ModuleType("pyarrow.parquet")
    sys.modules["pyarrow.fs"] = types.ModuleType("pyarrow.fs")

from data.interleave_datasets.perspective_single_pair_refine_dataset import (
    PerspectiveSinglePairRefineIterableDataset,
)
from data.reason_heatmap_prompts import (
    PAIR_HEATMAP_PROMPT,
    REFINE_PROMPT,
    SINGLE_HEATMAP_PROMPT,
)


class MarkerTransform:
    stride = 1

    def __call__(self, image):
        marker = image.getpixel((0, 0))[0]
        return torch.full((3, 1, 1), marker, dtype=torch.float32)


class RecordingTokenizer:
    def __init__(self):
        self.prompts = []

    def encode(self, text):
        self.prompts.append(text)
        return [len(self.prompts)]


class PerspectiveSinglePairRefineDatasetTest(unittest.TestCase):
    def setUp(self):
        images = {
            "good.png": Image.new("RGB", (2, 2), (10, 0, 0)),
            "bad.png": Image.new("RGB", (2, 2), (20, 0, 0)),
            "bad_heatmap.png": Image.new(
                "RGB", (2, 2), (255, 255, 255)
            ),
        }
        self.row = {
            "good_image": "good.png",
            "bad_image": "bad.png",
            "bad_heatmap": "bad_heatmap.png",
            "good_score": 1.0,
            "bad_score": 0.0,
        }
        self.tokenizer = RecordingTokenizer()
        self.dataset = object.__new__(
            PerspectiveSinglePairRefineIterableDataset
        )
        self.dataset.transform = MarkerTransform()
        self.dataset.vit_transform = MarkerTransform()
        self.dataset.tokenizer = self.tokenizer
        self.dataset._read_image = (
            lambda image_path: images[Path(image_path).name].copy()
        )

    @staticmethod
    def markers(sample):
        return [int(tensor[0, 0, 0].item()) for tensor in sample["image_tensor_list"]]

    def test_prompt_text_is_exact(self):
        self.assertEqual(
            SINGLE_HEATMAP_PROMPT,
            """Analyze the input image for perspective or projection inconsistencies.

Generate a binary localization mask of the erroneous region in the input image.
Use white for perspective/projection errors and black for all other regions.

If no perspective/projection error is present, output an entirely black mask.""",
        )
        self.assertEqual(
            PAIR_HEATMAP_PROMPT,
            """Compare the original image with the refined reference image.

Identify the perspective or projection inconsistency present in the original image and generate a binary localization mask on the original image.
Use white for the erroneous region and black for all other regions.

If the original image contains no perspective/projection error, output an entirely black mask.""",
        )
        self.assertEqual(
            REFINE_PROMPT,
            """Correct any perspective or projection inconsistency in the input image while preserving all unrelated content and appearance.

Preserve object identity, scene semantics, materials, textures, colors, lighting, and overall photorealism.
Keep the original camera viewpoint and framing.

If no correction is necessary, preserve the input image unchanged.""",
        )

    def test_exact_task_and_quality_ratio(self):
        samples = self.dataset.parse_row(self.row, "unused")

        self.assertEqual(len(samples), 8)
        self.assertEqual(
            Counter(sample["task_name"] for sample in samples),
            {
                "single_good_heatmap": 2,
                "single_bad_heatmap": 2,
                "pair_good_heatmap": 1,
                "pair_bad_heatmap": 1,
                "good_refine": 1,
                "bad_refine": 1,
            },
        )
        self.assertEqual(
            Counter(sample["gen_task"] for sample in samples),
            {"heatmap": 6, "repair": 2},
        )
        for gen_task in ("heatmap", "repair"):
            qualities = Counter(
                sample["gen_quality"]
                for sample in samples
                if sample["gen_task"] == gen_task
            )
            self.assertEqual(qualities["good"], qualities["bad"])

    def test_good_and_bad_use_identical_task_prompts(self):
        self.dataset.parse_row(self.row, "unused")

        self.assertEqual(
            self.tokenizer.prompts,
            [
                SINGLE_HEATMAP_PROMPT,
                SINGLE_HEATMAP_PROMPT,
                SINGLE_HEATMAP_PROMPT,
                SINGLE_HEATMAP_PROMPT,
                PAIR_HEATMAP_PROMPT,
                PAIR_HEATMAP_PROMPT,
                REFINE_PROMPT,
                REFINE_PROMPT,
            ],
        )

    def test_inputs_and_targets_follow_recipe(self):
        samples = self.dataset.parse_row(self.row, "unused")
        by_name = {sample["task_name"]: sample for sample in samples}

        # Every conditioning image appears twice: once for VAE and once for ViT.
        self.assertEqual(self.markers(by_name["single_good_heatmap"]), [10, 10, 0])
        self.assertEqual(self.markers(by_name["single_bad_heatmap"]), [20, 20, 255])
        self.assertEqual(
            self.markers(by_name["pair_good_heatmap"]),
            [10, 10, 10, 10, 0],
        )
        self.assertEqual(
            self.markers(by_name["pair_bad_heatmap"]),
            [20, 20, 10, 10, 255],
        )
        self.assertEqual(self.markers(by_name["good_refine"]), [10, 10, 10])
        self.assertEqual(self.markers(by_name["bad_refine"]), [20, 20, 10])

        for sample in samples:
            text_items = [
                item for item in sample["sequence_plan"] if item["type"] == "text"
            ]
            self.assertEqual(len(text_items), 1)
            self.assertEqual(text_items[0]["loss"], 0)


if __name__ == "__main__":
    unittest.main()

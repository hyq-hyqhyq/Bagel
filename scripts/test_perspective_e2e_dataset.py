# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import ast
import sys
import types
import unittest
from pathlib import Path

import torch
from PIL import Image

if "pyarrow" not in sys.modules:
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    sys.modules["pyarrow"] = pyarrow
    sys.modules["pyarrow.parquet"] = types.ModuleType("pyarrow.parquet")
    sys.modules["pyarrow.fs"] = types.ModuleType("pyarrow.fs")

from data.interleave_datasets.perspective_e2e_dataset import (
    PerspectiveE2EIterableDataset,
)
from data.reason_heatmap_prompts import (
    GEN_THINK_SYSTEM_PROMPT,
    PAIR_HEATMAP_PROMPT,
    REFINE_PROMPT,
)


class MarkerResize:
    def __call__(self, image):
        return image.resize((16, 16))


class MarkerTransform:
    stride = 1

    def __init__(self):
        self.resize_transform = MarkerResize()

    def __call__(self, image):
        marker = image.getpixel((0, 0))[0]
        return torch.full((3, 16, 16), marker, dtype=torch.float32)


class RecordingTokenizer:
    def __init__(self):
        self.values = {}

    def encode(self, text):
        self.values.setdefault(text, len(self.values) + 1)
        return [self.values[text]]


class PerspectiveE2EDatasetTest(unittest.TestCase):
    def setUp(self):
        images = {
            "good.png": Image.new("RGB", (20, 20), (10, 0, 0)),
            "bad.png": Image.new("RGB", (20, 20), (20, 0, 0)),
            "bad_heatmap.png": Image.new("RGB", (20, 20), (255, 255, 255)),
        }
        self.row = {
            "good_image": "good.png",
            "bad_image": "bad.png",
            "bad_heatmap": "bad_heatmap.png",
            "good_reason": "no error",
            "bad_reason": "projection error",
            "good_score": 1.0,
            "bad_score": 0.0,
        }
        self.dataset = object.__new__(PerspectiveE2EIterableDataset)
        self.dataset.transform = MarkerTransform()
        self.dataset.vit_transform = MarkerTransform()
        self.dataset.tokenizer = RecordingTokenizer()
        self.dataset._read_image = (
            lambda path: images[Path(path).name].copy()
        )

    def test_row_stays_coupled_and_is_balanced(self):
        good, bad = self.dataset.parse_row(self.row, "unused")
        self.assertEqual((good["quality"], bad["quality"]), ("good", "bad"))
        self.assertEqual(good["image_size"], (16, 16))
        self.assertEqual(bad["image_size"], (16, 16))
        self.assertEqual(good["score_label"], 1.0)
        self.assertEqual(bad["score_label"], 0.0)

        self.assertEqual(good["original_vae_image"][0, 0, 0].item(), 10)
        self.assertEqual(good["refined_vae_image"][0, 0, 0].item(), 10)
        self.assertEqual(good["heatmap_vae_image"][0, 0, 0].item(), 0)
        self.assertEqual(bad["original_vae_image"][0, 0, 0].item(), 20)
        self.assertEqual(bad["refined_vae_image"][0, 0, 0].item(), 10)
        self.assertEqual(bad["heatmap_vae_image"][0, 0, 0].item(), 255)

    def test_prompts_are_shared_across_quality(self):
        good, bad = self.dataset.parse_row(self.row, "unused")
        for key in ("system_ids", "repair_prompt_ids", "heatmap_prompt_ids"):
            self.assertEqual(good[key], bad[key])
        tokenizer = self.dataset.tokenizer
        self.assertEqual(good["system_ids"], [tokenizer.values[GEN_THINK_SYSTEM_PROMPT]])
        self.assertEqual(good["repair_prompt_ids"], [tokenizer.values[REFINE_PROMPT]])
        self.assertEqual(
            good["heatmap_prompt_ids"],
            [tokenizer.values[PAIR_HEATMAP_PROMPT]],
        )

    def test_loss_weights_match_the_agreed_design(self):
        source = Path("train/reason_heatmap_e2e.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "LOSS_WEIGHTS"
                for target in node.targets
            )
        )
        self.assertEqual(
            ast.literal_eval(assignment.value),
            {
                "heatmap_flow": 10.0,
                "repair_flow": 1.0,
                "heatmap_score": 1.0,
                "repair_score": 0.2,
                "heatmap_reason": 0.25,
                "repair_reason": 0.05,
            },
        )


if __name__ == "__main__":
    unittest.main()

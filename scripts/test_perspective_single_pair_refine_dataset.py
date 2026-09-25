# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections import Counter
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

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
if "decord" not in sys.modules:
    decord = types.ModuleType("decord")
    decord.VideoReader = object
    decord.video_reader = types.SimpleNamespace(VideoReader=object)
    sys.modules["decord"] = decord

from data.interleave_datasets.perspective_single_pair_refine_dataset import (
    PerspectiveSinglePairRefineIterableDataset,
)
from data.interleave_datasets.reason_heatmap_dataset import (
    ReasonHeatmapIterableDataset,
)
from data.dataset_base import PackedDataset
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
            "good_reason": "good reason",
            "bad_reason": "bad reason",
            "pair_reason": "pair reason",
        }
        self.tokenizer = RecordingTokenizer()
        self.dataset = object.__new__(
            PerspectiveSinglePairRefineIterableDataset
        )
        self.dataset.transform = MarkerTransform()
        self.dataset.vit_transform = MarkerTransform()
        self.dataset.tokenizer = self.tokenizer
        self.dataset.task_ratio = (2, 1, 1)
        self.dataset.include_reason = False
        self.dataset.disable_heatmap_visual_dropout = False
        self.dataset.use_pair_reason = False
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

    def test_visual_dropout_can_be_disabled_only_for_heatmap(self):
        self.dataset.disable_heatmap_visual_dropout = True
        samples = self.dataset.parse_row(self.row, "unused")

        for sample in samples:
            conditioning_images = [
                item
                for item in sample["sequence_plan"]
                if item["type"] in {"vae_image", "vit_image"}
                and item["loss"] == 0
            ]
            expected_enable_cfg = int(sample["gen_task"] == "repair")
            self.assertTrue(conditioning_images)
            self.assertTrue(
                all(
                    item["enable_cfg"] == expected_enable_cfg
                    for item in conditioning_images
                )
            )

    def test_visual_dropout_remains_enabled_by_default(self):
        samples = self.dataset.parse_row(self.row, "unused")

        for sample in samples:
            conditioning_images = [
                item
                for item in sample["sequence_plan"]
                if item["type"] in {"vae_image", "vit_image"}
                and item["loss"] == 0
            ]
            self.assertTrue(conditioning_images)
            self.assertTrue(
                all(item["enable_cfg"] == 1 for item in conditioning_images)
            )

    def test_pair_bad_can_use_pair_specific_reason(self):
        self.dataset.task_ratio = (0, 1, 0)
        self.dataset.include_reason = True
        self.dataset.use_pair_reason = True

        self.dataset.parse_row(self.row, "unused")

        self.assertEqual(
            self.tokenizer.prompts,
            [
                PAIR_HEATMAP_PROMPT,
                "<think>good reason</think>",
                PAIR_HEATMAP_PROMPT,
                "<think>pair reason</think>",
            ],
        )

    def test_pair_bad_can_keep_legacy_bad_reason_when_explicitly_disabled(self):
        self.dataset.task_ratio = (0, 1, 0)
        self.dataset.include_reason = True

        self.dataset.parse_row(self.row, "unused")

        self.assertEqual(
            self.tokenizer.prompts[-1],
            "<think>bad reason</think>",
        )

    def test_pair_reason_is_enabled_by_default(self):
        def fake_parent_init(instance, *args, **kwargs):
            instance.dataset_name = "test"

        with patch.object(
            ReasonHeatmapIterableDataset, "__init__", fake_parent_init
        ), patch.dict(os.environ, {}, clear=True):
            dataset = PerspectiveSinglePairRefineIterableDataset()

        self.assertTrue(dataset.use_pair_reason)

    def test_judgment_targets_are_separate_from_reason_and_global(self):
        self.dataset.task_ratio = (1, 0, 0)
        self.dataset.include_reason = True
        self.dataset.include_judgment = True
        row = dict(self.row)
        row["judge"] = {
            "checks": {"good_reason": [1, 1], "bad_reason": [1, 0]},
            "global": {"good": 1, "bad": 0},
        }
        samples = self.dataset.parse_row(row, "unused")
        self.assertEqual(
            [item["loss_type"] for item in samples[0]["sequence_plan"] if item["type"] == "text"],
            ["reason", "reason", "judgment", "judgment", "global"],
        )
        self.assertIn("Check 2 conclusion: incorrect", self.tokenizer.prompts[-2])
        self.assertIn("Global conclusion: incorrect", self.tokenizer.prompts[-1])

    def test_on_policy_judgment_uses_compact_reason_and_rollout_metadata(self):
        self.dataset.task_ratio = (1, 0, 0)
        self.dataset.include_reason = True
        self.dataset.include_judgment = True
        self.dataset.on_policy_judgment = True
        reason = """Scene:
A tiled room.

Structures:
- floor

Checks:
Check 1:
Elements:
- floor lines
Expected relationship:
The lines should converge.
Inspection:
The lines are badly skewed.

Check 2:
Elements:
- wall edges
Expected relationship:
The edges should remain upright.
Inspection:
The wall remains correct.

Conclusion:
The floor is inconsistent."""
        row = dict(self.row)
        row["good_reason"] = reason
        row["bad_reason"] = reason
        row["judge"] = {
            "checks": {"good_reason": [1, 1], "bad_reason": [0, 1]},
            "global": {"good": 1, "bad": 0},
        }

        samples = self.dataset.parse_row(row, "unused")
        bad = samples[1]
        supervised_text = [
            text
            for text in self.tokenizer.prompts
            if text.startswith("<think>")
        ][1]
        self.assertNotIn("Inspection:", supervised_text)
        self.assertNotIn("Conclusion:", supervised_text)
        self.assertIn("Expected relationship:", supervised_text)
        self.assertEqual(
            [
                item["loss_type"]
                for item in bad["sequence_plan"]
                if item["type"] == "text" and item["loss"]
            ],
            ["reason"],
        )
        rollout = bad["judgment_rollout"]
        self.assertEqual(rollout["check_labels"], [0, 1])
        self.assertEqual(rollout["global_label"], 0)
        self.assertEqual(len(rollout["vae_images"]), 1)
        self.assertEqual(len(rollout["vit_images"]), 1)

    def test_on_policy_rollouts_identify_all_three_logical_reasons(self):
        self.dataset.task_ratio = (1, 1, 0)
        self.dataset.include_reason = True
        self.dataset.include_judgment = True
        self.dataset.on_policy_judgment = True
        self.dataset.use_pair_reason = True
        row = dict(self.row)
        row["judge"] = {
            "checks": {
                "good_reason": [1, 1],
                "bad_reason": [0, 1],
                "pair_reason": [0, 1],
            },
            "global": {"good": 1, "bad": 0},
        }

        samples = self.dataset.parse_row(row, "unused")
        by_name = {sample["task_name"]: sample for sample in samples}

        self.assertEqual(
            by_name["single_good_heatmap"]["judgment_rollout"]["reason_key"],
            "good_reason",
        )
        self.assertEqual(
            by_name["single_bad_heatmap"]["judgment_rollout"]["reason_key"],
            "bad_reason",
        )
        self.assertEqual(
            by_name["pair_bad_heatmap"]["judgment_rollout"]["reason_key"],
            "pair_reason",
        )
        self.assertEqual(
            by_name["pair_bad_heatmap"]["judgment_rollout"]["check_labels"],
            [0, 1],
        )

    def test_tokenwise_reason_is_one_turn_with_balanced_field_metadata(self):
        self.dataset.task_ratio = (1, 0, 0)
        self.dataset.include_reason = True
        self.dataset.include_judgment = True
        self.dataset.on_policy_judgment = False
        self.dataset.tokenwise_judgment = True
        row = dict(self.row)
        row["judge"] = {
            "version": "visual_separate_focus_shared_controls_v3",
            "checks": {
                "good_reason": [1, 1],
                "bad_reason": [0, 1],
                "pair_reason": [0, 1],
            },
            "global": {"good": 1, "bad": 0},
        }

        def structured(labels, global_label):
            return {
                "scene": "A tiled room.",
                "structures": ["floor", "wall"],
                "checks": [
                    {
                        "elements": ["floor lines"],
                        "expected_relationship": "They share one convergence.",
                        "inspection": "The visible floor lines diverge from the boundary.",
                        "judgment": labels[0],
                    },
                    {
                        "elements": ["wall edges"],
                        "expected_relationship": "They remain upright.",
                        "inspection": "The visible wall edges retain one upright direction.",
                        "judgment": labels[1],
                    },
                ],
                "conclusion": "The floor projection conflicts with the stable wall.",
                "global_judgment": global_label,
            }

        row["tokenwise_reason"] = {
            "version": "visual_inspection_tokenwise_v1",
            "model": "gpt-test",
            "good_reason": structured([1, 1], 1),
            "bad_reason": structured([0, 1], 0),
            "pair_reason": structured([0, 1], 0),
        }

        samples = self.dataset.parse_row(row, "unused")
        bad = samples[1]
        supervised = [
            item for item in bad["sequence_plan"]
            if item["type"] == "text" and item["loss"]
        ]
        self.assertEqual(len(supervised), 1)
        item = supervised[0]
        self.assertEqual(
            len(item["token_loss_kinds"]),
            len(item["token_loss_polarities"]),
        )
        self.assertEqual(set(item["token_loss_kinds"]), {0, 1, 2, 3, 4})
        kind_polarities = list(zip(
            item["token_loss_kinds"], item["token_loss_polarities"]
        ))
        self.assertIn((1, 0), kind_polarities)
        self.assertIn((1, 1), kind_polarities)
        self.assertIn((2, 0), kind_polarities)
        self.assertIn((2, 1), kind_polarities)
        self.assertIn((3, 0), kind_polarities)
        self.assertIn((4, 0), kind_polarities)
        self.assertNotIn("judgment_rollout", bad)


class PackedJudgmentRolloutSelectionTest(unittest.TestCase):
    @staticmethod
    def candidate(name, reason_key=None):
        rollout = {"task_name": name}
        if reason_key is not None:
            rollout["reason_key"] = reason_key
        return {"judgment_rollout": rollout}

    def test_reservoir_does_not_permanently_keep_first_good_candidate(self):
        dataset = object.__new__(PackedDataset)
        dataset.judgment_rollout_max_samples_per_rank = 1
        status = {
            "judgment_rollouts": [],
            "judgment_rollout_seen": 0,
        }

        dataset._consider_judgment_rollout(self.candidate("good"), status)
        with patch("data.dataset_base.random.randrange", return_value=0):
            dataset._consider_judgment_rollout(self.candidate("bad"), status)

        self.assertEqual(status["judgment_rollout_seen"], 2)
        self.assertEqual(status["judgment_rollouts"][0]["task_name"], "bad")

    def test_reservoir_never_exceeds_configured_capacity(self):
        dataset = object.__new__(PackedDataset)
        dataset.judgment_rollout_max_samples_per_rank = 2
        status = {
            "judgment_rollouts": [],
            "judgment_rollout_seen": 0,
        }

        with patch("data.dataset_base.random.randrange", return_value=99):
            for index in range(8):
                dataset._consider_judgment_rollout(
                    self.candidate(str(index)), status
                )

        self.assertEqual(status["judgment_rollout_seen"], 8)
        self.assertEqual(len(status["judgment_rollouts"]), 2)

    def test_default_three_slots_keep_good_bad_and_pair_once_each(self):
        dataset = object.__new__(PackedDataset)
        dataset.judgment_rollout_max_samples_per_rank = 3
        status = {
            "judgment_rollouts": [],
            "judgment_rollout_seen": 0,
            "judgment_rollout_groups": {},
        }
        candidates = [
            self.candidate("good_refine", "good_reason"),
            self.candidate("single_good_heatmap", "good_reason"),
            self.candidate("single_bad_heatmap", "bad_reason"),
            self.candidate("pair_bad_heatmap", "pair_reason"),
            self.candidate("bad_refine", "bad_reason"),
        ]

        with patch("data.dataset_base.random.randrange", return_value=0):
            for candidate in candidates:
                dataset._consider_judgment_rollout(candidate, status)

        rollouts = status["judgment_rollouts"]
        self.assertEqual(
            [item["reason_key"] for item in rollouts],
            ["good_reason", "bad_reason", "pair_reason"],
        )
        self.assertEqual(
            [item["task_name"] for item in rollouts],
            [
                "single_good_heatmap",
                "single_bad_heatmap",
                "pair_bad_heatmap",
            ],
        )


if __name__ == "__main__":
    unittest.main()

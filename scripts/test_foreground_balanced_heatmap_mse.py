# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
import unittest
from types import SimpleNamespace

import torch

if "decord" not in sys.modules:
    sys.modules["decord"] = types.ModuleType("decord")

from data.dataset_base import PackedDataset


def make_packer(enabled):
    packer = PackedDataset.__new__(PackedDataset)
    packer.split_gen_adapter_by_task = True
    packer.gen_task_filter = "joint"
    packer.foreground_balanced_heatmap_mse = enabled
    packer.use_flex = True
    packer.bos_token_id = 10
    packer.eos_token_id = 11
    packer.start_of_image = 12
    packer.end_of_image = 13
    packer.data_config = SimpleNamespace(
        text_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vae_image_downsample=2,
        max_latent_size=8,
    )
    packer.get_flattened_position_ids = (
        lambda height, width, stride, max_num_patches_per_side: torch.arange(
            height * width // stride**2
        )
    )
    return packer


def make_heatmap_sample(quality):
    image = torch.full((3, 4, 4), -1.0)
    if quality == "bad":
        image[:, :2, :2] = 1.0
    return {
        "gen_task": "heatmap",
        "gen_quality": quality,
        "image_tensor_list": [image],
        "text_ids_list": [],
        "sequence_plan": [
            {
                "type": "vae_image",
                "enable_cfg": 0,
                "loss": 1,
                "special_token_loss": 0,
                "special_token_label": None,
            }
        ],
    }


class ForegroundBalancedHeatmapMseTest(unittest.TestCase):
    def test_bad_heatmap_foreground_is_max_pooled_to_latent_tokens(self):
        packer = make_packer(enabled=True)
        status = packer.pack_sequence(
            make_heatmap_sample("bad"), packer.set_sequence_status()
        )

        self.assertEqual(status["mse_task_labels"], [3, 3, 3, 3])
        self.assertEqual(
            status["mse_foreground_labels"], [True, False, False, False]
        )

    def test_default_path_does_not_create_foreground_labels(self):
        packer = make_packer(enabled=False)
        status = packer.pack_sequence(
            make_heatmap_sample("bad"), packer.set_sequence_status()
        )

        self.assertEqual(status["mse_foreground_labels"], [])


if __name__ == "__main__":
    unittest.main()

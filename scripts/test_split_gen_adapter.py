from types import SimpleNamespace
import unittest

import torch
from torch import nn

from data.dataset_base import PackedDataset
from modeling.bagel.bagel import Bagel, GenerationAdapter


def make_packer():
    packer = PackedDataset.__new__(PackedDataset)
    packer.split_gen_adapter_by_task = True
    packer.gen_task_filter = "joint"
    packer.use_flex = True
    packer.bos_token_id = 10
    packer.eos_token_id = 11
    packer.data_config = SimpleNamespace(text_cond_dropout_prob=0.0)
    return packer


def make_text_sample(gen_task):
    return {
        "gen_task": gen_task,
        "image_tensor_list": [],
        "text_ids_list": [[1, 2]],
        "sequence_plan": [
            {
                "type": "text",
                "enable_cfg": 0,
                "loss": 1,
                "special_token_loss": 0,
                "special_token_label": 0,
            }
        ],
    }


class SplitGenerationAdapterTest(unittest.TestCase):
    def test_packed_batch_rejects_mixed_generation_tasks(self):
        packer = make_packer()
        status = packer.set_sequence_status()
        status = packer.pack_sequence(make_text_sample("repair"), status)

        self.assertEqual(status["gen_task"], "repair")
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            packer.pack_sequence(make_text_sample("heatmap"), status)

    def test_legacy_weights_are_copied_to_both_adapters(self):
        holder = SimpleNamespace(
            config=SimpleNamespace(split_gen_adapter_by_task=True)
        )
        legacy = {
            "vae2llm.weight": torch.randn(3, 2),
            "vae2llm.bias": torch.randn(3),
            "llm2vae.weight": torch.randn(2, 3),
            "llm2vae.bias": torch.randn(2),
            "language_model.weight": torch.randn(1),
        }

        migrated = Bagel._migrate_generation_adapter_state_dict(
            holder, legacy
        )

        for module_name in ("vae2llm", "llm2vae"):
            for parameter_name in ("weight", "bias"):
                old_key = f"{module_name}.{parameter_name}"
                self.assertNotIn(old_key, migrated)
                repair_key = (
                    f"repair_gen_adapter.{module_name}.{parameter_name}"
                )
                heatmap_key = (
                    f"heatmap_gen_adapter.{module_name}.{parameter_name}"
                )
                self.assertTrue(
                    torch.equal(migrated[repair_key], legacy[old_key])
                )
                self.assertTrue(
                    torch.equal(migrated[heatmap_key], legacy[old_key])
                )
                self.assertNotEqual(
                    migrated[repair_key].data_ptr(),
                    migrated[heatmap_key].data_ptr(),
                )

        self.assertIs(
            migrated["language_model.weight"],
            legacy["language_model.weight"],
        )

    def test_split_adapter_requires_explicit_task(self):
        model = Bagel.__new__(Bagel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(split_gen_adapter_by_task=True)
        model.repair_gen_adapter = GenerationAdapter(2, 3)
        model.heatmap_gen_adapter = GenerationAdapter(2, 3)

        repair = model._get_generation_adapter("repair")
        heatmap = model._get_generation_adapter("heatmap")

        self.assertIs(repair[0], model.repair_gen_adapter.vae2llm)
        self.assertIs(repair[1], model.repair_gen_adapter.llm2vae)
        self.assertIs(heatmap[0], model.heatmap_gen_adapter.vae2llm)
        self.assertIs(heatmap[1], model.heatmap_gen_adapter.llm2vae)
        with self.assertRaisesRegex(ValueError, "explicit gen_task"):
            model._get_generation_adapter(None)


if __name__ == "__main__":
    unittest.main()

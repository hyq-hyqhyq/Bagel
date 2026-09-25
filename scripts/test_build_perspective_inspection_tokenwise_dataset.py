import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    sys.modules["openai"] = openai

import build_perspective_inspection_tokenwise_dataset as build


def reason(prefix):
    return f"""Scene:
{prefix} room

Structures:
- floor
- wall

Checks:
Check 1:
Elements:
- floor lines
Expected relationship:
The floor lines should share one convergence region.

Check 2:
Elements:
- wall edges
Expected relationship:
The wall edges should remain upright.

Conclusion:
old {prefix} conclusion"""


def source_row():
    return {
        "group_id": "group-1",
        "good_image": "train/good/group-1.jpg",
        "bad_image": "train/bad/group-1.jpg",
        "bad_heatmap": "train/bad_heatmap/group-1.png",
        "good_reason": reason("good"),
        "bad_reason": reason("bad"),
        "pair_reason": reason("pair"),
        "judge": {
            "version": build.SOURCE_JUDGE_VERSION,
            "checks": {
                "good_reason": [1, 1],
                "bad_reason": [0, 1],
                "pair_reason": [0, 1],
            },
            "global": {"good": 1, "bad": 0},
        },
    }


def api_result():
    return {
        "good_reason": {
            "inspections": [
                "Both floor lines narrow toward the same rear region.",
                "The wall edges retain the same upright direction.",
            ],
            "conclusion": "The visible structures follow one coherent projection.",
        },
        "bad_reason": {
            "inspections": [
                "The internal floor lines fan away from the stable boundary direction.",
                "The wall edges retain the same upright direction.",
            ],
            "conclusion": "The floor grid conflicts with the surrounding room projection.",
        },
        "pair_reason": {
            "inspections": [
                "GOOD shares one floor convergence while BAD fans toward another region.",
                "Both GOOD and BAD retain the same upright wall direction.",
            ],
            "conclusion": "BAD changes the floor projection that GOOD preserves.",
        },
    }


class InspectionTokenwiseDatasetBuilderTest(unittest.TestCase):
    def test_one_api_call_builds_all_three_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = source_row()
            for key in ("good_image", "bad_image", "bad_heatmap"):
                path = root / row[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
            calls = []

            def create(**kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(
                    output_text=json.dumps(api_result())
                )

            client = types.SimpleNamespace(
                responses=types.SimpleNamespace(create=create)
            )
            args = argparse.Namespace(
                image_root=root,
                base_url="unused",
                model="gpt-test",
                api_timeout=180,
            )
            with patch.object(build, "OpenAI", return_value=client):
                output = build.generate_row(row, "key", args)

            self.assertEqual(len(calls), 1)
            self.assertEqual(
                [part["type"] for part in calls[0]["input"][0]["content"]],
                ["input_text", "input_text", "input_image", "input_text", "input_image"],
            )
            supervision = output["tokenwise_reason"]
            self.assertEqual(supervision["version"], build.DATA_VERSION)
            self.assertEqual(
                [item["judgment"] for item in supervision["bad_reason"]["checks"]],
                [0, 1],
            )
            self.assertEqual(supervision["bad_reason"]["global_judgment"], 0)
            self.assertIn("Inspection:", output["bad_reason"])
            self.assertIn("Check judgment:\nVIOLATED", output["bad_reason"])
            self.assertIn("Global judgment:\nINCONSISTENT", output["bad_reason"])
            self.assertEqual(
                output["source_conclusions"]["bad_reason"],
                "old bad conclusion",
            )

    def test_copies_referenced_assets_as_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            output_root = root / "portable"
            row = source_row()
            for index, key in enumerate(("good_image", "bad_image", "bad_heatmap")):
                path = image_root / row[key]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"asset-{index}".encode())

            build.copy_assets([row], image_root, output_root, workers=2)

            for key in ("good_image", "bad_image", "bad_heatmap"):
                copied = output_root / row[key]
                self.assertTrue(copied.is_file())
                self.assertFalse(copied.is_symlink())
                self.assertEqual(
                    copied.read_bytes(), (image_root / row[key]).read_bytes()
                )

    def test_parser_rejects_inspection_count_mismatch(self):
        expected = build.trusted_payload(source_row())
        response = api_result()
        response["bad_reason"]["inspections"] = ["only one usable inspection"]
        with self.assertRaisesRegex(ValueError, "expected 2 inspections"):
            build.parse_generated(json.dumps(response), expected)


if __name__ == "__main__":
    unittest.main()

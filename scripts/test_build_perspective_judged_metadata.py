import json
import sys
import types
import unittest
from unittest.mock import patch

if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    sys.modules["openai"] = openai

import build_perspective_judged_metadata as judged


class JudgedMetadataTest(unittest.TestCase):
    def test_text_only_checks_and_pair_target(self):
        row = {
            "group_id": "x",
            "good_reason": {
                "scene": "Street", "structures": [],
                "checks": [{"elements": ["road"], "expected_relationship": "align", "inspection": "aligned"}],
                "conclusion": "coherent",
            },
            "bad_reason": {
                "scene": "Street", "structures": [],
                "checks": [{"elements": ["road"], "expected_relationship": "align", "inspection": "skewed"}],
                "conclusion": "inconsistent",
            },
            "pair_reason": "Compared with GOOD, BAD is skewed.",
            "bad_score_target": 0.2,
        }
        row = judged.prepared_row(row)
        payload, labels = judged.build_payload(row)
        self.assertEqual(set(payload), {"good_reason", "bad_reason"})
        self.assertEqual(labels, {"good": 1, "bad": 0})
        self.assertEqual(row["bad_score"], 0.2)
        self.assertEqual(len(judged.as_checks(row["bad_reason"])), 1)
        self.assertNotIn("Conclusion:", judged.as_checks(row["bad_reason"])[0])
        response = types.SimpleNamespace(output_text=json.dumps({
            "checks": {"good_reason": [1], "bad_reason": [0]}
        }))
        client = types.SimpleNamespace(responses=types.SimpleNamespace(create=lambda **kwargs: response))
        with patch.object(judged, "OpenAI", return_value=client):
            output = judged.judge(row, "unused", "unused", "gpt-5.6-terra")
        self.assertEqual(output["checks"]["pair_reason"], [0])
        self.assertEqual(output["global"], {"good": 1, "bad": 0})

    def test_rejects_missing_check_labels(self):
        with self.assertRaisesRegex(ValueError, "expected 2"):
            judged.parse_result(
                '{"checks":{"good_reason":[1],"bad_reason":[0]}}',
                {"good_reason": ["one", "two"], "bad_reason": ["one"]},
            )


if __name__ == "__main__":
    unittest.main()

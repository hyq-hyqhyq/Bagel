# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data.perspective_judgment_utils import build_judgment_sequence


class CharacterTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(int(value)) for value in ids)


class JudgmentRolloutSequenceTest(unittest.TestCase):
    def test_inserts_each_judgment_after_its_generated_check(self):
        tokenizer = CharacterTokenizer()
        generated = tokenizer.encode(
            "<think>Scene:\nRoom\n\n"
            "Check 1:\nElements: floor\nExpected relationship: converge\n\n"
            "Check 2:\nElements: wall\nExpected relationship: upright"
            "</think>"
        )
        sequence, positions, targets, kinds = build_judgment_sequence(
            tokenizer, generated, [0, 1], 0
        )
        text = tokenizer.decode(sequence)
        self.assertLess(text.index("Check 1:"), text.index("Judgment: VIOLATED"))
        self.assertLess(
            text.index("Judgment: VIOLATED"), text.index("Check 2:")
        )
        self.assertLess(text.index("Check 2:"), text.index("Judgment: SATISFIED"))
        self.assertTrue(text.endswith("Global judgment: INCONSISTENT</think>"))
        self.assertEqual(
            [sequence[position] for position in positions],
            targets,
        )
        self.assertEqual(kinds.count(1), len(" VIOLATED SATISFIED"))
        self.assertEqual(kinds.count(2), len(" INCONSISTENT"))

    def test_malformed_structure_uses_indexed_fallback(self):
        tokenizer = CharacterTokenizer()
        sequence, positions, targets, kinds = build_judgment_sequence(
            tokenizer,
            tokenizer.encode("free-form explanation"),
            [1, 0],
            1,
        )
        text = tokenizer.decode(sequence)
        self.assertIn("Check 1 judgment: SATISFIED", text)
        self.assertIn("Check 2 judgment: VIOLATED", text)
        self.assertIn("Global judgment: CONSISTENT", text)
        self.assertEqual([sequence[index] for index in positions], targets)
        self.assertIn(1, kinds)
        self.assertIn(2, kinds)


if __name__ == "__main__":
    unittest.main()

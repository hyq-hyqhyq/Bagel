# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight text utilities for perspective judgment supervision."""

from __future__ import annotations

import re
from typing import Iterable


CHECK_PATTERN = re.compile(r"(?im)^\s*Check\s+\d+\s*:\s*")


def _token_ids(tokenizer, text: str) -> list[int]:
    return [int(value) for value in tokenizer.encode(text)]


def _strip_think_wrapper(text: str) -> str:
    text = text.strip()
    if text.startswith("<think>"):
        text = text[len("<think>") :]
    if text.endswith("</think>"):
        text = text[: -len("</think>")]
    return text.strip()


def count_generated_checks(tokenizer, generated_ids: Iterable[int]) -> int:
    decoded = tokenizer.decode(list(generated_ids))
    return len(CHECK_PATTERN.findall(_strip_think_wrapper(decoded)))


def build_judgment_sequence(
    tokenizer,
    generated_ids: Iterable[int],
    check_labels: list[int],
    global_label: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Insert verdict targets after generated checks.

    Returns ``sequence_ids, label_positions, label_ids, label_kinds``. Kind 1
    denotes a local check and kind 2 denotes the global verdict. The returned
    positions index next-token predictors in the BOS-shifted sequence used by
    BAGEL's text update.
    """
    if any(int(value) not in (0, 1) for value in check_labels):
        raise ValueError("check labels must be binary")
    if int(global_label) not in (0, 1):
        raise ValueError("global label must be binary")

    decoded = tokenizer.decode(list(generated_ids))
    explanation = _strip_think_wrapper(decoded)
    matches = list(CHECK_PATTERN.finditer(explanation))

    sequence_ids: list[int] = []
    label_positions: list[int] = []
    label_ids: list[int] = []
    label_kinds: list[int] = []

    def append_text(text: str) -> None:
        sequence_ids.extend(_token_ids(tokenizer, text))

    def append_label(text: str, kind: int) -> None:
        ids = _token_ids(tokenizer, text)
        if not ids:
            raise ValueError(f"Tokenizer produced no ids for label {text!r}")
        start = len(sequence_ids)
        sequence_ids.extend(ids)
        label_positions.extend(range(start, start + len(ids)))
        label_ids.extend(ids)
        label_kinds.extend([kind] * len(ids))

    append_text("<think>")
    if len(matches) == len(check_labels) and matches:
        cursor = 0
        for index, (_, value) in enumerate(zip(matches, check_labels)):
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(explanation)
            )
            append_text(explanation[cursor:end].rstrip())
            append_text("\nJudgment:")
            append_label(" SATISFIED" if int(value) else " VIOLATED", kind=1)
            append_text("\n\n")
            cursor = end
    else:
        # Early in training the generated structure can be malformed. Keep
        # judgment supervision usable by appending indexed queries after the
        # complete generated explanation instead of dropping the sample.
        append_text(explanation)
        append_text("\n\n")
        for index, value in enumerate(check_labels, start=1):
            append_text(f"Check {index} judgment:")
            append_label(" SATISFIED" if int(value) else " VIOLATED", kind=1)
            append_text("\n")

    append_text("Global judgment:")
    append_label(
        " CONSISTENT" if int(global_label) else " INCONSISTENT",
        kind=2,
    )
    append_text("</think>")
    return sequence_ids, label_positions, label_ids, label_kinds

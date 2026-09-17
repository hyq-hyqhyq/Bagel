# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Prepare score-compatible metadata with fully structured reasoning text."""

import argparse
import json
import os
from pathlib import Path
from statistics import mean, median


REASON_KEYS = ("good_reason", "bad_reason", "pair_reason")


def format_reason(reason):
    """Serialize every structured reason field into supervised text."""
    if not isinstance(reason, dict):
        raise TypeError(f"Expected a structured reason object, got {type(reason)}")

    lines = ["Scene:", str(reason.get("scene", "")).strip(), "", "Structures:"]
    structures = reason.get("structures", [])
    lines.extend(f"- {item}" for item in structures)
    lines.extend(["", "Checks:"])

    for index, check in enumerate(reason.get("checks", []), start=1):
        lines.extend([f"Check {index}:", "Elements:"])
        lines.extend(f"- {item}" for item in check.get("elements", []))
        lines.extend(
            [
                "Expected relationship:",
                str(check.get("expected_relationship", "")).strip(),
                "Inspection:",
                str(check.get("inspection", "")).strip(),
                "",
            ]
        )

    lines.extend(
        ["Conclusion:", str(reason.get("conclusion", "")).strip()]
    )
    return "\n".join(lines).strip()


def convert_split(source, destination, data_root):
    word_counts = {key: [] for key in REASON_KEYS}
    row_count = 0
    missing_assets = []
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    with source.open("r", encoding="utf-8") as reader, temporary.open(
        "w", encoding="utf-8"
    ) as writer:
        for row_count, line in enumerate(reader, start=1):
            row = json.loads(line)
            for key in REASON_KEYS:
                row[key] = format_reason(row[key])
                word_counts[key].append(len(row[key].split()))

            row["good_score"] = 1.0
            row["bad_score"] = float(row["bad_score_target"])

            for key in ("good_image", "bad_image", "bad_heatmap"):
                asset = data_root / row[key]
                if not asset.is_file():
                    missing_assets.append(str(asset))

            writer.write(json.dumps(row, ensure_ascii=False) + "\n")

    if missing_assets:
        temporary.unlink(missing_ok=True)
        preview = "\n".join(missing_assets[:10])
        raise FileNotFoundError(
            f"Found {len(missing_assets)} missing assets. First entries:\n{preview}"
        )

    os.replace(temporary, destination)
    return row_count, word_counts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument(
        "--output-suffix", default="_longreason_allfields_v1"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for split in ("train", "test"):
        source = args.metadata_dir / f"{split}.jsonl"
        destination = args.metadata_dir / (
            f"{split}{args.output_suffix}.jsonl"
        )
        count, word_counts = convert_split(
            source, destination, args.data_root
        )
        print(f"{split}: wrote {count} rows -> {destination}")
        for key, values in word_counts.items():
            print(
                f"  {key}: mean={mean(values):.1f}, "
                f"median={median(values):.1f}, min={min(values)}, "
                f"max={max(values)}"
            )


if __name__ == "__main__":
    main()

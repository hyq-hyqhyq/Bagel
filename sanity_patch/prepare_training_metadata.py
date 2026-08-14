# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path


SPLITS = ("train", "val", "test")
GOOD_REASON = "No synthetic rectangular patch corruption is present in this image."
PAIR_EXPLANATION_TEMPLATE = (
    "Compared with the original image, the corrupted image contains a {color} "
    "rectangular patch in the {region} region as a synthetic local corruption."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add reason_heatmap training fields to sanity patch metadata."
    )
    parser.add_argument("--data-root", type=Path, default=Path("sanity_patch_data"))
    return parser.parse_args()


def resolve_data_path(data_root, value):
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def enrich_record(record):
    required_fields = (
        "source_image_path",
        "corrupted_image_path",
        "heatmap_path",
        "color_name",
        "region_name",
        "explanation",
    )
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise KeyError(f"Missing fields {missing} in sample {record.get('sample_id')}")

    record.update(
        {
            "good_image": record["source_image_path"],
            "bad_image": record["corrupted_image_path"],
            "bad_heatmap": record["heatmap_path"],
            "good_reason": GOOD_REASON,
            "bad_reason": record["explanation"],
            "pair_reason": PAIR_EXPLANATION_TEMPLATE.format(
                color=record["color_name"],
                region=record["region_name"],
            ),
        }
    )
    return record


def prepare_split(data_root, split):
    metadata_path = data_root / "metadata" / f"{split}.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    records = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = enrich_record(json.loads(line))
            for field in ("good_image", "bad_image", "bad_heatmap"):
                path = resolve_data_path(data_root, record[field])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {field} for {split} line {line_number}: {path}"
                    )
            records.append(record)

    temporary_path = metadata_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(metadata_path)
    print(f"Prepared {len(records)} {split} records.")
    return len(records)


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    counts = {split: prepare_split(data_root, split) for split in SPLITS}

    info_path = data_root / "metadata" / "dataset_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    dataset_info.update(
        {
            "good_reason": GOOD_REASON,
            "pair_explanation_template": PAIR_EXPLANATION_TEMPLATE,
            "reason_heatmap_compatible": True,
            "counts": counts,
        }
    )
    temporary_info_path = info_path.with_suffix(".json.tmp")
    with temporary_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temporary_info_path.replace(info_path)
    print(f"Prepared training metadata in {data_root}")


if __name__ == "__main__":
    main()

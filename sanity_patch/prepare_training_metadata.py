# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

from PIL import Image, ImageStat


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
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--train-size", type=int, default=3960)
    parser.add_argument("--test-size", type=int, default=40)
    return parser.parse_args()


def resolve_data_path(data_root, value):
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def image_black_ratio(image):
    mean_gray = ImageStat.Stat(image.convert("L")).mean[0]
    return max(0.0, min(1.0, 1.0 - mean_gray / 255.0))


def heatmap_black_ratio(path):
    with Image.open(path) as image:
        return image_black_ratio(image)


def enrich_record(record, data_root):
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

    heatmap_path = resolve_data_path(data_root, record["heatmap_path"])
    if not heatmap_path.is_file():
        raise FileNotFoundError(f"Missing heatmap for sample {record.get('sample_id')}: {heatmap_path}")
    bad_score = heatmap_black_ratio(heatmap_path)

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
            "good_score": 1.0,
            "bad_score": bad_score,
            "pair_score": bad_score,
        }
    )
    return record


def write_jsonl(path, records):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def prepare_split(data_root, split):
    metadata_path = data_root / "metadata" / f"{split}.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    records = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = enrich_record(json.loads(line), data_root)
            for field in ("good_image", "bad_image", "bad_heatmap"):
                path = resolve_data_path(data_root, record[field])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing {field} for {split} line {line_number}: {path}"
                    )
            records.append(record)

    write_jsonl(metadata_path, records)
    print(f"Prepared {len(records)} {split} records.")
    return records


def split_train_test_records(records, train_size, test_size):
    if train_size <= 0 or test_size <= 0:
        raise ValueError("Train and test sizes must both be positive.")

    expected_size = train_size + test_size
    if len(records) != expected_size:
        raise ValueError(
            f"Expected {expected_size} source train records for a "
            f"{train_size}/{test_size} split, got {len(records)}."
        )

    train_records = [
        {**record, "score_split": "train", "score_split_index": index}
        for index, record in enumerate(records[:train_size])
    ]
    test_records = [
        {**record, "score_split": "test", "score_split_index": index}
        for index, record in enumerate(records[train_size:])
    ]
    return train_records, test_records


def prepare_train_test_split(data_root, records, train_size, test_size):
    train_records, test_records = split_train_test_records(
        records,
        train_size,
        test_size,
    )
    train_path = data_root / "metadata" / f"train_{train_size}.jsonl"
    test_path = data_root / "metadata" / f"test_last{test_size}.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)
    print(f"Prepared score split: {len(train_records)} train, {len(test_records)} test.")
    return train_path, test_path


def main():
    args = parse_args()
    data_root = args.data_root.resolve()
    records_by_split = {
        split: prepare_split(data_root, split) for split in args.splits
    }
    counts = {split: len(records) for split, records in records_by_split.items()}

    score_split = None
    if "train" in records_by_split:
        train_path, test_path = prepare_train_test_split(
            data_root,
            records_by_split["train"],
            args.train_size,
            args.test_size,
        )
        score_split = {
            "train_size": args.train_size,
            "test_size": args.test_size,
            "train_metadata": train_path.relative_to(data_root).as_posix(),
            "test_metadata": test_path.relative_to(data_root).as_posix(),
        }

    info_path = data_root / "metadata" / "dataset_info.json"
    if set(args.splits) == set(SPLITS) and info_path.is_file():
        with info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
        dataset_info.update(
            {
                "good_reason": GOOD_REASON,
                "pair_explanation_template": PAIR_EXPLANATION_TEMPLATE,
                "reason_heatmap_compatible": True,
                "score_formula": "1 - mean(grayscale_heatmap) / 255",
                "score_split": score_split,
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

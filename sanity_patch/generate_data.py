# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SPLITS = ("train", "val", "test")
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}
DEFAULT_EXPLANATION_TEMPLATE = (
    "A {color} rectangular patch appears in the {region} region of the image "
    "as a synthetic local corruption."
)
DEFAULT_SOURCE_ROOT = os.environ.get(
    "BAGEL_REASON_HEATMAP_DATA_DIR",
    "/data/bagel/data/perspective_5k/canonical_5k_clean_4402",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a synthetic rectangular-patch sanity-check dataset."
    )
    parser.add_argument("--source-root", type=Path, default=Path(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", type=Path, default=Path("sanity_patch_data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area-fraction", type=float, default=0.02)
    parser.add_argument("--max-area-fraction", type=float, default=0.10)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.5)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.0)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=None,
        help="Optional limit for a quick smoke test.",
    )
    parser.add_argument(
        "--copy-source",
        action="store_true",
        help="Copy source images into images/source instead of referencing originals.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0 < args.min_area_fraction <= args.max_area_fraction < 1:
        raise ValueError("Area fractions must satisfy 0 < min <= max < 1.")
    if not 0 < args.min_aspect_ratio <= args.max_aspect_ratio:
        raise ValueError("Aspect ratios must satisfy 0 < min <= max.")
    if args.max_samples_per_split is not None and args.max_samples_per_split <= 0:
        raise ValueError("max-samples-per-split must be positive.")

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if source_root == output_dir or source_root.is_relative_to(output_dir):
        raise ValueError("output-dir must not contain source-root.")
    if output_dir in {Path.cwd().resolve(), repo_root}:
        raise ValueError("Refusing to use the repository root as output-dir.")


def read_split_rows(source_root, max_samples_per_split=None):
    rows_by_split = {}
    source_owners = {}

    for split in SPLITS:
        metadata_path = source_root / "metadata" / f"{split}.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        rows = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for row_index, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "good_image" not in row:
                    raise KeyError(f"Missing good_image in {metadata_path} row {row_index}")

                source_path = (source_root / row["good_image"]).resolve()
                source_key = os.path.normcase(str(source_path))
                previous_owner = source_owners.get(source_key)
                if previous_owner is not None:
                    raise ValueError(
                        f"Source image appears more than once: {source_path} "
                        f"({previous_owner} and {split}:{row_index})"
                    )
                source_owners[source_key] = f"{split}:{row_index}"
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing source image: {source_path}")

                if max_samples_per_split is None or len(rows) < max_samples_per_split:
                    rows.append((row_index, row, source_path))

        rows_by_split[split] = rows

    return rows_by_split


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    for split in SPLITS:
        (output_dir / "images" / "source" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "corrupted" / split).mkdir(
            parents=True, exist_ok=True
        )
        (output_dir / "heatmaps" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)


def make_sample_id(split, row_index, row, source_path):
    source_name = row.get("group_id") or source_path.stem
    source_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(source_name)).strip("_")
    source_name = source_name[:64] or "image"
    return f"{split}_{row_index:06d}_{source_name}"


def make_sample_rng(seed, source_path):
    payload = f"{seed}\0{source_path}".encode("utf-8")
    sample_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return random.Random(sample_seed)


def sample_bbox(width, height, rng, min_area, max_area, min_ratio, max_ratio):
    target_area = width * height * rng.uniform(min_area, max_area)
    aspect_ratio = math.exp(rng.uniform(math.log(min_ratio), math.log(max_ratio)))
    patch_width = max(1, round(math.sqrt(target_area * aspect_ratio)))
    patch_height = max(1, round(math.sqrt(target_area / aspect_ratio)))

    if patch_width > width:
        patch_width = width
        patch_height = min(height, max(1, round(target_area / patch_width)))
    if patch_height > height:
        patch_height = height
        patch_width = min(width, max(1, round(target_area / patch_height)))

    x0 = rng.randint(0, width - patch_width)
    y0 = rng.randint(0, height - patch_height)
    return [x0, y0, x0 + patch_width, y0 + patch_height]


def get_region_name(bbox, width, height):
    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) / (2 * width)
    center_y = (y0 + y1) / (2 * height)

    if 1 / 3 <= center_x <= 2 / 3 and 1 / 3 <= center_y <= 2 / 3:
        return "center"
    vertical = "upper" if center_y < 0.5 else "lower"
    horizontal = "left" if center_x < 0.5 else "right"
    return f"{vertical}-{horizontal}"


def relative_path(path, root):
    return path.relative_to(root).as_posix()


def generate_sample(
    split,
    row_index,
    row,
    source_path,
    output_dir,
    args,
):
    sample_id = make_sample_id(split, row_index, row, source_path)
    rng = make_sample_rng(args.seed, source_path)

    with Image.open(source_path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        width, height = source.size
        bbox = sample_bbox(
            width,
            height,
            rng,
            args.min_area_fraction,
            args.max_area_fraction,
            args.min_aspect_ratio,
            args.max_aspect_ratio,
        )
        color_name = rng.choice(tuple(COLORS))
        color = COLORS[color_name]
        region_name = get_region_name(bbox, width, height)

        x0, y0, x1, y1 = bbox
        corrupted = source.copy()
        ImageDraw.Draw(corrupted).rectangle(
            [x0, y0, x1 - 1, y1 - 1], fill=color
        )
        heatmap = Image.new("L", source.size, color=0)
        ImageDraw.Draw(heatmap).rectangle(
            [x0, y0, x1 - 1, y1 - 1], fill=255
        )

        corrupted_path = (
            output_dir / "images" / "corrupted" / split / f"{sample_id}.png"
        )
        heatmap_path = output_dir / "heatmaps" / split / f"{sample_id}.png"
        corrupted.save(corrupted_path)
        heatmap.save(heatmap_path)

        if args.copy_source:
            source_output_path = (
                output_dir / "images" / "source" / split / f"{sample_id}.png"
            )
            source.save(source_output_path)
            metadata_source_path = relative_path(source_output_path, output_dir)
        else:
            metadata_source_path = str(source_path)

    explanation = DEFAULT_EXPLANATION_TEMPLATE.format(
        color=color_name,
        region=region_name,
    )
    return {
        "sample_id": sample_id,
        "split": split,
        "source_row_index": row_index,
        "source_image_path": metadata_source_path,
        "corrupted_image_path": relative_path(corrupted_path, output_dir),
        "heatmap_path": relative_path(heatmap_path, output_dir),
        "bbox_xyxy": bbox,
        "color_name": color_name,
        "region_name": region_name,
        "explanation": explanation,
    }


def write_jsonl(path, records):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def main():
    args = parse_args()
    validate_args(args)
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    rows_by_split = read_split_rows(source_root, args.max_samples_per_split)
    prepare_output_dir(output_dir, args.overwrite)

    counts = {}
    for split in SPLITS:
        records = []
        for row_index, row, source_path in rows_by_split[split]:
            records.append(
                generate_sample(
                    split,
                    row_index,
                    row,
                    source_path,
                    output_dir,
                    args,
                )
            )
        write_jsonl(output_dir / "metadata" / f"{split}.jsonl", records)
        counts[split] = len(records)
        print(f"Generated {len(records)} {split} samples.")

    dataset_info = {
        "source_root": str(source_root),
        "seed": args.seed,
        "colors": COLORS,
        "min_area_fraction": args.min_area_fraction,
        "max_area_fraction": args.max_area_fraction,
        "min_aspect_ratio": args.min_aspect_ratio,
        "max_aspect_ratio": args.max_aspect_ratio,
        "copy_source": args.copy_source,
        "bbox_xyxy_convention": "[x0, y0, x1, y1] with x1/y1 exclusive",
        "heatmap_values": [0, 255],
        "explanation_template": DEFAULT_EXPLANATION_TEMPLATE,
        "counts": counts,
    }
    with (output_dir / "metadata" / "dataset_info.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved sanity-check dataset to {output_dir}")


if __name__ == "__main__":
    main()

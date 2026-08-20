#!/usr/bin/env python3
"""Build a self-contained, EXIF-normalized sanity-patch test subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}
PATH_FIELDS = (
    "good_image",
    "source_image_path",
    "bad_image",
    "corrupted_image_path",
    "bad_heatmap",
    "heatmap_path",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Line {line_number} is not a JSON object")
            value["_source_metadata_line"] = line_number
            rows.append(value)
    return rows


def resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.extend((path, data_root / str(path).lstrip("/\\")))
    else:
        candidates.append(data_root / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {value!r}; tried: {tried}")


def row_path(row: dict[str, Any], data_root: Path, *keys: str) -> Path:
    errors = []
    for key in keys:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            return resolve_data_path(data_root, value)
        except FileNotFoundError as error:
            errors.append(str(error))
    raise FileNotFoundError("; ".join(errors) or f"Missing path fields {keys}")


def normalized_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def normalized_mask(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("L")


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not value:
        raise ValueError("Sample name became empty after sanitization")
    return value


def relative(path: Path, root: Path) -> str:
    value = path.relative_to(root).as_posix()
    if PurePathValidator.is_unsafe(value):
        raise ValueError(f"Unsafe packaged path: {value}")
    return value


class PurePathValidator:
    @staticmethod
    def is_unsafe(value: str) -> bool:
        path = Path(value)
        return path.is_absolute() or ".." in path.parts


def validate_and_save_sample(
    row: dict[str, Any],
    data_root: Path,
    output_root: Path,
    test_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_index = row.get("source_row_index")
    if not isinstance(source_index, int):
        raise TypeError(f"Invalid source_row_index: {source_index!r}")

    good_path = row_path(row, data_root, "good_image", "source_image_path")
    bad_path = row_path(row, data_root, "bad_image", "corrupted_image_path")
    heatmap_path = row_path(row, data_root, "bad_heatmap", "heatmap_path")

    good = normalized_rgb(good_path)
    bad = normalized_rgb(bad_path)
    heatmap = normalized_mask(heatmap_path)
    if good.size != bad.size or good.size != heatmap.size:
        raise ValueError(
            f"Size mismatch at source row {source_index}: "
            f"good={good.size}, bad={bad.size}, heatmap={heatmap.size}"
        )

    heatmap_array = np.asarray(heatmap)
    values = set(np.unique(heatmap_array).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(
            f"Non-binary heatmap at source row {source_index}: {sorted(values)}"
        )
    mask = heatmap_array == 255
    actual_bbox = mask_bbox(mask)
    expected_bbox = row.get("bbox_xyxy")
    if actual_bbox != expected_bbox:
        raise ValueError(
            f"Heatmap bbox mismatch at source row {source_index}: "
            f"metadata={expected_bbox}, actual={actual_bbox}"
        )

    good_array = np.asarray(good)
    bad_array = np.asarray(bad)
    changed = np.any(good_array != bad_array, axis=2)
    changed_inside = int(np.count_nonzero(changed & mask))
    changed_outside = int(np.count_nonzero(changed & ~mask))
    if changed_inside <= 0 or changed_outside != 0:
        raise ValueError(
            f"Image/heatmap alignment failed at source row {source_index}: "
            f"changed_inside={changed_inside}, changed_outside={changed_outside}"
        )

    color_name = row.get("color_name")
    expected_color = COLORS.get(color_name)
    if expected_color is None:
        raise ValueError(f"Unknown color_name at source row {source_index}: {color_name}")
    patch_pixels = bad_array[mask]
    if not np.all(patch_pixels == np.asarray(expected_color, dtype=np.uint8)):
        raise ValueError(
            f"Bad-image patch color mismatch at source row {source_index}"
        )

    sample_id = safe_name(str(row.get("sample_id") or f"source_{source_index:06d}"))
    basename = f"test_{test_index:02d}_source{source_index:04d}_{sample_id}"
    packaged_good = output_root / "images" / "good" / f"{basename}.png"
    packaged_bad = output_root / "images" / "bad" / f"{basename}.png"
    packaged_heatmap = output_root / "heatmaps" / "test" / f"{basename}.png"
    for path in (packaged_good, packaged_bad, packaged_heatmap):
        path.parent.mkdir(parents=True, exist_ok=True)

    # Saving normalized pixels as PNG removes EXIF Orientation entirely.
    good.save(packaged_good, format="PNG")
    bad.save(packaged_bad, format="PNG")
    heatmap.save(packaged_heatmap, format="PNG")

    packaged = {key: value for key, value in row.items() if not key.startswith("_")}
    packaged.update(
        {
            "split": "test",
            "test_index": test_index,
            "source_metadata_line": row["_source_metadata_line"],
            "good_image": relative(packaged_good, output_root),
            "source_image_path": relative(packaged_good, output_root),
            "bad_image": relative(packaged_bad, output_root),
            "corrupted_image_path": relative(packaged_bad, output_root),
            "bad_heatmap": relative(packaged_heatmap, output_root),
            "heatmap_path": relative(packaged_heatmap, output_root),
        }
    )
    for key in PATH_FIELDS:
        if PurePathValidator.is_unsafe(packaged[key]):
            raise ValueError(f"Absolute or escaping path remained in {key}: {packaged[key]}")

    validation = {
        "test_index": test_index,
        "source_row_index": source_index,
        "sample_id": packaged.get("sample_id"),
        "size": list(good.size),
        "bbox_xyxy": actual_bbox,
        "changed_inside_mask": changed_inside,
        "changed_outside_mask": changed_outside,
        "good_source": str(good_path),
        "bad_source": str(bad_path),
        "heatmap_source": str(heatmap_path),
    }
    return packaged, validation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def validate_package(output_root: Path, rows: list[dict[str, Any]]) -> None:
    expected_files = set()
    for row in rows:
        for key in PATH_FIELDS:
            value = row[key]
            if PurePathValidator.is_unsafe(value):
                raise ValueError(f"Unsafe path in packaged metadata: {key}={value}")
            path = output_root / value
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_files.add(path.resolve())

    for path in output_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink found in package: {path}")
    if len(expected_files) != len(rows) * 3:
        raise ValueError(
            f"Expected {len(rows) * 3} unique image files, found {len(expected_files)}"
        )


def create_archive(output_root: Path, archive: Path) -> tuple[str, int]:
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(output_root, arcname=output_root.name, recursive=True)

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Link found in archive: {member.name}")

    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    sidecar.write_text(f"{sha256}  {archive.name}\n", encoding="ascii")
    return sha256, len(members)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-first-index", type=int)
    parser.add_argument("--expected-last-index", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    metadata = (
        args.metadata_path.expanduser().resolve()
        if args.metadata_path
        else data_root / "metadata" / "train.jsonl"
    )
    output_root = args.output_root.expanduser().resolve()
    archive = (
        args.archive.expanduser().resolve()
        if args.archive
        else output_root.parent / f"{output_root.name}.tar.gz"
    )
    if args.last_n <= 0:
        raise ValueError("--last-n must be positive")
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")

    rows = read_jsonl(metadata)
    if args.expected_total is not None and len(rows) != args.expected_total:
        raise ValueError(
            f"Expected {args.expected_total} rows in {metadata}, found {len(rows)}"
        )
    if len(rows) < args.last_n:
        raise ValueError(f"Only {len(rows)} rows are available")
    selected = rows[-args.last_n :]
    indexes = [row.get("source_row_index") for row in selected]
    if args.expected_first_index is not None and indexes[0] != args.expected_first_index:
        raise ValueError(f"Unexpected first source index: {indexes[0]}")
    if args.expected_last_index is not None and indexes[-1] != args.expected_last_index:
        raise ValueError(f"Unexpected last source index: {indexes[-1]}")
    if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
        raise ValueError(f"Source indexes are not contiguous: {indexes}")

    packaged_rows = []
    validations = []
    try:
        output_root.mkdir(parents=True)
        for test_index, row in enumerate(selected):
            packaged, validation = validate_and_save_sample(
                row, data_root, output_root, test_index
            )
            packaged_rows.append(packaged)
            validations.append(validation)
            print(
                f"validated test={test_index:02d} "
                f"source={validation['source_row_index']} "
                f"size={validation['size']} bbox={validation['bbox_xyxy']}"
            )

        write_jsonl(output_root / "metadata" / "test.jsonl", packaged_rows)
        info = {
            "source_data_root": str(data_root),
            "source_metadata": str(metadata),
            "selection": "last_n",
            "count": len(packaged_rows),
            "source_row_indexes": indexes,
            "split": "test",
            "images_exif_transposed": True,
            "images_saved_as_png": True,
            "self_contained": True,
            "relative_paths_only": True,
            "symlinks": False,
        }
        (output_root / "metadata" / "dataset_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_root / "metadata" / "validation.json").write_text(
            json.dumps(validations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        validate_package(output_root, packaged_rows)
        sha256, member_count = create_archive(output_root, archive)
    except Exception:
        # Keep the staged directory for diagnosis, but never produce a trusted archive.
        archive.unlink(missing_ok=True)
        archive.with_suffix(archive.suffix + ".sha256").unlink(missing_ok=True)
        raise

    print(f"package: {output_root}")
    print(f"archive: {archive}")
    print(f"archive_members: {member_count}")
    print(f"sha256: {sha256}")
    print("validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

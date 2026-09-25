#!/usr/bin/env python3
"""Build a portable perspective dataset with visual inspection supervision.

The source metadata must already contain the visually redesigned checks and
their ``judge`` labels (``visual_separate_focus_shared_controls_v3``).  Each
group makes one multimodal API request containing GOOD and BAD.  The response
regenerates all per-check inspections and all three conclusions, while the
existing deterministic check/global labels remain the source of truth.

The output directory is self-contained: train/test metadata and every image
referenced by ``good_image``, ``bad_image`` and ``bad_heatmap`` are copied
under the same relative paths.  Progress JSONL files make API work resumable.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


REASON_KEYS = ("good_reason", "bad_reason", "pair_reason")
ASSET_KEYS = ("good_image", "bad_image", "bad_heatmap")
DATA_VERSION = "visual_inspection_tokenwise_v1"
SOURCE_JUDGE_VERSION = "visual_separate_focus_shared_controls_v3"

PROMPT = """You are creating grounded supervision for perspective analysis.
Image A is GOOD and Image B is BAD; they depict the same scene. The neutral
checks and their trusted labels are supplied below. Do not create, remove,
merge, reorder, or relabel checks.

For every check, write one concise Inspection that describes concrete visible
geometric evidence in the appropriate image. Explain how the named structures
relate (directions, convergence, parallelism, coplanarity, depth scaling, or
boundary alignment). A label of 1 means the expected relationship is visibly
satisfied; a label of 0 means it is visibly violated. Express the evidence,
not a bare verdict, and do not use the literal words SATISFIED, VIOLATED,
CONSISTENT, INCONSISTENT, correct, or incorrect.

- good_reason: inspect only Image A.
- bad_reason: inspect only Image B.
- pair_reason: compare Image A with Image B explicitly for every check.

Also regenerate one concise Conclusion for each reason. It must synthesize
the inspections, identify the principal structure/region, and agree with the
trusted global label. The pair conclusion must explicitly contrast GOOD and
BAD. Do not mention labels, check numbers, annotation, editing instructions,
or image quality.

Return JSON only, exactly in this shape, with one inspection per supplied
check and in the same order:
{{
  "good_reason": {{"inspections": ["..."], "conclusion": "..."}},
  "bad_reason": {{"inspections": ["..."], "conclusion": "..."}},
  "pair_reason": {{"inspections": ["..."], "conclusion": "..."}}
}}

TRUSTED CHECKS AND LABELS:
{payload}
"""


def key_values() -> list[str]:
    raw = os.getenv("EVAL_KEY_ENVS", "")
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        names = [f"KEY_{index}" for index in range(40)]
    values = [os.environ[name] for name in names if os.environ.get(name)]
    if not values:
        raise SystemExit("Export API keys and EVAL_KEY_ENVS before running")
    return values


def response_text(response: Any) -> str:
    value = getattr(response, "output_text", None)
    if value:
        return str(value).strip()
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def parse_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("API response must be a JSON object")
    return value


def image_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _section(pattern: str, text: str, name: str) -> str:
    match = re.search(pattern, text, flags=re.I | re.M | re.S)
    if not match:
        raise ValueError(f"Cannot parse {name} section")
    return match.group(1).strip()


def parse_reason(value: Any) -> dict[str, Any]:
    """Parse the stable Scene/Structures/Checks/Conclusion representation."""
    if isinstance(value, dict):
        checks = []
        for check in value.get("checks", []):
            checks.append({
                "elements": [str(item).strip() for item in check.get("elements", [])],
                "expected_relationship": str(
                    check.get("expected_relationship", "")
                ).strip(),
            })
        result = {
            "scene": str(value.get("scene", "")).strip(),
            "structures": [str(item).strip() for item in value.get("structures", [])],
            "checks": checks,
            "conclusion": str(value.get("conclusion", "")).strip(),
        }
        validate_reason(result)
        return result

    text = str(value or "").strip()
    scene = _section(
        r"^\s*Scene\s*:\s*(.*?)^\s*Structures\s*:", text, "Scene"
    )
    structures_block = _section(
        r"^\s*Structures\s*:\s*(.*?)^\s*Checks\s*:", text, "Structures"
    )
    conclusion_match = re.search(
        r"^\s*Conclusion\s*:\s*(.*)\Z", text, flags=re.I | re.M | re.S
    )
    if not conclusion_match:
        raise ValueError("Cannot parse Conclusion section")
    conclusion = conclusion_match.group(1).strip()
    checks_start = re.search(r"^\s*Checks\s*:\s*$", text, flags=re.I | re.M)
    if not checks_start:
        raise ValueError("Cannot parse Checks section")
    checks_text = text[checks_start.end():conclusion_match.start()]
    starts = list(re.finditer(r"^\s*Check\s+\d+\s*:\s*$", checks_text, flags=re.I | re.M))
    checks: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(checks_text)
        block = checks_text[start.end():end]
        elements_block = _section(
            r"^\s*Elements\s*:\s*(.*?)^\s*Expected relationship\s*:",
            block,
            "Elements",
        )
        expected = _section(
            r"^\s*Expected relationship\s*:\s*(.*?)(?:^\s*Inspection\s*:|\Z)",
            block,
            "Expected relationship",
        )
        elements = [
            line.strip()[2:].strip()
            for line in elements_block.splitlines()
            if line.strip().startswith("- ")
        ]
        checks.append({"elements": elements, "expected_relationship": expected})
    result = {
        "scene": scene,
        "structures": [
            line.strip()[2:].strip()
            for line in structures_block.splitlines()
            if line.strip().startswith("- ")
        ],
        "checks": checks,
        "conclusion": conclusion,
    }
    validate_reason(result)
    return result


def validate_reason(reason: dict[str, Any]) -> None:
    if not reason.get("scene") or not reason.get("structures"):
        raise ValueError("Reason has an empty scene or structures list")
    checks = reason.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("Reason has no checks")
    for check in checks:
        if not check.get("elements") or not check.get("expected_relationship"):
            raise ValueError("Reason has a malformed check")
    if not reason.get("conclusion"):
        raise ValueError("Reason has an empty conclusion")


def trusted_payload(row: dict[str, Any]) -> dict[str, Any]:
    judge = row.get("judge") or {}
    if judge.get("version") != SOURCE_JUDGE_VERSION:
        raise ValueError(
            f"Expected judge version {SOURCE_JUDGE_VERSION}, got {judge.get('version')!r}"
        )
    check_labels = judge.get("checks") or {}
    global_labels = judge.get("global") or {}
    output: dict[str, Any] = {}
    for reason_key in REASON_KEYS:
        reason = parse_reason(row.get(reason_key))
        labels = check_labels.get(reason_key)
        if not isinstance(labels, list) or len(labels) != len(reason["checks"]):
            raise ValueError(f"{reason_key} check/label count mismatch")
        if any(type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError(f"{reason_key} contains a non-binary label")
        quality = "good" if reason_key == "good_reason" else "bad"
        global_label = global_labels.get(quality)
        if type(global_label) is not int or global_label not in (0, 1):
            raise ValueError(f"{reason_key} has an invalid global label")
        output[reason_key] = {
            "scene": reason["scene"],
            "structures": reason["structures"],
            "checks": [
                {**check, "trusted_label": int(label)}
                for check, label in zip(reason["checks"], labels)
            ],
            "trusted_global_label": int(global_label),
        }
    return output


def parse_generated(raw: str, expected: dict[str, Any]) -> dict[str, Any]:
    value = parse_json(raw)
    if set(value) != set(REASON_KEYS):
        raise ValueError(f"Response must contain exactly {REASON_KEYS}")
    output: dict[str, Any] = {}
    for reason_key in REASON_KEYS:
        item = value.get(reason_key)
        if not isinstance(item, dict):
            raise ValueError(f"{reason_key} must be an object")
        inspections = item.get("inspections")
        conclusion = item.get("conclusion")
        expected_count = len(expected[reason_key]["checks"])
        if not isinstance(inspections, list) or len(inspections) != expected_count:
            raise ValueError(
                f"{reason_key}: expected {expected_count} inspections, "
                f"got {len(inspections) if isinstance(inspections, list) else 'non-list'}"
            )
        normalized = [str(text).strip() for text in inspections]
        if any(len(text) < 12 for text in normalized):
            raise ValueError(f"{reason_key} contains an empty/short inspection")
        if not isinstance(conclusion, str) or len(conclusion.strip()) < 12:
            raise ValueError(f"{reason_key} contains an empty/short conclusion")
        output[reason_key] = {
            "inspections": normalized,
            "conclusion": conclusion.strip(),
        }
    return output


def build_supervision(
    expected: dict[str, Any], generated: dict[str, Any], model: str
) -> dict[str, Any]:
    result: dict[str, Any] = {"version": DATA_VERSION, "model": model}
    for reason_key in REASON_KEYS:
        source = expected[reason_key]
        generated_reason = generated[reason_key]
        result[reason_key] = {
            "scene": source["scene"],
            "structures": source["structures"],
            "checks": [
                {
                    "elements": check["elements"],
                    "expected_relationship": check["expected_relationship"],
                    "inspection": inspection,
                    "judgment": int(check["trusted_label"]),
                }
                for check, inspection in zip(
                    source["checks"], generated_reason["inspections"]
                )
            ],
            "conclusion": generated_reason["conclusion"],
            "global_judgment": int(source["trusted_global_label"]),
        }
    return result


def render_reason(reason: dict[str, Any]) -> str:
    lines = ["Scene:", reason["scene"], "", "Structures:"]
    lines.extend(f"- {item}" for item in reason["structures"])
    lines.extend(["", "Checks:"])
    for index, check in enumerate(reason["checks"], start=1):
        lines.extend(["", f"Check {index}:", "Elements:"])
        lines.extend(f"- {item}" for item in check["elements"])
        lines.extend([
            "Expected relationship:",
            check["expected_relationship"],
            "Inspection:",
            check["inspection"],
            "Check judgment:",
            "SATISFIED" if int(check["judgment"]) else "VIOLATED",
        ])
    lines.extend([
        "",
        "Conclusion:",
        reason["conclusion"],
        "",
        "Global judgment:",
        "CONSISTENT" if int(reason["global_judgment"]) else "INCONSISTENT",
    ])
    return "\n".join(lines).strip()


def generate_row(
    row: dict[str, Any], key: str, args: argparse.Namespace
) -> dict[str, Any]:
    expected = trusted_payload(row)
    prompt = PROMPT.format(payload=json.dumps(expected, ensure_ascii=False))
    image_paths = tuple(
        args.image_root / row[name] for name in ("good_image", "bad_image")
    )
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for label, path in zip(("A (GOOD)", "B (BAD)"), image_paths):
        content.append({"type": "input_text", "text": f"Image {label}:"})
        content.append({"type": "input_image", "image_url": image_data_url(path)})
    client = OpenAI(api_key=key, base_url=args.base_url, timeout=args.api_timeout)
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = client.responses.create(
                model=args.model,
                input=[{"role": "user", "content": content}],
            )
            generated = parse_generated(response_text(response), expected)
            supervision = build_supervision(expected, generated, args.model)
            output = dict(row)
            output["tokenwise_reason"] = supervision
            for reason_key in REASON_KEYS:
                output[reason_key] = render_reason(supervision[reason_key])
            output["source_conclusions"] = {
                reason_key: parse_reason(row[reason_key])["conclusion"]
                for reason_key in REASON_KEYS
            }
            judge = dict(output.get("judge") or {})
            judge["inspection_version"] = DATA_VERSION
            judge["inspection_model"] = args.model
            output["judge"] = judge
            return output
        except Exception as exc:
            last = exc
            if attempt < 3:
                time.sleep(min(20, 2**attempt))
    raise RuntimeError(f"inspection/conclusion generation failed: {last}")


def row_id(row: dict[str, Any], split: str, index: int) -> str:
    return str(row.get("group_id") or row.get("stage1_image_id") or f"{split}:{index}")


def is_complete(row: dict[str, Any], model: str) -> bool:
    supervision = row.get("tokenwise_reason") or {}
    if supervision.get("version") != DATA_VERSION or supervision.get("model") != model:
        return False
    try:
        for reason_key in REASON_KEYS:
            reason = supervision[reason_key]
            validate_reason(reason)
            if len(reason["checks"]) not in (2, 3):
                return False
            if any(
                type(check.get("judgment")) is not int
                or check["judgment"] not in (0, 1)
                or not str(check.get("inspection", "")).strip()
                for check in reason["checks"]
            ):
                return False
            if type(reason.get("global_judgment")) is not int:
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _safe_relative(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Asset path must be relative and contained: {value!r}")
    return path


def copy_assets(
    rows: list[dict[str, Any]], image_root: Path, output_dir: Path, workers: int
) -> None:
    relative_paths = sorted({
        _safe_relative(row[key]) for row in rows for key in ASSET_KEYS
    })

    def copy_one(relative: Path) -> None:
        source = image_root / relative
        destination = output_dir / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return
        temporary = destination.with_suffix(destination.suffix + ".copying")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(relative_paths)))) as pool:
        futures = [pool.submit(copy_one, path) for path in relative_paths]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 250 == 0 or completed == len(futures):
                print(f"copied assets {completed}/{len(futures)}", flush=True)


def process_split(
    source: Path,
    destination: Path,
    split: str,
    keys: list[str],
    args: argparse.Namespace,
) -> None:
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_assets(rows, args.image_root, args.output_dir, args.copy_workers)

    ids = [row_id(row, split, index) for index, row in enumerate(rows)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate group IDs in {source}")
    done: dict[str, dict[str, Any]] = {}
    progress = destination.with_suffix(".progress.jsonl")
    for cache in (destination, progress):
        if not cache.exists():
            continue
        for line in cache.read_text(encoding="utf-8").splitlines():
            try:
                cached = json.loads(line)
                if is_complete(cached, args.model):
                    done[row_id(cached, split, -1)] = cached
            except (json.JSONDecodeError, OSError):
                continue

    todo = [(index, row) for index, row in enumerate(rows) if ids[index] not in done]
    print(
        f"{split}: total={len(rows)} cached={len(rows)-len(todo)} todo={len(todo)}",
        flush=True,
    )
    errors: list[str] = []
    error_log = destination.with_suffix(".errors.jsonl")
    if todo:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(todo))) as pool:
            futures = {
                pool.submit(generate_row, row, keys[index % len(keys)], args): index
                for index, row in todo
            }
            with progress.open("a", encoding="utf-8") as writer:
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        output = future.result()
                        done[ids[index]] = output
                        writer.write(json.dumps(output, ensure_ascii=False) + "\n")
                        writer.flush()
                        print(f"{split} {index + 1}/{len(rows)} {ids[index]}", flush=True)
                    except Exception as exc:
                        message = f"{ids[index]}: {exc}"
                        errors.append(message)
                        with error_log.open("a", encoding="utf-8") as error_writer:
                            error_writer.write(json.dumps({
                                "split": split,
                                "index": index,
                                "group_id": ids[index],
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "time": time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                                ),
                            }, ensure_ascii=False) + "\n")
                        print(f"ERROR {split} {index + 1}/{len(rows)} {message}", flush=True)
    if errors:
        raise RuntimeError(
            f"{len(errors)} rows failed; rerun to resume. "
            f"Errors: {error_log}. First: {errors[0]}"
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(done[ident], ensure_ascii=False) + "\n" for ident in ids),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    progress.unlink(missing_ok=True)


def default_output_dir(data_root: Path) -> Path:
    name = data_root.name
    if name.endswith("_judge_new"):
        name = name[:-len("_judge_new")] + "_judge_inspect_v1"
    else:
        name += "_inspect_v1"
    return data_root.with_name(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://svip.xuedingmao.com/v1"),
    )
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--copy-workers", type=int, default=16)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.image_root = args.image_root.resolve()
    args.output_dir = (args.output_dir or default_output_dir(args.data_root)).resolve()
    if args.output_dir in (args.data_root, args.image_root):
        parser.error("--output-dir must differ from data and image roots")
    if args.workers < 1 or args.copy_workers < 1 or args.api_timeout <= 0:
        parser.error("workers/copy-workers/api-timeout must be positive")
    keys = key_values()
    for split in ("train", "test"):
        process_split(
            args.data_root / f"{split}.jsonl",
            args.output_dir / f"{split}.jsonl",
            split,
            keys,
            args,
        )
    print(f"WROTE={args.output_dir}")


if __name__ == "__main__":
    main()

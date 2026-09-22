#!/usr/bin/env python3
"""Create GPT-labelled check conclusions for the perspective metadata.

The API receives text only.  For each group all checks from good_reason,
bad_reason and pair_reason are sent in one request.  The returned label is
the *conclusion polarity* of a check (consistent/correct=1,
inconsistent/incorrect=0); it is deliberately not a judgement of whether
the annotation itself is factually correct.  The image-level label is
deterministic: GOOD=1 and BAD=0.

The output is resumable and contains copies of train.jsonl/test.jsonl with a
small ``judge`` object added to every row.  Images remain in the original
data root, so the output metadata can be used directly with BAGEL's
``BAGEL_REASON_HEATMAP_METADATA_PATH`` override.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

from prepare_perspective_strong_long_reason import format_reason


REASON_KEYS = ("good_reason", "bad_reason", "pair_reason")
JUDGE_VERSION = "check_polarity_v2"
PROMPT = """You label the polarity of perspective-analysis checks.
This is NOT a factual review of whether the check is right, and NOT a review
of the image. Read only the supplied check text and classify what conclusion
the check reaches:
- 1 / correct: it says the checked relationship is coherent, aligned,
  preserved, or otherwise has no violation.
- 0 / incorrect: it says the relationship is inconsistent, mismatched,
  violated, skewed, or otherwise detects a perspective error.
Return JSON only, with this exact shape:
{{"checks":{{"good_reason":[1],"bad_reason":[0],"pair_reason":[0]}},"global":{{"good":1,"bad":0}}}}
Each list must have one integer per supplied check, in the same order. Do not
judge annotation quality. Global labels are deterministic and must be good=1,
bad=0.

INPUT CHECKS:
{payload}
"""


def key_names() -> list[str]:
    raw = os.getenv("EVAL_KEY_ENVS", "")
    names = [x.strip() for x in raw.split(",") if x.strip()]
    if not names:
        names = [f"KEY_{i}" for i in list(range(10)) + list(range(20, 30))]
    values = [os.environ[name] for name in names if os.environ.get(name)]
    if not values:
        raise SystemExit("Export API keys first (KEY_0..KEY_9 / KEY_20..KEY_29)")
    return values


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(str(value))
    return "\n".join(parts).strip()


def as_checks(value: Any) -> list[str]:
    if isinstance(value, dict):
        checks = value.get("checks", [])
        if isinstance(checks, list):
            out = []
            for item in checks:
                if isinstance(item, dict):
                    parts = [
                        str(item.get("elements", "")),
                        str(item.get("expected_relationship", "")),
                        str(item.get("inspection", "")),
                    ]
                    out.append("\n".join(x for x in parts if x and x != "None"))
                else:
                    out.append(str(item))
            return out
    text = str(value or "").strip()
    if not text:
        return []
    # Short legacy reasons have no numbered checks; treating the whole reason
    # as one check keeps train/test metadata compatible.
    matches = list(re.finditer(r"(?:^|\n)\s*Check\s+(\d+)\s*:\s*", text, re.I))
    if not matches:
        return [text]
    end = re.search(r"(?:^|\n)\s*Conclusion\s*:", text, re.I)
    limit = end.start() if end else len(text)
    return [text[m.end() : (matches[i + 1].start() if i + 1 < len(matches) else limit)].strip() for i, m in enumerate(matches)]


def build_payload(row: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, int]]:
    checks = {key: as_checks(row.get(key)) for key in REASON_KEYS}
    payload = {
        key: [{"check_id": i + 1, "text": text} for i, text in enumerate(values)]
        for key, values in checks.items()
    }
    return payload, {"good": 1, "bad": 0}


def parse_result(raw: str, expected: dict[str, list[str]]) -> dict[str, Any]:
    cleaned = raw.strip().strip("`")
    if cleaned.startswith("json\n"):
        cleaned = cleaned[5:]
    value = json.loads(cleaned)
    source = value.get("checks", value.get("check_labels", {}))
    output: dict[str, list[int]] = {}
    for key, items in expected.items():
        got = source.get(key, []) if isinstance(source, dict) else []
        if isinstance(got, dict):
            got = [got.get(str(i + 1), got.get(i + 1)) for i in range(len(items))]
        labels: list[int] = []
        for item in got:
            if isinstance(item, dict):
                item = item.get("label", item.get("judgment", item.get("value")))
            if isinstance(item, bool):
                labels.append(int(item))
            elif str(item).strip().lower() in {"1", "true", "correct", "consistent", "coherent"}:
                labels.append(1)
            elif str(item).strip().lower() in {"0", "false", "incorrect", "inconsistent", "violation"}:
                labels.append(0)
            else:
                raise ValueError(f"Unknown check label: {item!r}")
        if len(labels) != len(items):
            raise ValueError(f"{key}: expected {len(items)} labels, got {len(labels)}")
        output[key] = labels
    return {"checks": output, "global": {"good": 1, "bad": 0}}


def judge(row: dict[str, Any], key: str, base_url: str, model: str) -> dict[str, Any]:
    payload, global_labels = build_payload(row)
    expected = {key: [x["text"] for x in values] for key, values in payload.items()}
    prompt = PROMPT.format(payload=json.dumps(payload, ensure_ascii=False))
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = OpenAI(api_key=key, base_url=base_url).responses.create(
                model=model, input=prompt
            )
            result = parse_result(response_text(response), expected)
            result["global"] = global_labels
            result["version"] = JUDGE_VERSION
            result["model"] = model
            return result
        except Exception as exc:
            last = exc
            time.sleep(min(20, 2**attempt))
    raise last  # type: ignore[misc]


def row_id(row: dict[str, Any], split: str, index: int) -> str:
    return str(row.get("group_id") or row.get("stage1_image_id") or f"{split}:{index}")


def prepared_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    for key in ALL_REASON_KEYS:
        if isinstance(row.get(key), dict):
            row[key] = format_reason(row[key])
        if not str(row.get(key) or "").strip():
            raise ValueError(f"Missing {key} in group {row.get('group_id')}")
    row["good_score"] = 1.0
    row["bad_score"] = float(row.get("bad_score", row.get("bad_score_target", 0.0)))
    return row


def process_split(source: Path, destination: Path, split: str, keys: list[str], args: argparse.Namespace) -> None:
    rows = [prepared_row(json.loads(line)) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    progress = destination.with_suffix(".progress.jsonl")
    for cache in (destination, progress):
        if not cache.exists():
            continue
        for line in cache.read_text(encoding="utf-8").splitlines():
            try:
                old = json.loads(line)
                if old.get("judge", {}).get("version") == JUDGE_VERSION:
                    done[row_id(old, split, -1)] = old
            except Exception:
                continue
    todo = [(i, row) for i, row in enumerate(rows) if row_id(row, split, i) not in done]
    print(f"{split}: total={len(rows)} cached={len(rows)-len(todo)} todo={len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(todo)))) as pool:
        futures = {
            pool.submit(judge, row, keys[i % len(keys)], args.base_url, args.model): (i, row)
            for i, row in todo
        }
        errors = []
        with progress.open("a", encoding="utf-8") as writer:
            for future in as_completed(futures):
                i, row = futures[future]
                try:
                    row = dict(row)
                    row["judge"] = future.result()
                    done[row_id(row, split, i)] = row
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    writer.flush()
                    print(f"{split} {i + 1}/{len(rows)} {row_id(row, split, i)}", flush=True)
                except Exception as exc:
                    errors.append(f"{row_id(row, split, i)}: {exc}")
        if errors:
            raise RuntimeError(f"{len(errors)} API rows failed; rerun to resume. First: {errors[0]}")
    ordered = [done[row_id(row, split, i)] for i, row in enumerate(rows)]
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")
    os.replace(tmp, destination)
    progress.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://svip.xuedingmao.com/v1"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_dir = (args.output_dir or args.data_root.parent / f"{args.data_root.name}_judge").resolve()
    keys = key_names()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for split in ("train", "test"):
        process_split(args.data_root / f"{split}.jsonl", args.output_dir / f"{split}.jsonl", split, keys, args)
    print(f"WROTE={args.output_dir}")


if __name__ == "__main__":
    main()

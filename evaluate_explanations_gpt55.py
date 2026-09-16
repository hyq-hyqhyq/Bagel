"""Judge generated perspective explanations against paired GT explanations.

The judge receives five pairs per request and returns one correctness verdict
per pair. Keys are read from KEY_20..KEY_29 by default and are never written.
"""
from __future__ import annotations

import argparse, json, os, tarfile, time
from pathlib import Path
from typing import Any

from openai import OpenAI

JUDGE_PROMPT = """You are evaluating perspective/projection explanations.
For each item, compare the MODEL explanation with the corresponding GROUND-TRUTH explanation.
Judge whether the MODEL explanation correctly identifies the same visible perspective issue (or correctly says there is no clear issue), focusing on the affected structure and the violated or preserved projection relationship. Paraphrasing is allowed; do not require identical wording. Mark incorrect if it contradicts the ground truth, misses the central issue, or invents an unrelated issue.

Return ONLY valid JSON in this exact shape:
{{"results":[{{"id":"...","correct":true,"verdict":"correct"}},{{"id":"...","correct":false,"verdict":"incorrect"}}]}}
Include exactly one result for every item and preserve each id.

Items:
{items}
"""


def keys() -> list[str]:
    names = [x.strip() for x in os.getenv("EVAL_KEY_ENVS", "").split(",") if x.strip()]
    if not names: names = [f"KEY_{i}" for i in range(20, 30)]
    out = [os.environ[n] for n in names if os.environ.get(n)]
    if not out: raise SystemExit("No evaluator keys found; export KEY_20..KEY_29 or EVAL_KEY_ENVS")
    return out


def load_gt(archive: Path) -> dict[str, dict[str, Any]]:
    with tarfile.open(archive, "r:gz") as tf:
        names = sorted(n for n in tf.getnames() if "/metadata/" in n and n.endswith(".jsonl"))
        if not names: raise SystemExit("metadata jsonl not found in archive")
        rows = []
        for name in names:
            raw = tf.extractfile(name)
            assert raw is not None
            rows.extend(json.loads(x) for x in raw.read().decode("utf8").splitlines() if x.strip())
    out = {}
    for row in rows:
        gid = str(row.get("group_id"))
        for label in ("good", "bad"):
            reason = row.get(f"{label}_reason") or {}
            conclusion = reason.get("conclusion", "") if isinstance(reason, dict) else ""
            out[f"{gid}::{label}"] = {"group_id": gid, "label": label, "gt_explanation": str(conclusion).strip()}
    return out


def response_text(resp: Any) -> str:
    if getattr(resp, "output_text", None): return str(resp.output_text)
    return ""


def call_batch(batch: list[dict[str, Any]], api_base: str, model: str, key_list: list[str]) -> list[dict[str, Any]]:
    items = json.dumps([{"id": x["id"], "MODEL explanation": x["explanation"], "GROUND-TRUTH explanation": x["gt_explanation"]} for x in batch], ensure_ascii=False)
    prompt = JUDGE_PROMPT.format(items=items)
    last = None
    for attempt in range(max(4, len(key_list))):
        try:
            client = OpenAI(api_key=key_list[attempt % len(key_list)], base_url=api_base)
            resp = client.responses.create(model=model, input=prompt)
            text = response_text(resp).strip()
            if text.startswith("```"):
                text = text.strip("`").replace("json\n", "", 1).strip()
            data = json.loads(text)
            result = data.get("results")
            if not isinstance(result, list) or len(result) != len(batch): raise ValueError("judge returned wrong result count")
            return result
        except Exception as exc:
            last = exc; time.sleep(min(30, 2 ** min(attempt, 4)))
    raise last


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://svip.xuedingmao.com/v1"))
    p.add_argument("--model", default="gpt-5.5")
    a = p.parse_args()
    if a.batch_size != 5: raise SystemExit("This evaluator is intentionally fixed to batches of 5; use --batch-size 5")
    gt = load_gt(a.archive.resolve()); rows=[]
    for line in a.predictions.read_text(encoding="utf8").splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get("status") != "ok" or not r.get("explanation"): continue
        key=f"{r.get('group_id')}::{r.get('label')}"
        if key not in gt: continue
        rows.append({"id": key, "group_id": r.get("group_id"), "label": r.get("label"), "explanation": r["explanation"], **gt[key]})
    rows_by_id={r["id"]:r for r in rows}; existing={}
    if a.output.exists():
        for line in a.output.read_text(encoding="utf8").splitlines():
            try:
                r=json.loads(line); existing[r["id"]]=r
            except Exception: pass
    todo=[r for r in rows if r["id"] not in existing]; key_list=keys()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a", encoding="utf8") as fh:
        for start in range(0, len(todo), 5):
            batch=todo[start:start+5]
            if len(batch) < 5: print(f"final partial batch: {len(batch)} items", flush=True)
            judged=call_batch(batch, a.base_url, a.model, key_list)
            by_id={str(x.get("id")):x for x in judged}
            for item in batch:
                j=by_id.get(item["id"], {})
                out={"id":item["id"],"group_id":item["group_id"],"label":item["label"],"explanation":item["explanation"],"gt_explanation":item["gt_explanation"],"correct":bool(j.get("correct")),"verdict":j.get("verdict"),"judge_model":a.model}
                fh.write(json.dumps(out, ensure_ascii=False)+"\n"); fh.flush(); print(item["id"], out["verdict"], flush=True)
    all_rows=list(existing.values()) + [json.loads(x) for x in a.output.read_text(encoding="utf8").splitlines() if x.strip() and json.loads(x)["id"] not in existing]
    correct=sum(bool(x.get("correct")) for x in all_rows); print(json.dumps({"total":len(all_rows),"correct":correct,"incorrect":len(all_rows)-correct,"accuracy":correct/len(all_rows) if all_rows else None}, ensure_ascii=False))


if __name__ == "__main__": main()

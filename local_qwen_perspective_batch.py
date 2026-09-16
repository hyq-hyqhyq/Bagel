"""Local two-stage Qwen replacement for the perspective API batch.

Stage 1: Qwen3-VL-8B-Instruct produces one perspective explanation.
Stage 2: Qwen-Image-Edit-2511 receives the original image plus that
explanation and its output is normalized to a strict black/white mask.
Run disjoint shards with --num-shards/--shard-index on separate GPUs.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bpipe.perspective_edit_planner.planner import PerspectiveEditPlanner
from bpipe.perspective_data_stage2.api_client import QwenImageEditClient

STAGE1 = """You are a visual reasoning expert specialized in perspective and projection consistency analysis.
Analyze the provided image only from the perspective of spatial structure and projection relationships.
Determine whether the image contains any visible perspective or projection inconsistency and explain the visual evidence.
Focus on structural edges and boundaries, planar surfaces, object orientations, depth relationships, convergence directions, and perspective consistency between related structures.
If an inconsistency exists, explain which structure is affected, which surrounding structures provide references, what relationship should hold, and how the observed structure violates it.
Do not discuss image quality, aesthetics, lighting, colors, textures, semantic correctness, or generation artifacts.
Avoid vague statements. If no clear inconsistency exists, state that no clear perspective error is observed.
Return only one concise explanation paragraph."""

STAGE2 = """The image above is the original input image.
Generate a binary localization mask for the described perspective or projection inconsistency.
Perspective analysis:
{explanation}
Output a black-and-white binary mask aligned with the original image.
Use pure black background and pure white only for the actual local error region. Do not highlight unrelated objects, background, textures, colors, lighting, or image quality. Do not add grayscale, transparency, text, labels, arrows, boxes, or outlines. If no clear perspective inconsistency exists, output an entirely black mask."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--vl-model", type=Path, default=Path("/data/bagel/models/Qwen3-VL-8B-Instruct"))
    p.add_argument("--edit-model", type=Path, default=Path("/data/bagel/repo/agent/bpipe/models/Qwen-Image-Edit-2511"))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def binaryize(src: Path, dst: Path, size: tuple[int, int]) -> None:
    image = Image.open(src).convert("L").resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(image)
    Image.fromarray(((arr >= 128).astype(np.uint8) * 255), mode="L").save(dst)


def main() -> None:
    a = parse_args()
    if not (0 <= a.shard_index < a.num_shards): raise SystemExit("invalid shard index")
    root, out = a.data_root.resolve(), a.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = root / "metadata" / "test.jsonl"
    rows = [json.loads(x) for x in manifest.read_text(encoding="utf8").splitlines() if x.strip()]
    items = [(row, label, root / row[f"{label}_image"]) for row in rows for label in ("good", "bad")]
    items = [x for i, x in enumerate(items) if i % a.num_shards == a.shard_index]
    print(f"shard={a.shard_index}/{a.num_shards} items={len(items)}", flush=True)
    planner = PerspectiveEditPlanner.from_pretrained(a.vl_model, device_map="auto", dtype="bfloat16", local_files_only=True, max_new_tokens=512, enable_thinking=False)
    edit = QwenImageEditClient.from_config({"model_path": str(a.edit_model), "device": "cuda", "device_map": "", "dtype": "bfloat16", "local_files_only": True, "max_input_side": 1024, "generation_image_size": "request", "output_image_size": "auto", "preserve_input_size": True, "num_inference_steps": a.steps, "true_cfg_scale": a.cfg, "negative_prompt": " ", "enable_model_cpu_offload": True, "enable_attention_slicing": True, "enable_vae_tiling": True, "request_timeout": 1200})
    log = out / f"results_shard{a.shard_index}.jsonl"
    done = {}
    if a.resume and log.exists():
        for line in log.read_text(encoding="utf8").splitlines():
            try:
                r=json.loads(line); done[(r["group_id"],r["label"])] = r
            except Exception: pass
    with log.open("a", encoding="utf8") as fh:
        for n, (row, label, image) in enumerate(items, 1):
            key=(row["group_id"],label); rec={"group_id":row["group_id"],"label":label,"category":row.get("intended_category"),"image":str(image)}
            if a.resume and key in done and done[key].get("status")=="ok": continue
            try:
                explanation = planner._generate(image, STAGE1)
                rec["explanation"] = explanation.strip()
                raw_dir = out / "raw" / label; mask_dir = out / "masks" / label; raw_dir.mkdir(parents=True, exist_ok=True); mask_dir.mkdir(parents=True, exist_ok=True)
                raw = raw_dir / f"{row['group_id']}.png"; mask = mask_dir / f"{row['group_id']}.png"
                edit.edit(image, STAGE2.format(explanation=rec["explanation"]), raw)
                with Image.open(image) as src: size=src.size
                binaryize(raw, mask, size); rec.update(raw_image=str(raw), mask=str(mask), status="ok")
            except Exception as exc:
                rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
            fh.write(json.dumps(rec, ensure_ascii=False)+"\n"); fh.flush(); print(f"[{n}/{len(items)}] {label} {row['group_id']} {rec['status']}", flush=True)


if __name__ == "__main__": main()

"""Two-stage perspective explanation + binary localization-mask batch job.

Keys are read only from KEY_0 ... KEY_29 (or KEY_ENVS). Nothing secret is
written to manifests or logs. The input may be the supplied tar.gz directly.
"""
from __future__ import annotations

import argparse, base64, io, json, os, re, tarfile, time
from pathlib import Path
from typing import Any

from PIL import Image
from openai import OpenAI

STAGE1_PROMPT = """You are a visual reasoning expert specialized in perspective and projection consistency analysis.

Analyze the provided image only from the perspective of spatial structure and projection relationships.

Your task is to determine whether the image contains any visible perspective or projection inconsistency and explain the visual evidence.

Focus on:
- structural edges and boundaries
- planar surfaces
- object orientations
- depth relationships
- convergence directions
- perspective consistency between related structures

If a perspective inconsistency exists, explain:
1. Which visible structure contains the issue.
2. Which surrounding structures provide perspective references.
3. What perspective relationship should hold.
4. How the observed structure violates this relationship.

The explanation should focus only on perspective reasoning.
Do not discuss general image quality, aesthetics, lighting, colors, textures, semantic correctness, or image generation artifacts.
Avoid vague statements such as "The perspective is wrong." or "The geometry is incorrect."
Instead, provide concrete visual evidence describing the affected structure and the violated perspective relationship.

If no clear perspective inconsistency exists, state that no clear perspective error is observed.
Return only one concise explanation paragraph."""

STAGE2_TEMPLATE = """The image above is the original input image.

Generate a binary localization mask for the described perspective or projection inconsistency.

Perspective analysis:
{explanation}

The output must be a black-and-white binary mask aligned with the original image.

Requirements:
- Use pure black background.
- Use pure white pixels only for the region containing the perspective/projection inconsistency.
- White indicates the affected error region.
- Black indicates all unrelated regions.

The mask should highlight the actual structure causing the inconsistency, localize it precisely, include only necessary local context, and follow the region described above.
Do not highlight the whole image, unrelated objects, background areas, textures, colors, lighting changes, or image quality issues.
Do not add grayscale, color heatmaps, transparency, text, labels, arrows, bounding boxes, or outlines.
Output only a clean binary mask. If the analysis indicates no clear perspective inconsistency, output an entirely black mask."""


def data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if text:
        return str(text).strip()
    out = getattr(resp, "output", None) or []
    parts = []
    for item in out:
        for c in getattr(item, "content", None) or []:
            t = getattr(c, "text", None)
            if t: parts.append(str(t))
    return "\n".join(parts).strip()


def make_binary(raw: bytes, target: Path) -> None:
    im = Image.open(io.BytesIO(raw)).convert("L")
    # Image APIs occasionally return a different resolution; align exactly to
    # the source image and enforce only 0/255 values.
    src = Image.open(target.with_name(target.stem + ".source" + target.suffix)) if False else None
    im = im.resize(Image.open(target.with_suffix(".jpg")).size, Image.Resampling.NEAREST) if False else im
    import numpy as np
    arr = np.asarray(im)
    Image.fromarray((arr >= 128).astype("uint8") * 255, mode="L").save(target)


def keys_for(stage: int) -> list[str]:
    rng = range(0, 10) if stage == 1 else range(10, 20)
    names = [f"KEY_{i}" for i in rng]
    keys = [os.environ[n] for n in names if os.environ.get(n)]
    if not keys:
        raise SystemExit(f"No keys found for stage {stage}; export KEY_{0 if stage == 1 else 10} ... first")
    return keys


def call_with_rotation(fn, keys: list[str], base_url: str, model: str, attempts: int = 4):
    last = None
    for n in range(attempts):
        key = keys[n % len(keys)]
        try:
            return fn(OpenAI(api_key=key, base_url=base_url), model)
        except Exception as exc:
            last = exc
            time.sleep(min(30, 2 ** n))
    raise last


def extract_if_needed(archive: Path, cache: Path) -> Path:
    # Discover the archive's single top-level directory; the fresh score
    # package uses a different name from the larger train/test archive.
    with tarfile.open(archive, "r:gz") as tf:
        tops = sorted({n.split("/", 1)[0] for n in tf.getnames() if "/" in n})
    if not tops: raise SystemExit("Archive has no top-level directory")
    root = cache / tops[0]
    if root.exists() and (root / "metadata").exists(): return root
    cache.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf: tf.extractall(cache)
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=Path(".perspective_api_cache"))
    # Compatible gateway used by this batch job. Override with
    # OPENAI_BASE_URL or --base-url when needed.
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://svip.xuedingmao.com/v1"))
    ap.add_argument("--stage1-model", default="gpt-5.6-sol")
    ap.add_argument("--stage2-model", default="gpt-image-2")
    ap.add_argument("--only", choices=["all", "good", "bad"], default="all")
    args = ap.parse_args(); root = extract_if_needed(args.archive.resolve(), args.cache_root.resolve())
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    records = []
    split_files = sorted((root / "metadata").glob("*.jsonl"))
    if not split_files: raise SystemExit(f"No metadata jsonl found under {root / 'metadata'}")
    for split_file in split_files:
        split = "train" if "train" in split_file.name.lower() else "test"
        for line in split_file.read_text(encoding="utf8").splitlines():
            if not line.strip(): continue
            m = json.loads(line)
            for label in ("good", "bad"):
                if args.only != "all" and label != args.only: continue
                rel = m[f"{label}_image"]; records.append({"split": split, "label": label, "group_id": m.get("group_id"), "category": m.get("intended_category"), "path": str((root / rel).resolve())})
    records.sort(key=lambda r: (r["split"], r["group_id"], r["label"]))
    manifest = out / "results.jsonl"; done = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf8").splitlines():
            try:
                r=json.loads(line); done[(r["split"],r["group_id"],r["label"])] = r
            except Exception: pass
    k1, k2 = keys_for(1), keys_for(2)
    for idx, r in enumerate(records, 1):
        key = (r["split"], r["group_id"], r["label"]); prev = done.get(key, {})
        try:
            explanation = prev.get("explanation")
            if not explanation:
                def f(client, model):
                    return client.responses.create(model=model, input=[{"role":"user","content":[{"type":"input_text","text":STAGE1_PROMPT},{"type":"input_image","image_url":data_url(Path(r["path"]))}]}])
                explanation = call_with_rotation(f, k1, args.base_url, args.stage1_model)
                explanation = response_text(explanation)
            r["explanation"] = explanation
            mask_rel = f"masks/{r['split']}/{r['label']}/{r['group_id']}.png"; mask = out / mask_rel; mask.parent.mkdir(parents=True, exist_ok=True)
            if not mask.exists():
                prompt = STAGE2_TEMPLATE.format(explanation=explanation)
                def g(client, model):
                    with open(r["path"], "rb") as fh: return client.images.edit(model=model, image=fh, prompt=prompt, size="auto")
                resp = call_with_rotation(g, k2, args.base_url, args.stage2_model)
                item = resp.data[0]; b64 = getattr(item, "b64_json", None)
                if not b64: raise RuntimeError("stage2 response did not contain b64_json")
                raw = base64.b64decode(b64); im = Image.open(io.BytesIO(raw)).convert("L").resize(Image.open(r["path"]).size, Image.Resampling.NEAREST)
                import numpy as np
                Image.fromarray((np.asarray(im) >= 128).astype("uint8") * 255, mode="L").save(mask)
            r["mask"] = str(mask); r["status"] = "ok"
        except Exception as exc:
            r["status"] = "error"; r["error"] = f"{type(exc).__name__}: {exc}"
        with manifest.open("a", encoding="utf8") as fh: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{idx}/{len(records)}] {r['split']} {r['label']} {r['group_id']} {r['status']}", flush=True)


if __name__ == "__main__": main()

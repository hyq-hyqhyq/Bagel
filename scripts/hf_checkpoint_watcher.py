#!/usr/bin/env python3
# Linux-only watcher: uses /proc and fcntl for safe single-instance operation.
"""Upload completed rolling checkpoints to Hugging Face and reclaim local space.

Policy:
1. A numeric checkpoint directory is complete after its file manifest stays unchanged
   for --stable-seconds and no process has an open file inside it.
2. A completed checkpoint is uploaded to <remote-prefix>/<checkpoint-name>/.
3. A remote _UPLOAD_COMPLETE.json marker is uploaded last and read back for verification.
4. An uploaded checkpoint is deleted locally only after a newer numeric checkpoint
   directory appears (the newer checkpoint may still be in the process of being saved).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


CHECKPOINT_RE = re.compile(r"^[0-9]{8}$")
MARKER_NAME = "_UPLOAD_COMPLETE.json"
STATE_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("version") == STATE_VERSION:
            value.setdefault("checkpoints", {})
            return value
    except FileNotFoundError:
        pass
    except Exception:
        logging.exception("Could not read state file %s; starting with empty state", path)
    return {"version": STATE_VERSION, "checkpoints": {}}


def checkpoint_dirs(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and CHECKPOINT_RE.fullmatch(path.name)
        ),
        key=lambda path: int(path.name),
    )


def checkpoint_has_payload(root: Path) -> bool:
    try:
        return any(path.is_file() and path.stat().st_size > 0 for path in root.rglob("*"))
    except FileNotFoundError:
        return False


def file_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == ".cache" or relative.startswith(".cache/"):
            continue
        stat = path.stat()
        rows.append(
            {
                "path": relative,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def manifest_signature(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "files": rows,
    }


def has_open_file(root: Path) -> bool:
    prefix = str(root.resolve()) + os.sep
    proc = Path("/proc")
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            if target.startswith(prefix):
                return True
    return False


def remote_marker_path(remote_prefix: str, checkpoint_name: str) -> str:
    return str(PurePosixPath(remote_prefix) / checkpoint_name / MARKER_NAME)


def read_remote_marker(
    repo_id: str,
    repo_type: str,
    marker_path: str,
) -> dict[str, Any] | None:
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=marker_path,
            force_download=True,
        )
        value = json.loads(Path(local_path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def marker_matches(
    marker: dict[str, Any] | None,
    checkpoint_name: str,
    manifest: dict[str, Any],
) -> bool:
    if not marker:
        return False
    return (
        marker.get("checkpoint") == checkpoint_name
        and marker.get("file_count") == manifest.get("file_count")
        and marker.get("total_bytes") == manifest.get("total_bytes")
        and marker.get("files") == manifest.get("files")
    )


def upload_checkpoint(
    api: HfApi,
    checkpoint: Path,
    manifest: dict[str, Any],
    repo_id: str,
    repo_type: str,
    remote_prefix: str,
) -> dict[str, Any]:
    remote_dir = str(PurePosixPath(remote_prefix) / checkpoint.name)
    marker_path = remote_marker_path(remote_prefix, checkpoint.name)

    existing = read_remote_marker(repo_id, repo_type, marker_path)
    if marker_matches(existing, checkpoint.name, manifest):
        logging.info("Remote marker already verifies %s; skipping upload", checkpoint.name)
        return existing

    logging.info(
        "Uploading %s (%d files, %.2f GiB) to %s/%s",
        checkpoint,
        manifest["file_count"],
        manifest["total_bytes"] / (1024**3),
        repo_id,
        remote_dir,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(checkpoint),
        path_in_repo=remote_dir,
        ignore_patterns=[".cache/**", "**/.cache/**"],
        commit_message=f"Upload checkpoint {checkpoint.name}",
    )

    marker = {
        "version": STATE_VERSION,
        "checkpoint": checkpoint.name,
        "repo_id": repo_id,
        "repo_type": repo_type,
        "remote_path": remote_dir,
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "files": manifest["files"],
        "upload_completed_at_utc": utc_now(),
    }
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=(
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        path_in_repo=marker_path,
        commit_message=f"Mark checkpoint {checkpoint.name} upload complete",
    )

    verified = read_remote_marker(repo_id, repo_type, marker_path)
    if not marker_matches(verified, checkpoint.name, manifest):
        raise RuntimeError(f"Remote marker verification failed for {checkpoint.name}")
    logging.info("Remote upload verified for %s", checkpoint.name)
    return verified


def safe_delete_checkpoint(root: Path, checkpoint: Path) -> None:
    resolved_root = root.resolve()
    resolved = checkpoint.resolve()
    if resolved.parent != resolved_root:
        raise RuntimeError(f"Refusing deletion outside checkpoint root: {resolved}")
    if not CHECKPOINT_RE.fullmatch(resolved.name):
        raise RuntimeError(f"Refusing deletion of non-checkpoint path: {resolved}")
    logging.warning("Deleting uploaded old checkpoint %s", resolved)
    shutil.rmtree(resolved)


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/bagel/outputs/"
            "reason_heatmap_train4000_full_30k_bf16/checkpoints"
        ),
    )
    parser.add_argument("--repo-id", default="dfgfhdhhhghg/bagel")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--remote-prefix", default="output/run_trial")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stable-seconds", type=int, default=600)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.checkpoint_root.resolve()
    output_root = root.parent
    state_file = args.state_file or output_root / "hf_checkpoint_watcher_state.json"
    log_file = args.log_file or output_root / "hf_checkpoint_watcher.log"
    lock_file = output_root / "hf_checkpoint_watcher.lock"
    configure_logging(log_file)

    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root does not exist: {root}")

    lock_handle = lock_file.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another checkpoint watcher is already running") from error
    lock_handle.write(str(os.getpid()) + "\n")
    lock_handle.flush()

    state = load_state(state_file)
    api = HfApi()
    whoami = api.whoami()
    api.repo_info(repo_id=args.repo_id, repo_type=args.repo_type)
    logging.info(
        "Authenticated as %s; watching %s; remote=%s/%s",
        whoami.get("name", "unknown"),
        root,
        args.repo_id,
        args.remote_prefix,
    )

    while True:
        try:
            checkpoints = checkpoint_dirs(root)
            current_names = {path.name for path in checkpoints}

            # Reclaim an uploaded checkpoint as soon as a newer save directory appears.
            newer_save_started = bool(checkpoints) and checkpoint_has_payload(checkpoints[-1])
            for checkpoint in checkpoints[:-1] if newer_save_started else []:
                entry = state["checkpoints"].get(checkpoint.name, {})
                uploaded_manifest = entry.get("uploaded_manifest")
                if not entry.get("uploaded") or not isinstance(uploaded_manifest, dict):
                    continue
                marker = read_remote_marker(
                    args.repo_id,
                    args.repo_type,
                    remote_marker_path(args.remote_prefix, checkpoint.name),
                )
                if not marker_matches(marker, checkpoint.name, uploaded_manifest):
                    logging.error(
                        "Remote verification no longer matches %s; keeping local copy",
                        checkpoint.name,
                    )
                    continue
                safe_delete_checkpoint(root, checkpoint)
                entry["deleted_at_utc"] = utc_now()
                atomic_write_json(state_file, state)

            checkpoints = checkpoint_dirs(root)
            for checkpoint in checkpoints:
                entry = state["checkpoints"].setdefault(checkpoint.name, {})
                if entry.get("uploaded"):
                    continue

                files = file_manifest(checkpoint)
                manifest = manifest_signature(files)
                signature = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                now = time.time()

                if entry.get("observed_signature") != signature:
                    entry["observed_signature"] = signature
                    entry["stable_since"] = now
                    entry["last_observed_at_utc"] = utc_now()
                    atomic_write_json(state_file, state)
                    logging.info(
                        "Checkpoint %s changing: %d files, %.2f GiB",
                        checkpoint.name,
                        manifest["file_count"],
                        manifest["total_bytes"] / (1024**3),
                    )
                    continue

                stable_for = now - float(entry.get("stable_since", now))
                if stable_for < args.stable_seconds:
                    continue
                if not files:
                    logging.warning("Checkpoint %s is empty; not uploading", checkpoint.name)
                    continue
                if has_open_file(checkpoint):
                    logging.info("Checkpoint %s still has open files", checkpoint.name)
                    continue

                verified_marker = upload_checkpoint(
                    api=api,
                    checkpoint=checkpoint,
                    manifest=manifest,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    remote_prefix=args.remote_prefix,
                )
                entry["uploaded"] = True
                entry["uploaded_manifest"] = manifest
                entry["remote_marker"] = verified_marker
                entry["uploaded_at_utc"] = utc_now()
                atomic_write_json(state_file, state)

            # Preserve history for deleted checkpoints, but note what is currently local.
            state["local_checkpoints"] = sorted(current_names)
            state["last_poll_at_utc"] = utc_now()
            atomic_write_json(state_file, state)
        except KeyboardInterrupt:
            logging.info("Interrupted")
            return 0
        except Exception:
            logging.exception("Watcher iteration failed; will retry")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

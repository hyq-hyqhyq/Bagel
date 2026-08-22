#!/usr/bin/env python3
"""Upload completed raw FSDP models and reclaim old local checkpoints.

This watcher is intentionally conservative about checkpoint completion. A numeric
step directory is eligible only after every expected FSDP file exists, its full
file manifest has remained unchanged for a stability window, and no process has
an open file below the directory. Only ``model.safetensors`` is uploaded. The
By default the entire local step directory is removed after the remote raw model
is verified. ``--keep-latest-local N`` instead retains the newest N numeric
checkpoints per watched run and prunes older verified uploads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - the watcher runs on Linux.
    fcntl = None

from huggingface_hub import HfApi
from safetensors import safe_open


CHECKPOINT_RE = re.compile(r"^[0-9]{7,8}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STATE_VERSION = 1


@dataclass(frozen=True)
class WatchSpec:
    name: str
    checkpoint_root: Path
    remote_stem: str


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
            value.setdefault("runs", {})
            return value
    except FileNotFoundError:
        pass
    except Exception:
        logging.exception("Could not read %s; starting with empty state", path)
    return {"version": STATE_VERSION, "runs": {}}


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def parse_watch_specs(values: list[list[str]]) -> list[WatchSpec]:
    specs: list[WatchSpec] = []
    names: set[str] = set()
    roots: set[Path] = set()
    for name, checkpoint_root, remote_stem in values:
        root = Path(checkpoint_root).expanduser().resolve()
        remote = str(PurePosixPath(remote_stem.strip("/")))
        if not name or name in names:
            raise ValueError(f"Duplicate or empty watch name: {name!r}")
        if root in roots:
            raise ValueError(f"Checkpoint root is watched twice: {root}")
        if not remote or remote == "." or ".." in PurePosixPath(remote).parts:
            raise ValueError(f"Unsafe remote stem: {remote_stem!r}")
        names.add(name)
        roots.add(root)
        specs.append(WatchSpec(name, root, remote))
    return specs


def checkpoint_dirs(
    root: Path,
    min_step: int,
    max_step: int | None,
    step_multiple: int,
) -> list[Path]:
    if not root.is_dir():
        return []
    checkpoints = []
    for path in root.iterdir():
        if not path.is_dir() or not CHECKPOINT_RE.fullmatch(path.name):
            continue
        step = int(path.name)
        if step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        if step_multiple > 0 and step % step_multiple:
            continue
        checkpoints.append(path)
    return sorted(checkpoints, key=lambda path: int(path.name))


def expected_files(optimizer_shards: int) -> list[str]:
    files = [
        "ema.safetensors",
        "model.safetensors",
        "scheduler.pt",
        "data_status.pt",
    ]
    files.extend(
        f"optimizer.{index:05d}-of-{optimizer_shards:05d}.pt"
        for index in range(optimizer_shards)
    )
    return files


def missing_expected_files(checkpoint: Path, optimizer_shards: int) -> list[str]:
    return [
        name
        for name in expected_files(optimizer_shards)
        if not (checkpoint / name).is_file()
        or (checkpoint / name).stat().st_size <= 0
    ]


def file_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return rows


def manifest_signature(rows: list[dict[str, Any]]) -> str:
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def has_open_file(root: Path) -> bool:
    prefix = str(root.resolve()) + os.sep
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            descriptors = list((process / "fd").iterdir())
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


def validate_raw_model(path: Path) -> int:
    # Reading keys with the NumPy backend validates the safetensors header without
    # importing PyTorch into the long-running watcher process.
    with safe_open(path, framework="np", device="cpu") as handle:
        tensor_count = len(handle.keys())
    if tensor_count <= 0:
        raise RuntimeError(f"No tensors found in {path}")
    return tensor_count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def remote_path(spec: WatchSpec, checkpoint: Path) -> str:
    step = int(checkpoint.name)
    directory = f"{spec.remote_stem}_step{step:07d}"
    return str(PurePosixPath(directory) / "model.safetensors")


def remote_file_info(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
) -> tuple[int | None, str | None]:
    info = api.repo_info(
        repo_id=repo_id,
        repo_type=repo_type,
        files_metadata=True,
    )
    sibling = next(
        (item for item in info.siblings if item.rfilename == path_in_repo),
        None,
    )
    if sibling is None:
        return None, None

    remote_sha: str | None = None
    lfs = getattr(sibling, "lfs", None)
    if isinstance(lfs, dict):
        remote_sha = lfs.get("sha256") or lfs.get("oid")
    elif lfs is not None:
        remote_sha = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)

    if isinstance(remote_sha, str):
        remote_sha = remote_sha.lower().removeprefix("sha256:")
        if not SHA256_RE.fullmatch(remote_sha):
            remote_sha = None

    return getattr(sibling, "size", None), remote_sha


def upload_and_verify_raw_model(
    api: HfApi,
    spec: WatchSpec,
    checkpoint: Path,
    repo_id: str,
    repo_type: str,
    model_sha256: str,
) -> dict[str, Any]:
    model = checkpoint / "model.safetensors"
    local_size = model.stat().st_size
    path_in_repo = remote_path(spec, checkpoint)

    logging.info(
        "Uploading raw model %s (%.2f GiB) to %s/%s",
        model,
        local_size / (1024**3),
        repo_id,
        path_in_repo,
    )
    commit = api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(model),
        path_in_repo=path_in_repo,
        commit_message=f"Upload {spec.name} raw model at step {int(checkpoint.name)}",
    )

    remote_size, remote_sha = remote_file_info(
        api, repo_id, repo_type, path_in_repo
    )
    if remote_size != local_size:
        raise RuntimeError(
            f"Remote size mismatch for {path_in_repo}: "
            f"local={local_size}, remote={remote_size}"
        )
    if remote_sha is not None and remote_sha != model_sha256:
        raise RuntimeError(
            f"Remote SHA-256 mismatch for {path_in_repo}: "
            f"local={model_sha256}, remote={remote_sha}"
        )

    logging.info(
        "Verified remote raw model %s (size=%d, sha256=%s)",
        path_in_repo,
        remote_size,
        remote_sha or "not exposed by Hub API",
    )
    return {
        "path_in_repo": path_in_repo,
        "local_size": local_size,
        "local_sha256": model_sha256,
        "remote_size": remote_size,
        "remote_sha256": remote_sha,
        "commit_url": str(commit),
        "uploaded_at_utc": utc_now(),
    }


def reuse_verified_upload(
    api: HfApi,
    entry: dict[str, Any],
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    local_size: int,
    local_sha256: str,
) -> dict[str, Any] | None:
    upload = entry.get("upload")
    if not isinstance(upload, dict):
        return None
    if (
        upload.get("path_in_repo") != path_in_repo
        or upload.get("local_size") != local_size
        or upload.get("local_sha256") != local_sha256
    ):
        return None

    remote_size, remote_sha = remote_file_info(
        api, repo_id, repo_type, path_in_repo
    )
    if remote_size != local_size:
        return None
    if remote_sha is not None and remote_sha != local_sha256:
        return None
    logging.info("Reusing previously verified remote raw model %s", path_in_repo)
    return upload


def safe_delete_checkpoint(root: Path, checkpoint: Path) -> None:
    resolved_root = root.resolve()
    resolved = checkpoint.resolve()
    if resolved.parent != resolved_root:
        raise RuntimeError(f"Refusing deletion outside {resolved_root}: {resolved}")
    if not CHECKPOINT_RE.fullmatch(resolved.name):
        raise RuntimeError(f"Refusing deletion of non-checkpoint path: {resolved}")
    logging.warning("Deleting complete local checkpoint %s", resolved)
    shutil.rmtree(resolved)


def cached_upload_for_current_checkpoint(
    spec: WatchSpec,
    checkpoint: Path,
    entry: dict[str, Any],
    signature: str,
) -> dict[str, Any] | None:
    """Return verified state for an unchanged local checkpoint."""
    upload = entry.get("upload")
    if not isinstance(upload, dict):
        return None
    if entry.get("observed_signature") != signature:
        return None

    model = checkpoint / "model.safetensors"
    local_size = model.stat().st_size
    local_sha256 = upload.get("local_sha256")
    if (
        upload.get("path_in_repo") != remote_path(spec, checkpoint)
        or upload.get("local_size") != local_size
        or not isinstance(local_sha256, str)
        or not SHA256_RE.fullmatch(local_sha256)
    ):
        return None

    return upload


def verified_upload_for_current_checkpoint(
    api: HfApi,
    spec: WatchSpec,
    checkpoint: Path,
    entry: dict[str, Any],
    signature: str,
    repo_id: str,
    repo_type: str,
) -> dict[str, Any] | None:
    """Recheck a cached upload against Hugging Face without hashing again."""
    upload = cached_upload_for_current_checkpoint(
        spec,
        checkpoint,
        entry,
        signature,
    )
    if upload is None:
        return None
    return reuse_verified_upload(
        api,
        entry,
        repo_id,
        repo_type,
        upload["path_in_repo"],
        upload["local_size"],
        upload["local_sha256"],
    )


def prune_old_uploaded_checkpoints(
    api: HfApi,
    spec: WatchSpec,
    run_state: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if args.keep_local or args.keep_latest_local <= 0:
        return

    checkpoints = checkpoint_dirs(
        spec.checkpoint_root,
        min_step=args.min_step,
        max_step=args.max_step,
        step_multiple=args.step_multiple,
    )
    uploaded_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if isinstance(
            run_state.get("checkpoints", {})
            .get(checkpoint.name, {})
            .get("upload"),
            dict,
        )
    ]
    while len(uploaded_checkpoints) > args.keep_latest_local:
        checkpoint = uploaded_checkpoints[0]
        entry = run_state.setdefault("checkpoints", {}).setdefault(
            checkpoint.name, {}
        )
        missing = missing_expected_files(checkpoint, args.optimizer_shards)
        if missing:
            logging.info(
                "Cannot prune oldest %s/%s; checkpoint is incomplete",
                spec.name,
                checkpoint.name,
            )
            return
        if has_open_file(checkpoint):
            logging.info(
                "Cannot prune oldest %s/%s; files are still open",
                spec.name,
                checkpoint.name,
            )
            return

        signature = manifest_signature(file_manifest(checkpoint))
        upload = verified_upload_for_current_checkpoint(
            api,
            spec,
            checkpoint,
            entry,
            signature,
            args.repo_id,
            args.repo_type,
        )
        if upload is None:
            logging.info(
                "Cannot prune oldest %s/%s; remote upload is not verified",
                spec.name,
                checkpoint.name,
            )
            return

        safe_delete_checkpoint(spec.checkpoint_root, checkpoint)
        entry["deleted_at_utc"] = utc_now()
        entry.pop("retained_local_at_utc", None)
        atomic_write_json(args.state_file, args.state)
        uploaded_checkpoints.pop(0)


def process_checkpoint(
    api: HfApi,
    spec: WatchSpec,
    checkpoint: Path,
    run_state: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    entry = run_state.setdefault("checkpoints", {}).setdefault(checkpoint.name, {})
    missing = missing_expected_files(checkpoint, args.optimizer_shards)
    if missing:
        if entry.get("missing_files") != missing:
            logging.info("%s/%s incomplete; missing %s", spec.name, checkpoint.name, missing)
            entry["missing_files"] = missing
            entry.pop("stable_since", None)
        return False
    entry.pop("missing_files", None)

    manifest = file_manifest(checkpoint)
    signature = manifest_signature(manifest)
    now = time.time()
    if entry.get("observed_signature") != signature:
        entry["observed_signature"] = signature
        entry["stable_since"] = now
        entry["last_changed_at_utc"] = utc_now()
        logging.info(
            "%s/%s observed complete; waiting %ds stability (%d files, %.2f GiB)",
            spec.name,
            checkpoint.name,
            args.stable_seconds,
            len(manifest),
            sum(item["size"] for item in manifest) / (1024**3),
        )
        return False

    stable_for = now - float(entry.get("stable_since", now))
    if stable_for < args.stable_seconds:
        return False
    if has_open_file(checkpoint):
        logging.info("%s/%s still has open files", spec.name, checkpoint.name)
        return False

    model = checkpoint / "model.safetensors"
    model_stat = model.stat()
    upload = cached_upload_for_current_checkpoint(
        spec,
        checkpoint,
        entry,
        signature,
    )
    if upload is not None:
        if args.keep_local or args.keep_latest_local > 0:
            return True
        upload = verified_upload_for_current_checkpoint(
            api,
            spec,
            checkpoint,
            entry,
            signature,
            args.repo_id,
            args.repo_type,
        )
        if upload is None:
            return False
        safe_delete_checkpoint(spec.checkpoint_root, checkpoint)
        entry["deleted_at_utc"] = utc_now()
        atomic_write_json(args.state_file, args.state)
        return True

    tensor_count = validate_raw_model(model)
    logging.info(
        "%s/%s raw model header is valid (%d tensors); computing SHA-256",
        spec.name,
        checkpoint.name,
        tensor_count,
    )
    model_sha256 = sha256_file(model)

    path_in_repo = remote_path(spec, checkpoint)
    upload = reuse_verified_upload(
        api,
        entry,
        args.repo_id,
        args.repo_type,
        path_in_repo,
        model_stat.st_size,
        model_sha256,
    )
    if upload is None:
        upload = upload_and_verify_raw_model(
            api=api,
            spec=spec,
            checkpoint=checkpoint,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            model_sha256=model_sha256,
        )

    # Do not delete if the trainer rewrote this step while hashing/uploading it.
    current_stat = model.stat()
    if (
        current_stat.st_size != model_stat.st_size
        or current_stat.st_mtime_ns != model_stat.st_mtime_ns
        or manifest_signature(file_manifest(checkpoint)) != signature
        or has_open_file(checkpoint)
    ):
        logging.error(
            "%s/%s changed during upload; keeping it for another pass",
            spec.name,
            checkpoint.name,
        )
        entry["observed_signature"] = None
        entry["last_upload_attempt"] = upload
        return False

    entry["upload"] = upload
    entry["tensor_count"] = tensor_count
    entry["verified_at_utc"] = utc_now()
    atomic_write_json(args.state_file, args.state)

    if args.keep_local:
        logging.info("Keeping %s because --keep-local was set", checkpoint)
        entry["kept_local"] = True
        return True

    if args.keep_latest_local > 0:
        entry["retained_local_at_utc"] = utc_now()
        logging.info(
            "Retaining %s locally; newest %d checkpoints are kept",
            checkpoint,
            args.keep_latest_local,
        )
        atomic_write_json(args.state_file, args.state)
        return True

    safe_delete_checkpoint(spec.checkpoint_root, checkpoint)
    entry["deleted_at_utc"] = utc_now()
    atomic_write_json(args.state_file, args.state)
    return True


def stop_step_is_done(specs: list[WatchSpec], state: dict[str, Any], step: int) -> bool:
    checkpoint_name = f"{step:07d}"
    for spec in specs:
        entry = (
            state.get("runs", {})
            .get(spec.name, {})
            .get("checkpoints", {})
            .get(checkpoint_name, {})
        )
        if not (
            entry.get("deleted_at_utc")
            or entry.get("kept_local")
            or isinstance(entry.get("upload"), dict)
        ):
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch multiple FSDP checkpoint roots, upload only each completed "
            "model.safetensors, verify it on Hugging Face, then delete the full "
            "local checkpoint directory."
        )
    )
    parser.add_argument(
        "--watch",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "CHECKPOINT_ROOT", "REMOTE_STEM"),
        help=(
            "Repeat for each run. REMOTE_STEM receives _stepXXXXXXX/"
            "model.safetensors."
        ),
    )
    parser.add_argument("--repo-id", default="dfgfhdhhhghg/bagel")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--optimizer-shards", type=int, default=4)
    parser.add_argument("--min-step", type=int, default=2000)
    parser.add_argument("--max-step", type=int, default=30000)
    parser.add_argument("--step-multiple", type=int, default=2000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-seconds", type=int, default=180)
    parser.add_argument("--stop-after-step", type=int, default=30000)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--keep-local", action="store_true")
    parser.add_argument(
        "--keep-latest-local",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Keep the newest N checkpoints per run after verified raw-model "
            "uploads; older verified checkpoint directories are deleted."
        ),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.optimizer_shards <= 0:
        parser.error("--optimizer-shards must be positive")
    if args.poll_seconds <= 0 or args.stable_seconds < 0:
        parser.error("poll/stability intervals are invalid")
    if args.step_multiple <= 0:
        parser.error("--step-multiple must be positive")
    if args.max_step is not None and args.max_step < args.min_step:
        parser.error("--max-step must be >= --min-step")
    if args.keep_latest_local < 0:
        parser.error("--keep-latest-local must be non-negative")
    if args.keep_local and args.keep_latest_local:
        parser.error("--keep-local and --keep-latest-local cannot be combined")
    return args


def main() -> int:
    args = parse_args()
    if fcntl is None:
        raise RuntimeError("This watcher requires Linux fcntl locking")

    specs = parse_watch_specs(args.watch)
    args.state_file = args.state_file.expanduser().resolve()
    args.log_file = args.log_file.expanduser().resolve()
    lock_file = (
        args.lock_file.expanduser().resolve()
        if args.lock_file
        else args.state_file.with_suffix(".lock")
    )
    configure_logging(args.log_file)

    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_file.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another watcher owns {lock_file}") from error
    lock_handle.write(str(os.getpid()) + "\n")
    lock_handle.flush()

    state = load_state(args.state_file)
    args.state = state
    for spec in specs:
        run_state = state["runs"].setdefault(spec.name, {})
        run_state["checkpoint_root"] = str(spec.checkpoint_root)
        run_state["remote_stem"] = spec.remote_stem
        run_state.setdefault("checkpoints", {})
    atomic_write_json(args.state_file, state)

    api = HfApi()
    user = api.whoami()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=True,
        exist_ok=True,
    )
    logging.info(
        "Authenticated as %s; repo=%s; delete_after_upload=%s",
        user.get("name", "unknown"),
        args.repo_id,
        not args.keep_local,
    )
    for spec in specs:
        logging.info(
            "Watching %s: %s -> %s_stepXXXXXXX/model.safetensors",
            spec.name,
            spec.checkpoint_root,
            spec.remote_stem,
        )

    while True:
        try:
            for spec in specs:
                run_state = state["runs"][spec.name]
                checkpoints = checkpoint_dirs(
                    spec.checkpoint_root,
                    min_step=args.min_step,
                    max_step=args.max_step,
                    step_multiple=args.step_multiple,
                )
                if not spec.checkpoint_root.is_dir():
                    logging.warning(
                        "Checkpoint root does not exist yet: %s",
                        spec.checkpoint_root,
                    )
                for checkpoint in checkpoints:
                    try:
                        process_checkpoint(api, spec, checkpoint, run_state, args)
                    except Exception:
                        logging.exception(
                            "Failed processing %s/%s; keeping it and continuing",
                            spec.name,
                            checkpoint.name,
                        )

                prune_old_uploaded_checkpoints(
                    api,
                    spec,
                    run_state,
                    args,
                )

                run_state["local_checkpoints"] = [
                    path.name
                    for path in checkpoint_dirs(
                        spec.checkpoint_root,
                        min_step=args.min_step,
                        max_step=args.max_step,
                        step_multiple=args.step_multiple,
                    )
                ]
                run_state["last_poll_at_utc"] = utc_now()
                atomic_write_json(args.state_file, state)

            if args.stop_after_step and stop_step_is_done(
                specs, state, args.stop_after_step
            ):
                logging.info(
                    "All runs processed step %d; watcher is complete",
                    args.stop_after_step,
                )
                return 0
            if args.once:
                return 0
        except KeyboardInterrupt:
            logging.info("Interrupted")
            return 0
        except Exception:
            logging.exception("Watcher iteration failed; will retry")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

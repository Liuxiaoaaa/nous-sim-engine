#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import Manager
import pickle
import tarfile
from pathlib import Path
from typing import Any

from nous_sim_engine.adapters.dataset_loader import save_source_scene_context
from nous_sim_engine.adapters.internal import build_future_trajectory_from_frame
from nous_sim_engine.core.types import SceneContext


def _discover_shard_tar_paths(shard_root: Path, *, max_paths: int = 0) -> list[Path]:
    if shard_root.is_file():
        return [shard_root]

    shard_list = shard_root / "_shard_list.json"
    if max_paths <= 0 and shard_list.exists():
        paths = _load_shard_list(shard_list, shard_root)
        if paths:
            return paths

    paths: list[Path] = []
    for pattern in ("*.tar", "*.tar.gz", "*.tgz"):
        for path in shard_root.glob(pattern):
            if not path.is_file():
                continue
            paths.append(path)
            if max_paths > 0 and len(paths) >= max_paths:
                return paths
    return sorted(paths)


def _load_shard_list(shard_list: Path, shard_root: Path) -> list[Path]:
    try:
        data = json.loads(shard_list.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("files") or data.get("shards") or []
    else:
        entries = []
    paths: list[Path] = []
    for entry in entries:
        raw_path = entry.get("path") if isinstance(entry, dict) else entry
        if not isinstance(raw_path, str):
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = shard_root / path
        if path.exists():
            paths.append(path)
    return sorted(paths)


def _frame_member_ids(member_name: str) -> tuple[str, str]:
    parts = Path(member_name).parts
    if len(parts) >= 3 and parts[-1] == "frame.json":
        return parts[-3], parts[-2]
    raise ValueError(f"Unexpected frame member path: {member_name}")


def _empty_result() -> dict[str, Any]:
    return {
        "tar_files": 0,
        "frames": 0,
        "pkl_missing": 0,
        "already_gt": 0,
        "old_gt_missing": 0,
        "source_gt_missing": 0,
        "updated": 0,
        "failed": 0,
        "failures": [],
    }


def _limit_claim(payload: dict[str, Any]) -> bool:
    limit = int(payload.get("limit") or 0)
    if limit <= 0:
        return True
    counter = payload.get("limit_counter")
    lock = payload.get("limit_lock")
    if counter is None or lock is None:
        return True
    with lock:
        if counter.value >= limit:
            return False
        counter.value += 1
        return True


def _load_scene_context(path: Path) -> SceneContext:
    with path.open("rb") as file_obj:
        ctx = pickle.load(file_obj)
    if not isinstance(ctx, SceneContext):
        raise TypeError(f"Expected SceneContext in {path}, got {type(ctx).__name__}")
    return ctx


def _process_tar_task(payload: dict[str, Any]) -> dict[str, Any]:
    tar_path = Path(payload["tar_path"])
    scene_context_root = Path(payload["scene_context_root"])
    dry_run = bool(payload.get("dry_run"))
    overwrite_existing = bool(payload.get("overwrite_existing"))
    update_pdm = bool(payload.get("update_pdm"))

    result = _empty_result()
    result["tar_files"] = 1

    try:
        tar_file = tarfile.open(tar_path, "r:*")
    except (OSError, tarfile.TarError) as exc:
        result["failed"] += 1
        result["failures"].append({"path": str(tar_path), "error": str(exc)})
        return result

    with tar_file:
        for member in tar_file:
            if not member.isfile() or not member.name.endswith("/frame.json"):
                continue
            if not _limit_claim(payload):
                break

            result["frames"] += 1
            try:
                case_id, scene_token = _frame_member_ids(member.name)
                scene_path = scene_context_root / case_id / f"{scene_token}.pkl"
                if not scene_path.exists():
                    result["pkl_missing"] += 1
                    continue

                ctx = _load_scene_context(scene_path)
                old_gt_missing = ctx.gt_trajectory is None
                if old_gt_missing:
                    result["old_gt_missing"] += 1
                elif not overwrite_existing:
                    result["already_gt"] += 1
                    continue

                extracted = tar_file.extractfile(member)
                if extracted is None:
                    raise ValueError("extractfile returned None")
                frame_data = json.load(extracted)
                new_gt = build_future_trajectory_from_frame(frame_data)
                if new_gt is None:
                    result["source_gt_missing"] += 1
                    continue

                if not dry_run:
                    ctx.gt_trajectory = new_gt
                    if update_pdm and old_gt_missing:
                        ctx.pdm_trajectory = new_gt.copy()
                    save_source_scene_context(ctx, scene_context_root)
                result["updated"] += 1
            except Exception as exc:
                result["failed"] += 1
                if len(result["failures"]) < 20:
                    result["failures"].append(
                        {
                            "path": f"{tar_path}:{member.name}",
                            "error": str(exc),
                        }
                    )
    return result


def _merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    merged = _empty_result()
    failures: list[dict[str, str]] = []
    for result in results:
        for key in merged:
            if key == "failures":
                continue
            merged[key] += int(result.get(key, 0))
        failures.extend(result.get("failures") or [])
    merged["failures"] = failures[:20]
    return merged


def _run_payloads(payloads: list[dict[str, Any]], *, workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_process_tar_task(payload) for payload in payloads]

    results: list[dict[str, Any]] = []
    payload_iter = iter(payloads)
    max_pending = max(1, workers * 2)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()

        def fill_pending() -> None:
            while len(pending) < max_pending:
                try:
                    payload = next(payload_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(_process_tar_task, payload))

        fill_pending()
        while pending:
            for future in as_completed(pending):
                pending.remove(future)
                results.append(future.result())
                break
            fill_pending()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill internal SceneContext GT trajectories from shard frame.json files.",
    )
    parser.add_argument("--scene-context-root", required=True, help="Existing SceneContext cache root.")
    parser.add_argument("--shard-root", required=True, help="Internal shard tar root.")
    parser.add_argument("--workers", type=int, default=1, help="Number of shard workers.")
    parser.add_argument("--limit", type=int, default=0, help="Max frame.json records to scan.")
    parser.add_argument("--max-tars", type=int, default=0, help="Max shard tar files to scan.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count records that would be updated.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rebuild GT even when the SceneContext already has gt_trajectory.",
    )
    parser.add_argument(
        "--no-update-pdm",
        action="store_true",
        help="Do not set pdm_trajectory to the recovered GT for old GT-missing samples.",
    )
    parser.add_argument("--report-path", help="Optional JSON report output path.")
    args = parser.parse_args()

    scene_context_root = Path(args.scene_context_root)
    shard_root = Path(args.shard_root)
    tar_paths = _discover_shard_tar_paths(shard_root, max_paths=max(0, args.max_tars))
    workers = max(1, int(args.workers))

    manager = Manager() if args.limit > 0 else None
    limit_counter = manager.Value("i", 0) if manager is not None else None
    limit_lock = manager.Lock() if manager is not None else None
    payloads = [
        {
            "tar_path": str(tar_path),
            "scene_context_root": str(scene_context_root),
            "dry_run": args.dry_run,
            "overwrite_existing": args.overwrite_existing,
            "update_pdm": not args.no_update_pdm,
            "limit": args.limit,
            "limit_counter": limit_counter,
            "limit_lock": limit_lock,
        }
        for tar_path in tar_paths
    ]

    try:
        results = _run_payloads(payloads, workers=workers)
    finally:
        if manager is not None:
            manager.shutdown()

    summary = _merge_results(results)
    summary.update(
        {
            "scene_context_root": str(scene_context_root),
            "shard_root": str(shard_root),
            "workers": workers,
            "dry_run": bool(args.dry_run),
            "overwrite_existing": bool(args.overwrite_existing),
            "update_pdm": not args.no_update_pdm,
        }
    )

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

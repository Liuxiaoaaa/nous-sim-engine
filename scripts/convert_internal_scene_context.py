#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import Manager
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable

from nous_sim_engine.adapters.dataset_loader import (
    save_source_scene_context,
    write_scene_context_metadata,
)
from nous_sim_engine.adapters.internal import (
    InternalCaseRecordSceneContextBuilder,
    build_scene_context_from_frame,
    build_scene_context_from_info,
    load_frame_json,
    load_info_json,
)


SHARD_SCENE_CONTEXT_SOURCE = "internal_shard_case_record_scene_context"


def _discover_info_paths(input_root: Path) -> list[Path]:
    if input_root.is_file() and input_root.name == "info.json":
        return [input_root]
    return sorted(input_root.glob("**/info/info.json"))


def _discover_frame_paths(frame_root: Path) -> list[Path]:
    if frame_root.is_file() and frame_root.name == "frame.json":
        return [frame_root]
    return sorted(frame_root.glob("**/frame.json"))


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

    paths: list[Path] = []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("files") or data.get("shards") or []
    else:
        entries = []
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


def _scene_ids(info_path: Path, input_root: Path) -> tuple[str, str]:
    sample_dir = info_path.parent.parent
    if "__" in sample_dir.name:
        log_name, token = sample_dir.name.split("__", 1)
        return log_name, token

    rel = sample_dir.relative_to(input_root) if sample_dir.is_relative_to(input_root) else None
    if rel is not None and len(rel.parts) >= 2:
        return rel.parts[-2], rel.parts[-1]
    return sample_dir.parent.name, sample_dir.name


def _frame_member_ids(member_name: str) -> tuple[str, str]:
    parts = Path(member_name).parts
    if len(parts) >= 3 and parts[-1] == "frame.json":
        return parts[-3], parts[-2]
    raise ValueError(f"Unexpected frame member path: {member_name}")


def _frame_timestamp(frame_data: dict[str, Any]) -> float:
    raw_timestamp = frame_data.get("timestamp") or frame_data.get("image_id")
    if raw_timestamp is None:
        raise ValueError("frame_data must contain timestamp or image_id")
    return float(raw_timestamp)


def _frame_has_lanes(frame_data: dict[str, Any]) -> bool:
    lanes = (frame_data.get("map") or {}).get("lanes")
    return isinstance(lanes, list) and len(lanes) > 0


def _load_raw_info_json(
    raw_root: Path | None,
    *,
    case_id: str,
    scene_token: str,
) -> dict[str, Any] | None:
    if raw_root is None:
        return None

    zip_path = raw_root / case_id / f"{scene_token}.zip"
    if not zip_path.is_file():
        return None

    with zipfile.ZipFile(zip_path) as zip_file:
        with zip_file.open("info/info.json") as file_obj:
            data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"raw info/info.json must contain an object: {zip_path}")
    return data


def _empty_worker_result() -> dict[str, Any]:
    return {
        "cases": 0,
        "total": 0,
        "converted": 0,
        "failed": 0,
        "failures": [],
    }


def _worker_limit_claim(payload: dict[str, Any]) -> Callable[[], bool] | None:
    global_limit = int(payload.get("global_limit") or 0)
    counter = payload.get("limit_counter")
    lock = payload.get("limit_lock")
    if global_limit <= 0 or counter is None or lock is None:
        return None

    def claim() -> bool:
        with lock:
            if counter.value >= global_limit:
                return False
            counter.value += 1
            return True

    return claim


def _worker_limit_reached(payload: dict[str, Any]) -> bool:
    global_limit = int(payload.get("global_limit") or 0)
    counter = payload.get("limit_counter")
    lock = payload.get("limit_lock")
    if global_limit <= 0 or counter is None or lock is None:
        return False
    with lock:
        return counter.value >= global_limit


def _aggregate_worker_results(
    results: list[dict[str, Any]],
    *,
    source: str,
    input_root_key: str,
    input_root: Path,
    workers: int,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for result in results:
        failures.extend(result.get("failures") or [])

    return {
        "source": source,
        input_root_key: str(input_root),
        "cases": sum(int(result.get("cases", 0)) for result in results),
        "total": sum(int(result.get("total", 0)) for result in results),
        "converted": sum(int(result.get("converted", 0)) for result in results),
        "failed": sum(int(result.get("failed", 0)) for result in results),
        "failures": failures[:20],
        "workers": workers,
    }


def _collect_worker_results(
    task_payloads: list[dict[str, Any]],
    task_fn: Any,
    *,
    workers: int,
    global_limit: int = 0,
    limit_counter: Any = None,
    limit_lock: Any = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    max_pending = max(1, workers * 2)
    payload_iter = iter(task_payloads)

    def limit_reached() -> bool:
        if global_limit <= 0 or limit_counter is None or limit_lock is None:
            return False
        with limit_lock:
            return limit_counter.value >= global_limit

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()

        def fill_pending() -> None:
            while len(pending) < max_pending and not limit_reached():
                try:
                    payload = next(payload_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(task_fn, payload))

        fill_pending()
        while pending:
            for future in as_completed(pending):
                pending.remove(future)
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(
                        {
                            "cases": 0,
                            "total": 0,
                            "converted": 0,
                            "failed": 1,
                            "failures": [{"path": "<worker>", "error": str(exc)}],
                        }
                    )
                break
            fill_pending()
    return results


def _convert_case_frames(
    frames_by_case_id: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    limit: int,
    seen: int,
    raw_root: Path | None = None,
    limit_claim: Callable[[], bool] | None = None,
) -> tuple[int, int, int, int, list[dict[str, str]]]:
    builder = InternalCaseRecordSceneContextBuilder()
    converted = 0
    failed = 0
    output_case_ids: set[str] = set()
    failures: list[dict[str, str]] = []

    for case_id, case_frames in sorted(frames_by_case_id.items()):
        sorted_frames = sorted(case_frames, key=_frame_timestamp)
        for target_index, frame_data in enumerate(sorted_frames):
            if limit > 0 and seen >= limit:
                return converted, failed, seen, len(output_case_ids), failures
            if limit_claim is not None and not limit_claim():
                return converted, failed, seen, len(output_case_ids), failures
            seen += 1

            scene_token = str(frame_data.get("timestamp") or frame_data.get("image_id"))
            try:
                target_info_data = None
                if not _frame_has_lanes(frame_data):
                    target_info_data = _load_raw_info_json(
                        raw_root,
                        case_id=case_id,
                        scene_token=scene_token,
                    )
                ctx = builder.build_target(
                    sorted_frames,
                    target_index=target_index,
                    log_name=case_id,
                    target_info_data=target_info_data,
                )
                save_source_scene_context(ctx, output_dir)
                converted += 1
                output_case_ids.add(case_id)
            except Exception as exc:
                failed += 1
                failures.append(
                    {
                        "path": f"{case_id}/{scene_token}/frame.json",
                        "error": str(exc),
                    }
                )
                if len(failures) > 1000:
                    failures = failures[-1000:]

    return converted, failed, seen, len(output_case_ids), failures


def _convert_tar_path_task(payload: dict[str, Any]) -> dict[str, Any]:
    if _worker_limit_reached(payload):
        return _empty_worker_result()

    tar_path = Path(payload["tar_path"])
    output_dir = Path(payload["output_dir"])
    raw_root = Path(payload["raw_root"]) if payload.get("raw_root") else None
    target_limit = int(payload.get("limit") or 0)
    limit_claim = _worker_limit_claim(payload)
    failed = 0
    failures: list[dict[str, str]] = []
    frames_by_case_id: dict[str, list[dict[str, Any]]] = {}

    try:
        tar_file = tarfile.open(tar_path, "r:*")
    except (OSError, tarfile.TarError) as exc:
        return {
            "cases": 0,
            "total": 0,
            "converted": 0,
            "failed": 1,
            "failures": [{"path": str(tar_path), "error": str(exc)}],
        }

    with tar_file:
        for member in tar_file:
            if not member.isfile() or not member.name.endswith("/frame.json"):
                continue
            try:
                extracted = tar_file.extractfile(member)
                if extracted is None:
                    raise ValueError("extractfile returned None")
                frame_data = json.load(extracted)
                case_id, _ = _frame_member_ids(member.name)
                frames_by_case_id.setdefault(case_id, []).append(frame_data)
            except Exception as exc:
                failed += 1
                failures.append({"path": f"{tar_path}:{member.name}", "error": str(exc)})
                if len(failures) > 1000:
                    failures = failures[-1000:]

    converted, build_failed, seen, cases, build_failures = _convert_case_frames(
        frames_by_case_id,
        output_dir=output_dir,
        limit=target_limit,
        seen=0,
        raw_root=raw_root,
        limit_claim=limit_claim,
    )
    failed += build_failed
    failures.extend(build_failures)
    return {
        "cases": cases,
        "total": seen,
        "converted": converted,
        "failed": failed,
        "failures": failures[:20],
    }


def _convert_info_cache(input_root: Path, output_dir: Path, limit: int) -> dict:
    info_paths = _discover_info_paths(input_root)
    if limit > 0:
        info_paths = info_paths[:limit]

    converted = 0
    failed = 0
    failures: list[dict[str, str]] = []
    for info_path in info_paths:
        try:
            log_name, scene_token = _scene_ids(info_path, input_root)
            ctx = build_scene_context_from_info(
                load_info_json(info_path),
                log_name=log_name,
                scene_token=scene_token,
            )
            save_source_scene_context(ctx, output_dir)
            converted += 1
        except Exception as exc:
            failed += 1
            failures.append({"path": str(info_path), "error": str(exc)})

    return {
        "source": "internal_info_json",
        "input_root": str(input_root),
        "total": len(info_paths),
        "converted": converted,
        "failed": failed,
        "failures": failures[:20],
    }


def _convert_frame_cache(frame_root: Path, output_dir: Path, limit: int) -> dict:
    frame_paths = _discover_frame_paths(frame_root)
    if limit > 0:
        frame_paths = frame_paths[:limit]

    converted = 0
    failed = 0
    failures: list[dict[str, str]] = []
    for frame_path in frame_paths:
        try:
            log_name = frame_path.parent.parent.name
            scene_token = frame_path.parent.name
            ctx = build_scene_context_from_frame(
                load_frame_json(frame_path),
                log_name=log_name,
                scene_token=scene_token,
            )
            save_source_scene_context(ctx, output_dir)
            converted += 1
        except Exception as exc:
            failed += 1
            failures.append({"path": str(frame_path), "error": str(exc)})

    return {
        "source": "internal_frame_json",
        "frame_root": str(frame_root),
        "total": len(frame_paths),
        "converted": converted,
        "failed": failed,
        "failures": failures[:20],
    }


def _convert_shard_cache(
    shard_root: Path,
    output_dir: Path,
    limit: int,
    workers: int,
    raw_root: Path | None = None,
) -> dict:
    discovery_limit = max(limit, workers * 16) if limit > 0 else 0
    tar_paths = _discover_shard_tar_paths(shard_root, max_paths=discovery_limit)
    if workers > 1:
        manager = Manager() if limit > 0 else None
        limit_counter = manager.Value("i", 0) if manager is not None else None
        limit_lock = manager.Lock() if manager is not None else None
        payloads = [
            {
                "tar_path": str(tar_path),
                "output_dir": str(output_dir),
                "raw_root": str(raw_root) if raw_root is not None else "",
                "limit": 0,
                "global_limit": limit,
                "limit_counter": limit_counter,
                "limit_lock": limit_lock,
            }
            for tar_path in tar_paths
        ]
        try:
            results = _collect_worker_results(
                payloads,
                _convert_tar_path_task,
                workers=workers,
                global_limit=limit,
                limit_counter=limit_counter,
                limit_lock=limit_lock,
            )
        finally:
            if manager is not None:
                manager.shutdown()

        summary = _aggregate_worker_results(
            results,
            source=SHARD_SCENE_CONTEXT_SOURCE,
            input_root_key="shard_root",
            input_root=shard_root,
            workers=workers,
        )
        summary["tar_files"] = len(tar_paths)
        if raw_root is not None:
            summary["raw_root"] = str(raw_root)
        return summary

    converted = 0
    failed = 0
    seen = 0
    cases = 0
    failures: list[dict[str, str]] = []

    for tar_path in tar_paths:
        try:
            tar_file = tarfile.open(tar_path, "r:*")
        except (OSError, tarfile.TarError) as exc:
            failed += 1
            failures.append({"path": str(tar_path), "error": str(exc)})
            continue

        with tar_file:
            frames_by_case_id: dict[str, list[dict[str, Any]]] = {}
            for member in tar_file:
                if not member.isfile() or not member.name.endswith("/frame.json"):
                    continue
                try:
                    extracted = tar_file.extractfile(member)
                    if extracted is None:
                        raise ValueError("extractfile returned None")
                    frame_data = json.load(extracted)
                    case_id, _ = _frame_member_ids(member.name)
                    frames_by_case_id.setdefault(case_id, []).append(frame_data)
                except Exception as exc:
                    failed += 1
                    failures.append(
                        {
                            "path": f"{tar_path}:{member.name}",
                            "error": str(exc),
                        }
                    )
                    if len(failures) > 1000:
                        failures = failures[-1000:]

            case_converted, case_failed, seen, case_count, case_failures = _convert_case_frames(
                frames_by_case_id,
                output_dir=output_dir,
                limit=limit,
                seen=seen,
                raw_root=raw_root,
            )
            converted += case_converted
            failed += case_failed
            cases += case_count
            failures.extend(case_failures)
            if limit > 0 and seen >= limit:
                break

    summary = {
        "source": SHARD_SCENE_CONTEXT_SOURCE,
        "shard_root": str(shard_root),
        "tar_files": len(tar_paths),
        "cases": cases,
        "total": seen,
        "converted": converted,
        "failed": failed,
        "failures": failures[:20],
        "workers": 1,
    }
    if raw_root is not None:
        summary["raw_root"] = str(raw_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert internal data to SceneContext cache.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-root", help="Root containing */info/info.json samples.")
    input_group.add_argument("--frame-root", help="Root containing enriched */frame.json samples.")
    input_group.add_argument("--shard-root", help="Root containing internal shard tar files.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output scene_context_v1 cache directory.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max scenes to convert.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for --shard-root conversion.",
    )
    parser.add_argument(
        "--raw-root",
        help=(
            "Optional raw vlm_vehicle batch root containing <case_id>/<scene_token>.zip. "
            "Used as a map fallback for --shard-root when frame.json has empty map.lanes."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve() if args.raw_root else None

    if args.input_root:
        input_root = Path(args.input_root).expanduser().resolve()
        source_meta = {
            "source": "internal_info_json",
            "input_root": str(input_root),
        }
    elif args.frame_root:
        input_root = Path(args.frame_root).expanduser().resolve()
        source_meta = {
            "source": "internal_frame_json",
            "frame_root": str(input_root),
        }
    else:
        input_root = Path(args.shard_root).expanduser().resolve()
        source_meta = {
            "source": SHARD_SCENE_CONTEXT_SOURCE,
            "shard_root": str(input_root),
        }
        if raw_root is not None:
            source_meta["raw_root"] = str(raw_root)

    write_scene_context_metadata(
        output_dir,
        source_meta,
    )

    if args.input_root:
        summary = _convert_info_cache(input_root, output_dir, args.limit)
    elif args.frame_root:
        summary = _convert_frame_cache(input_root, output_dir, args.limit)
    else:
        summary = _convert_shard_cache(
            input_root,
            output_dir,
            args.limit,
            workers=max(1, int(args.workers)),
            raw_root=raw_root,
        )

    summary["output_dir"] = str(output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

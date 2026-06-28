#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import Manager
import tarfile
from pathlib import Path
from typing import Any, Callable

from nous_sim_engine.adapters.internal import InternalCaseRecordSceneContextBuilder, load_frame_json


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


def _frame_timestamp(frame_data: dict[str, Any]) -> float:
    raw_timestamp = frame_data.get("timestamp") or frame_data.get("image_id")
    if raw_timestamp is None:
        raise ValueError("frame_data must contain timestamp or image_id")
    return float(raw_timestamp)


def _case_id_from_frame_path(frame_path: Path, frame_data: dict[str, Any]) -> str:
    return str(frame_data.get("case_id") or frame_path.parent.parent.name)


def _scene_token_from_frame(frame_data: dict[str, Any], fallback: str) -> str:
    raw_token = frame_data.get("timestamp") or frame_data.get("image_id") or fallback
    return str(raw_token)


def _write_enriched_frame(
    frame_data: dict[str, Any],
    *,
    output_root: Path,
    case_id: str,
    scene_token: str,
    overwrite: bool,
) -> bool:
    output_path = output_root / case_id / scene_token / "frame.json"
    if output_path.exists() and not overwrite:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(frame_data, file, ensure_ascii=False, separators=(",", ":"))
        file.write("\n")
    return True


def _write_metadata(
    *,
    output_root: Path,
    source_meta: dict[str, Any],
    horizon_seconds: float,
    interval_time: float,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "internal_enriched_frame_json_v1",
        "future_obstacle_tracks": {
            "horizon_seconds": horizon_seconds,
            "interval_time": interval_time,
            "coordinate_frame": "nous_ego_current_frame",
        },
        **source_meta,
    }
    with (output_root / "_enrichment_meta.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _enrich_cases(
    frames_by_case_id: dict[str, list[dict[str, Any]]],
    *,
    builder: InternalCaseRecordSceneContextBuilder,
    output_root: Path,
    limit: int,
    overwrite: bool,
    seen: int,
    limit_claim: Callable[[], bool] | None = None,
) -> tuple[int, int, int, int, list[dict[str, str]]]:
    converted = 0
    skipped = 0
    output_case_ids: set[str] = set()
    failures: list[dict[str, str]] = []

    for case_id, case_frames in sorted(frames_by_case_id.items()):
        sorted_frames = sorted(case_frames, key=_frame_timestamp)
        for target_index, frame_data in enumerate(sorted_frames):
            if limit > 0 and seen >= limit:
                return converted, skipped, seen, len(output_case_ids), failures
            if limit_claim is not None and not limit_claim():
                return converted, skipped, seen, len(output_case_ids), failures
            seen += 1

            scene_token = _scene_token_from_frame(frame_data, str(target_index))
            try:
                enriched_frame = builder.enrich_frame(sorted_frames, target_index=target_index)
                wrote = _write_enriched_frame(
                    enriched_frame,
                    output_root=output_root,
                    case_id=case_id,
                    scene_token=scene_token,
                    overwrite=overwrite,
                )
                if wrote:
                    converted += 1
                else:
                    skipped += 1
                output_case_ids.add(case_id)
            except Exception as exc:
                failures.append(
                    {
                        "path": f"{case_id}/{scene_token}/frame.json",
                        "error": str(exc),
                    }
                )
                if len(failures) > 1000:
                    failures = failures[-1000:]

    return converted, skipped, seen, len(output_case_ids), failures


def _empty_worker_result() -> dict[str, Any]:
    return {
        "cases": 0,
        "total": 0,
        "converted": 0,
        "skipped": 0,
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
        "skipped": sum(int(result.get("skipped", 0)) for result in results),
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
                            "skipped": 0,
                            "failed": 1,
                            "failures": [{"path": "<worker>", "error": str(exc)}],
                        }
                    )
                break
            fill_pending()
    return results


def _enrich_frame_path_group_task(payload: dict[str, Any]) -> dict[str, Any]:
    if _worker_limit_reached(payload):
        return _empty_worker_result()

    frame_paths = [Path(path) for path in payload["frame_paths"]]
    output_root = Path(payload["output_root"])
    builder = InternalCaseRecordSceneContextBuilder(
        horizon_seconds=float(payload["horizon_seconds"]),
        interval_time=float(payload["interval_time"]),
    )

    frames_by_case_id: dict[str, list[dict[str, Any]]] = {}
    failed = 0
    failures: list[dict[str, str]] = []
    target_limit = int(payload.get("limit") or 0)
    limit_claim = _worker_limit_claim(payload)
    for frame_path in frame_paths:
        try:
            frame_data = load_frame_json(frame_path)
            case_id = _case_id_from_frame_path(frame_path, frame_data)
            frames_by_case_id.setdefault(case_id, []).append(frame_data)
        except Exception as exc:
            failed += 1
            failures.append({"path": str(frame_path), "error": str(exc)})

    converted, skipped, seen, output_cases, enrich_failures = _enrich_cases(
        frames_by_case_id,
        builder=builder,
        output_root=output_root,
        limit=target_limit,
        overwrite=bool(payload["overwrite"]),
        seen=0,
        limit_claim=limit_claim,
    )
    failed += len(enrich_failures)
    failures.extend(enrich_failures)
    return {
        "cases": output_cases,
        "total": seen,
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "failures": failures[:20],
    }


def _enrich_tar_path_task(payload: dict[str, Any]) -> dict[str, Any]:
    if _worker_limit_reached(payload):
        return _empty_worker_result()

    tar_path = Path(payload["tar_path"])
    output_root = Path(payload["output_root"])
    builder = InternalCaseRecordSceneContextBuilder(
        horizon_seconds=float(payload["horizon_seconds"]),
        interval_time=float(payload["interval_time"]),
    )

    failed = 0
    failures: list[dict[str, str]] = []
    frames_by_case_id: dict[str, list[dict[str, Any]]] = {}
    target_limit = int(payload.get("limit") or 0)
    limit_claim = _worker_limit_claim(payload)
    try:
        tar_file = tarfile.open(tar_path, "r:*")
    except (OSError, tarfile.TarError) as exc:
        return {
            "cases": 0,
            "total": 0,
            "converted": 0,
            "skipped": 0,
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

    converted, skipped, seen, output_cases, enrich_failures = _enrich_cases(
        frames_by_case_id,
        builder=builder,
        output_root=output_root,
        limit=target_limit,
        overwrite=bool(payload["overwrite"]),
        seen=0,
        limit_claim=limit_claim,
    )
    failed += len(enrich_failures)
    failures.extend(enrich_failures)
    return {
        "cases": output_cases,
        "total": seen,
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "failures": failures[:20],
    }


def _enrich_frame_root(
    frame_root: Path,
    output_root: Path,
    *,
    builder: InternalCaseRecordSceneContextBuilder,
    limit: int,
    overwrite: bool,
    workers: int,
) -> dict[str, Any]:
    if workers > 1:
        frame_paths_by_case_dir: dict[str, list[Path]] = {}
        for frame_path in _discover_frame_paths(frame_root):
            case_dir = str(frame_path.parent.parent)
            frame_paths_by_case_dir.setdefault(case_dir, []).append(frame_path)

        payloads: list[dict[str, Any]] = []
        manager = Manager() if limit > 0 else None
        limit_counter = manager.Value("i", 0) if manager is not None else None
        limit_lock = manager.Lock() if manager is not None else None
        for _, frame_paths in sorted(frame_paths_by_case_dir.items()):
            sorted_frame_paths = sorted(frame_paths)
            payloads.append(
                {
                    "frame_paths": [str(path) for path in sorted_frame_paths],
                    "output_root": str(output_root),
                    "horizon_seconds": builder.horizon_seconds,
                    "interval_time": builder.interval_time,
                    "overwrite": overwrite,
                    "limit": 0,
                    "global_limit": limit,
                    "limit_counter": limit_counter,
                    "limit_lock": limit_lock,
                }
            )
        results: list[dict[str, Any]] = []
        try:
            results = _collect_worker_results(
                payloads,
                _enrich_frame_path_group_task,
                workers=workers,
                global_limit=limit,
                limit_counter=limit_counter,
                limit_lock=limit_lock,
            )
        finally:
            if manager is not None:
                manager.shutdown()

        return _aggregate_worker_results(
            results,
            source="internal_frame_json",
            input_root_key="frame_root",
            input_root=frame_root,
            workers=workers,
        )

    frames_by_case_id: dict[str, list[dict[str, Any]]] = {}
    failed = 0
    failures: list[dict[str, str]] = []

    for frame_path in _discover_frame_paths(frame_root):
        try:
            frame_data = load_frame_json(frame_path)
            case_id = _case_id_from_frame_path(frame_path, frame_data)
            frames_by_case_id.setdefault(case_id, []).append(frame_data)
        except Exception as exc:
            failed += 1
            failures.append({"path": str(frame_path), "error": str(exc)})

    converted, skipped, seen, output_cases, enrich_failures = _enrich_cases(
        frames_by_case_id,
        builder=builder,
        output_root=output_root,
        limit=limit,
        overwrite=overwrite,
        seen=0,
    )
    failed += len(enrich_failures)
    failures.extend(enrich_failures)
    return {
        "source": "internal_frame_json",
        "frame_root": str(frame_root),
        "cases": output_cases,
        "total": seen,
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "failures": failures[:20],
        "workers": 1,
    }


def _enrich_shard_root(
    shard_root: Path,
    output_root: Path,
    *,
    builder: InternalCaseRecordSceneContextBuilder,
    limit: int,
    overwrite: bool,
    workers: int,
) -> dict[str, Any]:
    discovery_limit = max(limit, workers * 16) if limit > 0 else 0
    tar_paths = _discover_shard_tar_paths(shard_root, max_paths=discovery_limit)
    if workers > 1:
        payloads: list[dict[str, Any]] = []
        manager = Manager() if limit > 0 else None
        limit_counter = manager.Value("i", 0) if manager is not None else None
        limit_lock = manager.Lock() if manager is not None else None
        for tar_path in tar_paths:
            payloads.append(
                {
                    "tar_path": str(tar_path),
                    "output_root": str(output_root),
                    "horizon_seconds": builder.horizon_seconds,
                    "interval_time": builder.interval_time,
                    "overwrite": overwrite,
                    "limit": 0,
                    "global_limit": limit,
                    "limit_counter": limit_counter,
                    "limit_lock": limit_lock,
                }
            )
        results: list[dict[str, Any]] = []
        try:
            results = _collect_worker_results(
                payloads,
                _enrich_tar_path_task,
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
            source="internal_shard_frame_json",
            input_root_key="shard_root",
            input_root=shard_root,
            workers=workers,
        )
        summary["tar_files"] = len(tar_paths)
        return summary

    converted = 0
    skipped = 0
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
                    failures.append({"path": f"{tar_path}:{member.name}", "error": str(exc)})
                    if len(failures) > 1000:
                        failures = failures[-1000:]

        case_converted, case_skipped, seen, case_count, enrich_failures = _enrich_cases(
            frames_by_case_id,
            builder=builder,
            output_root=output_root,
            limit=limit,
            overwrite=overwrite,
            seen=seen,
        )
        converted += case_converted
        skipped += case_skipped
        cases += case_count
        failed += len(enrich_failures)
        failures.extend(enrich_failures)
        if limit > 0 and seen >= limit:
            break

    return {
        "source": "internal_shard_frame_json",
        "shard_root": str(shard_root),
        "tar_files": len(tar_paths),
        "cases": cases,
        "total": seen,
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
        "failures": failures[:20],
        "workers": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich internal frame.json files with case-level future obstacle tracks.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--frame-root", help="Root containing unpacked */frame.json files.")
    input_group.add_argument("--shard-root", help="Root containing internal shard tar files.")
    parser.add_argument("--output-root", required=True, help="Output root for enriched frame.json.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max frames to write.")
    parser.add_argument("--horizon-seconds", type=float, default=4.0)
    parser.add_argument("--interval-time", type=float, default=0.1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes. Limit is split across case/tar tasks.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frame.json.")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    workers = max(1, int(args.workers))
    builder = InternalCaseRecordSceneContextBuilder(
        horizon_seconds=args.horizon_seconds,
        interval_time=args.interval_time,
    )

    if args.frame_root:
        input_root = Path(args.frame_root).expanduser().resolve()
        summary = _enrich_frame_root(
            input_root,
            output_root,
            builder=builder,
            limit=args.limit,
            overwrite=args.overwrite,
            workers=workers,
        )
    else:
        input_root = Path(args.shard_root).expanduser().resolve()
        summary = _enrich_shard_root(
            input_root,
            output_root,
            builder=builder,
            limit=args.limit,
            overwrite=args.overwrite,
            workers=workers,
        )

    _write_metadata(
        output_root=output_root,
        source_meta=summary,
        horizon_seconds=args.horizon_seconds,
        interval_time=args.interval_time,
    )
    summary["output_root"] = str(output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

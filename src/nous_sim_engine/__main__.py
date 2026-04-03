from __future__ import annotations

import argparse
import os


def _default_workers() -> int:
    return min(4, os.cpu_count() or 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nous-sim-engine HTTP service.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Register a dataset: name=path. Can be specified multiple times. "
        "Example: --dataset navtest=/data/metric_cache_navtest",
    )
    parser.add_argument(
        "--metric-cache-dir",
        default=None,
        help="(Deprecated) NavSim MetricCache source directory. "
        "Equivalent to --dataset default=PATH.",
    )
    parser.add_argument(
        "--boost-cache-dir",
        default=None,
        help="Boost cache directory for pre-converted SceneContext pickles (e.g. on SSD). "
        "Enables background warmup on startup when combined with --metric-cache-dir.",
    )
    parser.add_argument(
        "--warmup-workers",
        type=int,
        default=32,
        help="Number of parallel threads for boost cache warmup (default: 32)",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    # Validate --dataset format
    for entry in args.dataset:
        if "=" not in entry:
            parser.error(f"--dataset must be NAME=PATH, got: {entry}")

    return args


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "FastAPI server dependencies are not installed. Install nous-sim-engine[server]."
        ) from exc

    try:
        from nous_sim_engine.server import app
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "FastAPI server dependencies are not installed. Install nous-sim-engine[server]."
        ) from exc

    args = _parse_args()

    # Build SIM_ENGINE_DATASETS env var for multi-worker mode
    datasets_parts = list(args.dataset)  # already "name=path" strings
    if args.metric_cache_dir:
        # Backward compat: register as "default"
        datasets_parts.append(f"default={args.metric_cache_dir}")
        os.environ["SIM_ENGINE_METRIC_CACHE_DIR"] = args.metric_cache_dir

    if datasets_parts:
        os.environ["SIM_ENGINE_DATASETS"] = ",".join(datasets_parts)

    if args.boost_cache_dir:
        os.environ["SIM_ENGINE_BOOST_CACHE_DIR"] = args.boost_cache_dir
    os.environ["SIM_ENGINE_WARMUP_WORKERS"] = str(args.warmup_workers)

    uvicorn_app = app if args.workers == 1 else "nous_sim_engine.server.app:app"
    uvicorn.run(
        uvicorn_app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

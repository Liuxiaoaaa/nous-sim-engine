from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional, Sequence

from fastapi import FastAPI, HTTPException, Request

from nous_sim_engine import __version__
from nous_sim_engine.adapters.navsim.cache_loader import (
    get_boost_cache_dir,
    get_warmup_stats,
    load_scene_context,
    set_boost_cache_dir,
    warmup_boost_cache,
)
from nous_sim_engine.core.scoring import PDMScorerV1, PDMScorerV2, RLScorer
from nous_sim_engine.core.scoring.base import RLScorerConfig
from nous_sim_engine.core.types import RLScoringResult, ScoringResult

from .registry import DatasetRegistry
from .schemas import (
    BatchControlScoreRequest,
    BatchRLScoreRequest,
    BatchRLScoreResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    ControlScoreRequest,
    DatasetListResponse,
    DatasetRegisterRequest,
    HealthResponse,
    RLConfigOverrides,
    RLScoreRequest,
    RLScoreResponse,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger(__name__)


def _result_to_response(result: ScoringResult) -> ScoreResponse:
    return ScoreResponse.from_result(result)


def _rl_result_to_response(result: RLScoringResult) -> RLScoreResponse:
    return RLScoreResponse.from_result(result)


def _error_result(message: str) -> ScoreResponse:
    return _result_to_response(ScoringResult(error=message))


def _error_results(message: str, batch_size: int) -> BatchScoreResponse:
    return BatchScoreResponse(results=[_error_result(message) for _ in range(batch_size)])


def _rl_error_result(message: str) -> RLScoreResponse:
    return _rl_result_to_response(RLScoringResult(error=message))


def _rl_error_results(message: str, batch_size: int) -> BatchRLScoreResponse:
    return BatchRLScoreResponse(results=[_rl_error_result(message) for _ in range(batch_size)])


def _cache_stats() -> dict[str, int]:
    info = load_scene_context.cache_info()
    return {
        "size": info.currsize,
        "maxsize": 0 if info.maxsize is None else info.maxsize,
        "hits": info.hits,
        "misses": info.misses,
    }


def _build_rl_config(
    scoring_mode: str,
    overrides: RLConfigOverrides | None,
) -> RLScorerConfig:
    """Build RLScorerConfig from scoring_mode and optional per-request overrides."""
    from dataclasses import asdict
    base_config = RLScorerConfig.v1()
    kwargs: dict = {**asdict(base_config), "safety_mode": scoring_mode}
    if overrides is not None:
        for field, value in overrides.model_dump(exclude_none=True).items():
            kwargs[field] = value
    return RLScorerConfig(**kwargs)


def _resolve_dataset(registry: DatasetRegistry, dataset: str) -> str:
    """Resolve dataset name to metric_cache_dir path, or raise 404."""
    try:
        return registry.resolve(dataset)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset}' not registered. "
            f"Available: {list(registry.list_all().keys())}",
        )


def _score_batch(
    trajectories: Sequence[Sequence[Sequence[float]]],
    scene_token: str,
    log_name: str,
    metric_cache_dir: str,
    scoring_version: str = "v1",
    include_ego: bool = False,
    scorer_v1: PDMScorerV1 | None = None,
    scorer_v2: PDMScorerV2 | None = None,
) -> BatchScoreResponse:
    batch_trajectories = list(trajectories)
    try:
        scene = load_scene_context(
            cache_dir=metric_cache_dir,
            log_name=log_name,
            token=scene_token,
        )
        if scoring_version == "v2":
            scorer = scorer_v2 or PDMScorerV2()
            results = scorer.score_batch(trajectories_xy=batch_trajectories, scene=scene, include_ego=include_ego)
        else:
            scorer = scorer_v1 or PDMScorerV1()
            results = scorer.score_batch(trajectories_xy=batch_trajectories, scene=scene, include_ego=include_ego)
        return BatchScoreResponse(results=[_result_to_response(r) for r in results])
    except Exception as exc:
        return _error_results(str(exc), batch_size=len(batch_trajectories))


def _score_batch_controls(
    control_signals_batch: Sequence[Sequence[Sequence[float]]],
    scene_token: str,
    log_name: str,
    metric_cache_dir: str,
    scoring_version: str = "v1",
    scorer_v1: PDMScorerV1 | None = None,
) -> BatchScoreResponse:
    """Score from direct control signals (bypass LQR controller)."""
    import numpy as np
    batch = list(control_signals_batch)
    try:
        scene = load_scene_context(
            cache_dir=metric_cache_dir,
            log_name=log_name,
            token=scene_token,
        )
        scorer = scorer_v1 or PDMScorerV1()
        controls = np.array(batch, dtype=np.float64)
        results = scorer.score_batch_from_controls(control_signals=controls, scene=scene)
        return BatchScoreResponse(results=[_result_to_response(r) for r in results])
    except Exception as exc:
        return _error_results(str(exc), batch_size=len(batch))


def _score_batch_rl(
    trajectories: Sequence[Sequence[Sequence[float]]],
    scene_token: str,
    log_name: str,
    metric_cache_dir: str,
    rl_config: RLScorerConfig,
    include_ego: bool = False,
    scorer_rl: RLScorer | None = None,
) -> BatchRLScoreResponse:
    batch_trajectories = list(trajectories)
    try:
        scene = load_scene_context(
            cache_dir=metric_cache_dir,
            log_name=log_name,
            token=scene_token,
        )
        scorer = scorer_rl or RLScorer()
        results = scorer.score_batch(
            trajectories_xy=batch_trajectories, scene=scene, rl_config=rl_config, include_ego=include_ego,
        )
        return BatchRLScoreResponse(results=[_rl_result_to_response(r) for r in results])
    except Exception as exc:
        return _rl_error_results(str(exc), batch_size=len(batch_trajectories))


def _init_registry() -> DatasetRegistry:
    """Initialize DatasetRegistry from environment variables."""
    registry = DatasetRegistry()

    # SIM_ENGINE_DATASETS: "name=path,name=path"
    datasets_env = os.environ.get("SIM_ENGINE_DATASETS", "")
    if datasets_env:
        for entry in datasets_env.split(","):
            entry = entry.strip()
            if "=" in entry:
                name, path = entry.split("=", 1)
                registry.register(name.strip(), path.strip())
                logger.info("Registered dataset: %s → %s", name.strip(), path.strip())

    # Backward compat: SIM_ENGINE_METRIC_CACHE_DIR → "default" dataset
    legacy_dir = os.environ.get("SIM_ENGINE_METRIC_CACHE_DIR")
    if legacy_dir and "default" not in registry:
        registry.register("default", legacy_dir)
        logger.info("Registered legacy metric_cache_dir as 'default': %s", legacy_dir)

    return registry


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Pre-create scorer instances (shared across requests).
    app.state.scorer_v1 = PDMScorerV1()
    app.state.scorer_v2 = PDMScorerV2()
    app.state.scorer_rl = RLScorer()
    app.state.registry = _init_registry()

    # Boost cache: background warmup if configured
    boost_dir = os.environ.get("SIM_ENGINE_BOOST_CACHE_DIR")
    source_dir = os.environ.get("SIM_ENGINE_METRIC_CACHE_DIR")
    warmup_workers = int(os.environ.get("SIM_ENGINE_WARMUP_WORKERS", "32"))

    if boost_dir:
        set_boost_cache_dir(boost_dir)
        logger.info("Boost cache enabled: %s", boost_dir)

        source_dirs = list(app.state.registry.list_all().values())
        if source_dir and source_dir not in source_dirs:
            source_dirs.append(source_dir)

        for src in source_dirs:
            logger.info("Starting background warmup from %s (%d workers)", src, warmup_workers)
            thread = threading.Thread(
                target=warmup_boost_cache,
                args=(src, boost_dir, warmup_workers),
                daemon=True,
            )
            thread.start()

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="nous-sim-engine", version=__version__, lifespan=_lifespan)

    # ── PDM Scoring (NavSim-compatible) ─────────────────────────────────

    @app.post("/v1/score/batch", response_model=BatchScoreResponse)
    def score_batch(payload: BatchScoreRequest, request: Request) -> BatchScoreResponse:
        cache_dir = _resolve_dataset(request.app.state.registry, payload.dataset)
        return _score_batch(
            trajectories=payload.trajectories,
            scene_token=payload.scene_token,
            log_name=payload.log_name,
            metric_cache_dir=cache_dir,
            scoring_version=payload.scoring_version,
            include_ego=payload.include_ego,
            scorer_v1=request.app.state.scorer_v1,
            scorer_v2=request.app.state.scorer_v2,
        )

    @app.post("/v1/score", response_model=ScoreResponse)
    def score(payload: ScoreRequest, request: Request) -> ScoreResponse:
        cache_dir = _resolve_dataset(request.app.state.registry, payload.dataset)
        batch_response = _score_batch(
            trajectories=[payload.trajectory],
            scene_token=payload.scene_token,
            log_name=payload.log_name,
            metric_cache_dir=cache_dir,
            scoring_version=payload.scoring_version,
            include_ego=payload.include_ego,
            scorer_v1=request.app.state.scorer_v1,
            scorer_v2=request.app.state.scorer_v2,
        )
        return batch_response.results[0]

    # ── Control Signal Scoring (bypass LQR) ────────────────────────────

    @app.post("/v1/score/control/batch", response_model=BatchScoreResponse)
    def score_control_batch(payload: BatchControlScoreRequest, request: Request) -> BatchScoreResponse:
        cache_dir = _resolve_dataset(request.app.state.registry, payload.dataset)
        return _score_batch_controls(
            control_signals_batch=payload.control_signals_batch,
            scene_token=payload.scene_token,
            log_name=payload.log_name,
            metric_cache_dir=cache_dir,
            scoring_version=payload.scoring_version,
            scorer_v1=request.app.state.scorer_v1,
        )

    @app.post("/v1/score/control", response_model=ScoreResponse)
    def score_control(payload: ControlScoreRequest, request: Request) -> ScoreResponse:
        cache_dir = _resolve_dataset(request.app.state.registry, payload.dataset)
        batch_response = _score_batch_controls(
            control_signals_batch=[payload.control_signals],
            scene_token=payload.scene_token,
            log_name=payload.log_name,
            metric_cache_dir=cache_dir,
            scoring_version=payload.scoring_version,
            scorer_v1=request.app.state.scorer_v1,
        )
        return batch_response.results[0]

    # ── RL Scoring (continuous / discrete) ──────────────────────────────

    @app.post("/v1/score/rl/batch", response_model=BatchRLScoreResponse)
    def score_rl_batch(payload: BatchRLScoreRequest, request: Request) -> BatchRLScoreResponse:
        cache_dir = _resolve_dataset(request.app.state.registry, payload.dataset)
        rl_config = _build_rl_config(payload.scoring_mode, payload.config_overrides)
        return _score_batch_rl(
            trajectories=payload.trajectories,
            scene_token=payload.scene_token,
            log_name=payload.log_name,
            metric_cache_dir=cache_dir,
            rl_config=rl_config,
            include_ego=payload.include_ego,
            scorer_rl=request.app.state.scorer_rl,
        )

    @app.post("/v1/score/rl", response_model=RLScoreResponse)
    def score_rl(payload: RLScoreRequest, request: Request) -> RLScoreResponse:
        cache_dir = _resolve_dataset(request.app.state.registry, payload.dataset)
        rl_config = _build_rl_config(payload.scoring_mode, payload.config_overrides)
        batch_response = _score_batch_rl(
            trajectories=[payload.trajectory],
            scene_token=payload.scene_token,
            log_name=payload.log_name,
            metric_cache_dir=cache_dir,
            rl_config=rl_config,
            include_ego=payload.include_ego,
            scorer_rl=request.app.state.scorer_rl,
        )
        return batch_response.results[0]

    # ── Dataset Management ──────────────────────────────────────────────

    @app.get("/v1/datasets", response_model=DatasetListResponse)
    def list_datasets(request: Request) -> DatasetListResponse:
        return DatasetListResponse(datasets=request.app.state.registry.list_all())

    @app.post("/v1/datasets", status_code=201)
    def register_dataset(payload: DatasetRegisterRequest, request: Request) -> dict:
        request.app.state.registry.register(payload.name, payload.path)
        logger.info("Registered dataset: %s → %s", payload.name, payload.path)
        return {"status": "registered", "name": payload.name, "path": payload.path}

    @app.delete("/v1/datasets/{name}")
    def unregister_dataset(name: str, request: Request) -> dict:
        try:
            request.app.state.registry.unregister(name)
            logger.info("Unregistered dataset: %s", name)
            return {"status": "unregistered", "name": name}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Dataset '{name}' not found")

    # ── Health ──────────────────────────────────────────────────────────

    @app.get("/v1/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        boost_info: Optional[dict] = None
        boost_dir = get_boost_cache_dir()
        if boost_dir is not None:
            stats = get_warmup_stats()
            total = stats.get("total", 0)
            converted = stats.get("converted", 0)
            boost_info = {
                "enabled": True,
                "dir": boost_dir,
                "converted": converted,
                "total": total,
                "progress_pct": round(converted / total * 100, 1) if total > 0 else 0.0,
            }
        return HealthResponse(
            status="ok",
            version=__version__,
            cache_stats=_cache_stats(),
            datasets=request.app.state.registry.list_all(),
            boost_cache=boost_info,
        )

    return app


app = create_app()

__all__ = ["app", "create_app"]

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from nous_sim_engine.core.types import RLScoringResult, ScoringResult


class ScoreRequest(BaseModel):
    trajectory: List[List[float]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"


class BatchScoreRequest(BaseModel):
    trajectories: List[List[List[float]]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"


class ScoreResponse(BaseModel):
    scoring_version: str = "v1"
    pdm_score: float = 0.0
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0
    ego_progress: float = 0.0
    time_to_collision: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0
    error: str | None = None

    @classmethod
    def from_result(cls, result: ScoringResult) -> "ScoreResponse":
        return cls(
            scoring_version=result.scoring_version,
            pdm_score=result.pdm_score,
            no_at_fault_collisions=result.no_at_fault_collisions,
            drivable_area_compliance=result.drivable_area_compliance,
            driving_direction_compliance=result.driving_direction_compliance,
            traffic_light_compliance=result.traffic_light_compliance,
            ego_progress=result.ego_progress,
            time_to_collision=result.time_to_collision,
            lane_keeping=result.lane_keeping,
            history_comfort=result.history_comfort,
            error=result.error,
        )


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]


# ── RL Scoring Schemas ──────────────────────────────────────────────────


class RLConfigOverrides(BaseModel):
    """Optional per-request overrides for RLScorerConfig fields."""

    # Active metrics
    nc_weight: Optional[float] = None
    dac_weight: Optional[float] = None
    ep_weight: Optional[float] = None
    ttc_weight: Optional[float] = None
    hc_weight: Optional[float] = None
    # Inactive in v1 by default (still overridable)
    ddc_weight: Optional[float] = None
    tlc_weight: Optional[float] = None
    lk_weight: Optional[float] = None
    collision_distance_scale: Optional[float] = None
    dac_margin: Optional[float] = None
    tlc_margin: Optional[float] = None
    progress_distance_threshold: Optional[float] = None
    ttc_horizon: Optional[float] = None
    lane_keeping_deviation: Optional[float] = None
    lane_keeping_max_deviation: Optional[float] = None


class RLScoreRequest(BaseModel):
    trajectory: List[List[float]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_mode: Literal["continuous", "discrete"] = "continuous"
    config_overrides: Optional[RLConfigOverrides] = None


class BatchRLScoreRequest(BaseModel):
    trajectories: List[List[List[float]]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_mode: Literal["continuous", "discrete"] = "continuous"
    config_overrides: Optional[RLConfigOverrides] = None


class RLScoreResponse(BaseModel):
    rl_score: float = 0.0
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0
    ego_progress: float = 0.0
    time_to_collision: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0
    sub_rewards: Dict[str, float] = {}
    error: str | None = None

    @classmethod
    def from_result(cls, result: RLScoringResult) -> "RLScoreResponse":
        return cls(
            rl_score=result.rl_score,
            no_at_fault_collisions=result.no_at_fault_collisions,
            drivable_area_compliance=result.drivable_area_compliance,
            driving_direction_compliance=result.driving_direction_compliance,
            traffic_light_compliance=result.traffic_light_compliance,
            ego_progress=result.ego_progress,
            time_to_collision=result.time_to_collision,
            lane_keeping=result.lane_keeping,
            history_comfort=result.history_comfort,
            sub_rewards=result.sub_rewards(),
            error=result.error,
        )


class BatchRLScoreResponse(BaseModel):
    results: List[RLScoreResponse]


# ── Dataset Management Schemas ──────────────────────────────────────────


class DatasetRegisterRequest(BaseModel):
    name: str
    path: str


class DatasetListResponse(BaseModel):
    datasets: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    version: str
    cache_stats: dict[str, int]
    datasets: Dict[str, str] = {}
    boost_cache: Optional[dict] = None

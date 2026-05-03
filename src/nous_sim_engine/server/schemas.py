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
    include_ego: bool = False


class BatchScoreRequest(BaseModel):
    trajectories: List[List[List[float]]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"
    include_ego: bool = False


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


# ── Control Signal Scoring Schemas ─────────────────────────────────────


class ControlScoreRequest(BaseModel):
    """Score from direct control signals, bypassing LQR controller."""
    control_signals: List[List[float]]  # 8 × [accel_m/s², heading_rate_rad/s]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"
    include_ego: bool = False


class BatchControlScoreRequest(BaseModel):
    control_signals_batch: List[List[List[float]]]  # B × 8 × [accel, heading_rate]
    scene_token: str
    log_name: str
    dataset: str
    scoring_version: str = "v1"
    include_ego: bool = False


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
    include_ego: bool = False


class BatchRLScoreRequest(BaseModel):
    trajectories: List[List[List[float]]]
    scene_token: str
    log_name: str
    dataset: str
    scoring_mode: Literal["continuous", "discrete"] = "continuous"
    config_overrides: Optional[RLConfigOverrides] = None
    include_ego: bool = False


def _round2(value: float) -> float:
    return round(float(value), 2)


def _round2_list(values: list[float]) -> list[float]:
    return [_round2(v) for v in values]


def _round2_points(points: list[list[float]]) -> list[list[float]]:
    return [[_round2(x), _round2(y)] for x, y in points]


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
    max_collision_overlap: float = 0.0
    max_collision_penetration_distance: float = 0.0
    min_obstacle_distance: float = 0.0
    min_boundary_distance: float = 0.0
    mean_obstacle_distance_5m: float = 5.0
    half_lane_width: float = 2.0
    lateral_offset_signed: float = 0.0
    lateral_offset_change: float = 0.0
    centerline_lateral_offset_start_signed: float = 0.0
    centerline_lateral_offset_end_signed: float = 0.0
    centerline_distance_mean: float = 0.0
    centerline_distance_max: float = 0.0
    local_centerline_points: List[List[float]] = []
    boundary_distance_start: float = 0.0
    boundary_distance_end: float = 0.0
    boundary_distance_min: float = 0.0
    boundary_distance_mean: float = 0.0
    boundary_distances: List[float] = []
    boundary_side: str | None = None
    in_intersection_fraction: float = 0.0
    oncoming_fraction: float = 0.0
    non_drivable_fraction: float = 0.0
    multiple_lanes_fraction: float = 0.0
    in_intersection_flags: List[bool] = []
    oncoming_flags: List[bool] = []
    non_drivable_flags: List[bool] = []
    multiple_lanes_flags: List[bool] = []
    safety_gate: float = 1.0
    raw_progress: float = 0.0
    pdms_score: float = 0.0
    sub_rewards: Dict[str, float] = {}
    error: str | None = None

    @classmethod
    def from_result(cls, result: RLScoringResult) -> "RLScoreResponse":
        return cls(
            rl_score=_round2(result.rl_score),
            no_at_fault_collisions=_round2(result.no_at_fault_collisions),
            drivable_area_compliance=_round2(result.drivable_area_compliance),
            driving_direction_compliance=_round2(result.driving_direction_compliance),
            traffic_light_compliance=_round2(result.traffic_light_compliance),
            ego_progress=_round2(result.ego_progress),
            time_to_collision=_round2(result.time_to_collision),
            lane_keeping=_round2(result.lane_keeping),
            history_comfort=_round2(result.history_comfort),
            max_collision_overlap=_round2(result.max_collision_overlap),
            max_collision_penetration_distance=_round2(result.max_collision_penetration_distance),
            min_obstacle_distance=_round2(result.min_obstacle_distance),
            min_boundary_distance=_round2(result.min_boundary_distance),
            mean_obstacle_distance_5m=_round2(result.mean_obstacle_distance_5m),
            half_lane_width=_round2(result.half_lane_width),
            lateral_offset_signed=_round2(result.lateral_offset_signed),
            lateral_offset_change=_round2(result.lateral_offset_change),
            centerline_lateral_offset_start_signed=_round2(result.centerline_lateral_offset_start_signed),
            centerline_lateral_offset_end_signed=_round2(result.centerline_lateral_offset_end_signed),
            centerline_distance_mean=_round2(result.centerline_distance_mean),
            centerline_distance_max=_round2(result.centerline_distance_max),
            local_centerline_points=_round2_points(result.local_centerline_points),
            boundary_distance_start=_round2(result.boundary_distance_start),
            boundary_distance_end=_round2(result.boundary_distance_end),
            boundary_distance_min=_round2(result.boundary_distance_min),
            boundary_distance_mean=_round2(result.boundary_distance_mean),
            boundary_distances=_round2_list(result.boundary_distances),
            boundary_side=result.boundary_side,
            in_intersection_fraction=_round2(result.in_intersection_fraction),
            oncoming_fraction=_round2(result.oncoming_fraction),
            non_drivable_fraction=_round2(result.non_drivable_fraction),
            multiple_lanes_fraction=_round2(result.multiple_lanes_fraction),
            in_intersection_flags=result.in_intersection_flags,
            oncoming_flags=result.oncoming_flags,
            non_drivable_flags=result.non_drivable_flags,
            multiple_lanes_flags=result.multiple_lanes_flags,
            safety_gate=_round2(result.safety_gate),
            raw_progress=_round2(result.raw_progress),
            pdms_score=_round2(result.pdms_score),
            sub_rewards={k: _round2(v) for k, v in result.sub_rewards().items()},
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

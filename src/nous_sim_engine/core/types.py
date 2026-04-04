from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import numpy as np

if TYPE_CHECKING:
    from .geometry import PDMPath
    from .observation import PDMObservation
    from .occupancy import DrivableMap


@dataclass
class VehicleParams:
    """Default: Chrysler Pacifica (nuPlan ego vehicle), matching recogdrive/NavSim."""

    half_length: float = 2.588  # (front_length + rear_length) / 2 = (4.049 + 1.127) / 2
    half_width: float = 1.1485  # width / 2 = 2.297 / 2
    rear_axle_to_center: float = 1.461
    wheel_base: float = 3.089


@dataclass
class SceneContext:
    scene_token: str
    log_name: str
    ego_state: np.ndarray
    ego_past_states: np.ndarray
    observation: "PDMObservation"
    drivable_area_map: "DrivableMap"
    route_lane_ids: Set[str]
    centerline: "PDMPath"
    collided_track_ids: Set[str] = field(default_factory=set)
    gt_trajectory: Optional[np.ndarray] = None  # (T, 2) ego-relative GT xy waypoints
    gt_progress: Optional[float] = None  # precomputed GT centerline progress (raw)
    gt_masked_progress: Optional[float] = None  # gt_progress × gt_NC × gt_DAC (for EP normalization)
    track_object_types: Dict[str, str] = field(default_factory=dict)  # token → "agent" | "static"


@dataclass
class ScoringResult:
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
    error: Optional[str] = None
    human_penalty_applied: Optional[List[str]] = None  # metric names overridden by human filter

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RLScoringResult:
    """RL reward scoring — 8 independent sub-rewards with additive aggregation.

    All metrics are in [0, 1] range. Safety metrics can be either continuous
    (distance-based) or discrete (NavSim-compatible) depending on config.
    """

    rl_score: float = 0.0
    # Safety layer
    no_at_fault_collisions: float = 1.0
    drivable_area_compliance: float = 1.0
    driving_direction_compliance: float = 1.0
    traffic_light_compliance: float = 1.0
    # Performance layer
    ego_progress: float = 0.0
    time_to_collision: float = 1.0
    lane_keeping: float = 1.0
    history_comfort: float = 1.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def sub_rewards(self) -> dict:
        """Return 8 independent sub-rewards for per-component GRPO advantage."""
        return {
            "nc": self.no_at_fault_collisions,
            "dac": self.drivable_area_compliance,
            "ddc": self.driving_direction_compliance,
            "tlc": self.traffic_light_compliance,
            "ep": self.ego_progress,
            "ttc": self.time_to_collision,
            "lk": self.lane_keeping,
            "hc": self.history_comfort,
        }

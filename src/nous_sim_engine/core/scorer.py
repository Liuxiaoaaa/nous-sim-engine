from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import numpy.typing as npt
import shapely.vectorized
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .comfort import ego_comfort_violation, ego_is_comfortable
from .enums import (
    BBCoordsIndex,
    CollisionType,
    DRIVABLE_LAYERS,
    EgoAreaIndex,
    MultiMetricIndex,
    SemanticMapLayer,
    StateIndex,
    WeightedMetricIndex,
)
from .geometry import coords_to_polygons, normalize_angle, state_to_coords
from .occupancy import DrivableMap, OccupancyMap, _LAYER_NAME_TO_ENUM, _normalize_layer_name
from .simulator import PDMSimulator
from .types import SceneContext, ScoringResult, RLScoringResult, VehicleParams


RED_LIGHT_TOKEN_PREFIX = "red_light"

# Official NavSim uses 5e-2 for collision stopped classification (distinct from config's 5e-3 for TTC)
_COLLISION_STOPPED_THRESHOLD = 5e-2


@dataclass
class _GTSimResult:
    """Cached GT simulation result, shared by progress normalization and human_penalty_filter."""
    progress: float
    multi_metrics: np.ndarray    # (len(MultiMetricIndex),)
    weighted_metrics: np.ndarray  # (len(WeightedMetricIndex),)


@dataclass(frozen=True)
class PDMScorerConfig:
    scoring_version: str = "v1"  # "v1" (PDMS) or "v2" (EPDMS-like)
    progress_weight: float = 5.0
    ttc_weight: float = 5.0
    lane_keeping_weight: float = 2.0
    comfort_weight: float = 2.0
    driving_direction_horizon: float = 1.0
    driving_direction_compliance_threshold: float = 2.0
    driving_direction_violation_threshold: float = 6.0
    stopped_speed_threshold: float = 5e-03
    future_collision_horizon: float = 1.0
    progress_distance_threshold: float = 5.0
    lane_keeping_deviation: float = 0.5
    lane_keeping_horizon: float = 2.0
    human_penalty_filter: bool = True

    @classmethod
    def v1(cls) -> "PDMScorerConfig":
        """NavSim v1 PDMS: (NC×DAC) × (5EP+5TTC+2C)/12"""
        return cls(scoring_version="v1", lane_keeping_weight=0.0, human_penalty_filter=False)

    @classmethod
    def v2(cls) -> "PDMScorerConfig":
        """NavSim v2 EPDMS-like: (NC×DAC×DDC×TLC) × (5EP+5TTC+2LK+2HC)/14"""
        return cls(scoring_version="v2")

    @property
    def weighted_metrics_array(self) -> npt.NDArray[np.float64]:
        weights = np.zeros(len(WeightedMetricIndex), dtype=np.float64)
        weights[WeightedMetricIndex.PROGRESS] = self.progress_weight
        weights[WeightedMetricIndex.TTC] = self.ttc_weight
        weights[WeightedMetricIndex.LANE_KEEPING] = self.lane_keeping_weight
        weights[WeightedMetricIndex.COMFORT] = self.comfort_weight
        return weights


@dataclass(frozen=True)
class RLScorerConfig:
    """RL reward scoring configuration.

    Safety metrics support 'continuous' (distance-based) or 'discrete' (NavSim-compatible) mode.
    Performance metrics are always continuous.
    Aggregation: soft safety gate × weighted performance average.
        rl_score = (NC × DAC × DDC × TLC) ^ alpha × weighted_avg(EP, TTC, LK, HC)
    """

    # Per-metric weights (performance layer only; safety uses multiplicative gate)
    ep_weight: float = 5.0
    ttc_weight: float = 5.0
    hc_weight: float = 2.0
    lk_weight: float = 0.0  # v1 inactive

    # Safety gate exponent: 1.0 = hard gate (PDMS-like), 0.5 = soft gate (default)
    safety_gate_alpha: float = 0.5

    # Safety layer mode: 'continuous' or 'discrete'
    safety_mode: str = "continuous"

    # Safety metric weights (only used for sub_rewards reporting, not aggregation)
    nc_weight: float = 5.0
    dac_weight: float = 3.0
    ddc_weight: float = 0.0  # v1 inactive
    tlc_weight: float = 0.0  # v1 inactive

    # Safety layer mode: 'continuous' or 'discrete'
    safety_mode: str = "continuous"

    # Continuous safety parameters
    collision_distance_scale: float = 2.0  # sigmoid scale in meters
    dac_margin: float = 2.0  # kept for API compatibility; unused in sweep-area DAC
    tlc_margin: float = 1.0  # red-light penetration decay distance

    # Shared thresholds
    progress_distance_threshold: float = 5.0  # fallback threshold when gt_progress unavailable
    ttc_horizon: float = 3.0  # TTC normalization upper bound in seconds
    lane_keeping_deviation: float = 0.5
    lane_keeping_max_deviation: float = 2.0
    lane_keeping_horizon: float = 2.0
    driving_direction_compliance_threshold: float = 2.0
    driving_direction_violation_threshold: float = 6.0
    stopped_speed_threshold: float = 5e-03
    future_collision_horizon: float = 1.0

    @classmethod
    def v1(cls) -> "RLScorerConfig":
        """Continuous RL aligned with NavSim v1 active metric set."""
        return cls(
            ddc_weight=0.0,
            tlc_weight=0.0,
            lk_weight=0.0,
        )

    @property
    def weights_array(self) -> npt.NDArray[np.float64]:
        """Full 8-element weight array for sub_rewards reporting."""
        return np.array(
            [
                self.nc_weight,
                self.dac_weight,
                self.ddc_weight,
                self.tlc_weight,
                self.ep_weight,
                self.ttc_weight,
                self.lk_weight,
                self.hc_weight,
            ],
            dtype=np.float64,
        )

    @property
    def performance_weights(self) -> npt.NDArray[np.float64]:
        """Weights for performance metrics only: [EP, TTC, LK, HC]."""
        return np.array(
            [self.ep_weight, self.ttc_weight, self.lk_weight, self.hc_weight],
            dtype=np.float64,
        )


class PDMScorer:
    def __init__(
        self,
        config: PDMScorerConfig | None = None,
        vehicle: VehicleParams | None = None,
        simulator: PDMSimulator | None = None,
        discretization_time: float = 0.1,
    ) -> None:
        self._config = config or PDMScorerConfig()
        self._vehicle = vehicle or VehicleParams()
        self._simulator = simulator or PDMSimulator(
            discretization_time=discretization_time,
            vehicle=self._vehicle,
        )

    def score(self, waypoints_xy: np.ndarray, scene: SceneContext) -> ScoringResult:
        return self.score_batch(waypoints_xy[None, ...], scene)[0]

    def score_batch(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
    ) -> List[ScoringResult]:
        batch_waypoints = self._coerce_trajectories(trajectories_xy)
        # Convert ego-relative → global, then interpolate in global frame (matches recogdrive)
        proposals = self._build_proposals(batch_waypoints, scene)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state,
            proposals=proposals,
            observation=scene.observation,
        )
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = self._calculate_ego_areas(ego_coords, scene)

        multi_metrics = np.ones((len(batch_waypoints), len(MultiMetricIndex)), dtype=np.float64)
        weighted_metrics = np.ones((len(batch_waypoints), len(WeightedMetricIndex)), dtype=np.float64)

        multi_metrics[:, MultiMetricIndex.NO_COLLISION] = self._no_at_fault_collision(
            simulated_states,
            ego_polygons,
            ego_areas,
            scene,
        )
        multi_metrics[:, MultiMetricIndex.DRIVABLE_AREA] = self._drivable_area_compliance(ego_areas)
        multi_metrics[:, MultiMetricIndex.DRIVING_DIRECTION] = self._driving_direction_compliance(
            ego_coords,
            ego_areas,
            scene,
        )
        multi_metrics[:, MultiMetricIndex.TRAFFIC_LIGHT] = self._traffic_light_compliance(
            ego_polygons,
            scene,
        )

        progress_raw = self._progress(ego_coords, scene)

        # GT: simulate once, share result for progress + human_penalty_filter
        gt_result = self._simulate_and_score_gt(scene)

        # Progress normalization (uses GT progress as denominator)
        gt_progress = gt_result.progress if gt_result else None
        weighted_metrics[:, WeightedMetricIndex.PROGRESS] = self._normalize_progress(
            progress_raw,
            multi_metrics,
            gt_progress=gt_progress,
        )
        weighted_metrics[:, WeightedMetricIndex.TTC] = self._time_to_collision(
            simulated_states,
            ego_coords,
            ego_areas,
            scene,
        )
        weighted_metrics[:, WeightedMetricIndex.LANE_KEEPING] = self._lane_keeping(
            ego_coords,
            ego_areas,
            scene,
        )
        weighted_metrics[:, WeightedMetricIndex.COMFORT] = self._history_comfort(
            simulated_states,
            scene,
        )

        # Human penalty filter: forgive metrics where GT also fails (only when agent has failures)
        human_penalty_names: list[str] | None = None
        if gt_result is not None and self._config.human_penalty_filter:
            has_failure = (
                (multi_metrics < 1.0).any() or (weighted_metrics < 1.0).any()
            )
            if has_failure:
                human_penalty_names = self._apply_human_penalty_from_gt(
                    multi_metrics, weighted_metrics, gt_result,
                )

        scores = self._aggregate(multi_metrics, weighted_metrics)
        results: List[ScoringResult] = []
        for proposal_idx, score in enumerate(scores):
            results.append(
                ScoringResult(
                    scoring_version=self._config.scoring_version,
                    pdm_score=float(score),
                    no_at_fault_collisions=float(
                        multi_metrics[proposal_idx, MultiMetricIndex.NO_COLLISION]
                    ),
                    drivable_area_compliance=float(
                        multi_metrics[proposal_idx, MultiMetricIndex.DRIVABLE_AREA]
                    ),
                    driving_direction_compliance=float(
                        multi_metrics[proposal_idx, MultiMetricIndex.DRIVING_DIRECTION]
                    ),
                    traffic_light_compliance=float(
                        multi_metrics[proposal_idx, MultiMetricIndex.TRAFFIC_LIGHT]
                    ),
                    ego_progress=float(weighted_metrics[proposal_idx, WeightedMetricIndex.PROGRESS]),
                    time_to_collision=float(weighted_metrics[proposal_idx, WeightedMetricIndex.TTC]),
                    lane_keeping=float(
                        weighted_metrics[proposal_idx, WeightedMetricIndex.LANE_KEEPING]
                    ),
                    history_comfort=float(weighted_metrics[proposal_idx, WeightedMetricIndex.COMFORT]),
                    human_penalty_applied=human_penalty_names,
                )
            )
        return results

    # NavSim v1 requires exactly 8 waypoints at 0.5s intervals (4s horizon).
    REQUIRED_NUM_WAYPOINTS = 8

    @staticmethod
    def _coerce_trajectories(trajectories_xy: np.ndarray) -> np.ndarray:
        trajectory_array = np.asarray(trajectories_xy, dtype=np.float64)
        if trajectory_array.ndim == 2:
            if trajectory_array.shape[1] not in (2, 3):
                raise ValueError(f"waypoints must have shape [T,2|3], got {trajectory_array.shape}")
            trajectory_array = trajectory_array[None, ...]
        elif trajectory_array.ndim != 3 or trajectory_array.shape[-1] not in (2, 3):
            raise ValueError(
                f"trajectories must have shape [B,T,2|3] or [T,2|3], got {trajectory_array.shape}"
            )

        T = trajectory_array.shape[1]
        if T != PDMScorer.REQUIRED_NUM_WAYPOINTS:
            raise ValueError(
                f"Input must be exactly {PDMScorer.REQUIRED_NUM_WAYPOINTS} waypoints "
                f"at 0.5s intervals (4s horizon), got {T} waypoints. "
                f"Expected shape: [B, {PDMScorer.REQUIRED_NUM_WAYPOINTS}, 2|3]"
            )
        return trajectory_array

    @staticmethod
    def _derive_relative_headings(waypoints_xy: np.ndarray) -> np.ndarray:
        """Estimate ego-frame waypoint headings using forward-difference (matches recogdrive).

        For i < n-1: heading[i] = atan2(wp[i+1] - wp[i])  (forward difference)
        For i == n-1: heading[n-1] = atan2(wp[n-1] - wp[n-2])  (backward difference)
        """
        # Forward differences for all but last point
        dxy = np.diff(waypoints_xy, axis=1)  # (B, n-1, 2)
        raw_fwd = np.arctan2(dxy[..., 1], dxy[..., 0])  # (B, n-1)

        # Last point uses backward difference (same as second-to-last)
        headings = np.concatenate([raw_fwd, raw_fwd[:, -1:]], axis=1)  # (B, n)

        # For near-zero displacement, carry forward the previous heading
        dist = np.linalg.norm(dxy, axis=-1)  # (B, n-1)
        dist_full = np.concatenate([dist, dist[:, -1:]], axis=1)  # (B, n)
        near_zero = dist_full < 1e-6

        if np.any(near_zero):
            # Iterative fill for rare near-zero segments
            for i in range(headings.shape[1]):
                if i > 0:
                    headings[:, i] = np.where(near_zero[:, i], headings[:, i - 1], headings[:, i])
                else:
                    headings[:, i] = np.where(near_zero[:, i], 0.0, headings[:, i])

        return headings

    @staticmethod
    def _ego_to_global(
        waypoints_xy: np.ndarray,
        ego_state: np.ndarray,
    ) -> np.ndarray:
        """Convert ego-relative (x,y[,heading]) waypoints to global (x,y,heading).

        Input:  (B, T, 2|3)  ego-relative
        Output: (B, T, 3)    global (x, y, heading)
        """
        batch_size, horizon, _ = waypoints_xy.shape
        ego_x = float(ego_state[StateIndex.X])
        ego_y = float(ego_state[StateIndex.Y])
        ego_heading = float(ego_state[StateIndex.HEADING])
        cos_h = np.cos(ego_heading)
        sin_h = np.sin(ego_heading)

        relative_xy = waypoints_xy[..., :2]
        global_xy = np.zeros((batch_size, horizon, 2), dtype=np.float64)
        global_xy[..., 0] = relative_xy[..., 0] * cos_h - relative_xy[..., 1] * sin_h + ego_x
        global_xy[..., 1] = relative_xy[..., 0] * sin_h + relative_xy[..., 1] * cos_h + ego_y

        relative_heading = (
            waypoints_xy[..., 2]
            if waypoints_xy.shape[-1] == 3
            else PDMScorer._derive_relative_headings(relative_xy)
        )
        headings = normalize_angle(relative_heading + ego_heading)

        return np.concatenate([global_xy, headings[..., None]], axis=-1)

    def _build_proposals(
        self,
        waypoints: np.ndarray,
        scene: SceneContext,
        input_interval: float = 0.5,
    ) -> np.ndarray:
        """Convert ego-relative waypoints to global proposals with interpolation.

        Matches recogdrive/NavSim flow: ego→global first, then linear interpolation
        in the global frame. This ensures heading interpolation is consistent with
        nuPlan's InterpolatedTrajectory (scipy interp1d on global x/y + AngularInterpolator
        on global heading).

        Input: (B, 8, 2|3) ego-relative at 0.5s
        Output: (B, 40, 3) global (x, y, heading) at 0.1s
        """
        ego_state = scene.ego_state

        # Step 1: ego-relative → global on coarse keyframes
        global_coarse = self._ego_to_global(waypoints, ego_state)  # (B, 8, 3)

        # Step 2: interpolate in global frame
        sim_dt = float(scene.observation.interval_time)
        if sim_dt <= 0 or abs(input_interval - sim_dt) < 1e-6:
            return global_coarse

        ratio = round(input_interval / sim_dt)  # 5
        if ratio <= 1:
            return global_coarse

        batch_size, num_coarse, _ = global_coarse.shape

        # Prepend ego pose as origin keyframe (9 keyframes total)
        ego_pose = np.array(
            [ego_state[StateIndex.X], ego_state[StateIndex.Y], ego_state[StateIndex.HEADING]],
            dtype=np.float64,
        )
        extended = np.concatenate(
            [np.broadcast_to(ego_pose, (batch_size, 1, 3)), global_coarse],
            axis=1,
        )  # (B, 9, 3)

        # Unwrap heading for linear interpolation
        extended_heading = np.unwrap(extended[..., 2], axis=1)

        num_extended = num_coarse + 1  # 9
        num_fine = (num_extended - 1) * ratio + 1  # 41

        # Vectorized piecewise linear interpolation
        alphas = np.arange(ratio, dtype=np.float64) / ratio  # (ratio,)

        # For each segment i, fine[i*ratio + j] = extended[i]*(1-alpha[j]) + extended[i+1]*alpha[j]
        start_xy = extended[:, :-1, :2]  # (B, 8, 2)
        end_xy = extended[:, 1:, :2]     # (B, 8, 2)
        start_h = extended_heading[:, :-1]  # (B, 8)
        end_h = extended_heading[:, 1:]     # (B, 8)

        # Broadcast: (B, 8, ratio, 2) = (B, 8, 1, 2) * (ratio, 1)
        a = alphas[None, None, :, None]  # (1, 1, ratio, 1)
        interp_xy = start_xy[:, :, None, :] * (1 - a) + end_xy[:, :, None, :] * a  # (B, 8, ratio, 2)
        a_h = alphas[None, None, :]  # (1, 1, ratio)
        interp_h = start_h[:, :, None] * (1 - a_h) + end_h[:, :, None] * a_h  # (B, 8, ratio)

        # Reshape to (B, 8*ratio, ...) and append last point
        fine_xy = np.concatenate(
            [interp_xy.reshape(batch_size, -1, 2), extended[:, -1:, :2]], axis=1
        )  # (B, 41, 2)
        fine_heading = np.concatenate(
            [interp_h.reshape(batch_size, -1), extended_heading[:, -1:]], axis=1
        )  # (B, 41)

        proposals = np.zeros((batch_size, num_fine, 3), dtype=np.float64)
        proposals[..., :2] = fine_xy
        proposals[..., 2] = normalize_angle(fine_heading)

        # Drop origin (t=0), keep 40 fine poses
        return proposals[:, 1:, :]

    def _waypoints_to_proposals(self, waypoints_xy: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
        """Convert ego-relative (x,y[,heading]) waypoints to global (x,y,heading) proposals.

        Used by GT trajectory scoring path where no interpolation is needed.
        """
        return self._ego_to_global(waypoints_xy, ego_state)

    def _calculate_ego_areas(self, ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
        batch_size, horizon, _, _ = ego_coords.shape
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        corners = ego_coords[:, :, :4, :]

        ego_areas = np.zeros((batch_size, horizon, len(EgoAreaIndex)), dtype=bool)

        # NavSim uses LANE + LANE_CONNECTOR for lane membership checks
        lane_membership = self._points_in_map_tokens(
            corners, scene.drivable_area_map,
            {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR},
        )
        if len(lane_membership) > 0:
            lane_corner_counts = lane_membership.sum(axis=-1)
            multiple_lane_mask = (lane_corner_counts > 0).sum(axis=0) > 1
            single_lane_mask = np.any(lane_corner_counts == 4, axis=0)
            ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES] = multiple_lane_mask & ~single_lane_mask

        # Batch point-in-polygon for centers: returns (B, T, num_layers)
        center_membership = scene.drivable_area_map.points_in_polygons(centers)

        # Official NavSim: drivable area checks ALL 4 CORNERS, not just center
        corner_membership = scene.drivable_area_map.points_in_polygons(corners)
        drivable_layers_all = list(DRIVABLE_LAYERS)
        corner_in_drivable = corner_membership[..., drivable_layers_all[0]]
        for layer in drivable_layers_all[1:]:
            corner_in_drivable = corner_in_drivable | corner_membership[..., layer]
        # (B, T, 4) bool — True if corner is in any drivable layer
        # NON_DRIVABLE if ANY corner is outside drivable area
        ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA] = ~corner_in_drivable.all(axis=-1)

        route_layers = self._points_in_route_lanes(centers, scene.drivable_area_map, scene.route_lane_ids)
        ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC] = ~route_layers

        # Precompute intersection flag (reused by DDC, LK, TTC)
        ego_areas[:, :, EgoAreaIndex.IN_INTERSECTION] = center_membership[..., SemanticMapLayer.INTERSECTION]

        return ego_areas

    def _no_at_fault_collision(
        self,
        simulated_states: np.ndarray,
        ego_polygons: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
    ) -> np.ndarray:
        scores = np.ones(len(simulated_states), dtype=np.float64)
        collided_track_ids = [set(scene.collided_track_ids) for _ in range(len(simulated_states))]

        for time_idx in range(simulated_states.shape[1]):
            occupancy_map = self._get_occupancy_map(scene, time_idx)
            if occupancy_map is None:
                continue

            for proposal_idx in range(len(simulated_states)):
                if scores[proposal_idx] == 0.0:
                    continue

                ego_polygon = ego_polygons[proposal_idx, time_idx]
                if not occupancy_map.intersects(ego_polygon)[0]:
                    continue

                for token in occupancy_map.get_colliding_tokens(ego_polygon):
                    if RED_LIGHT_TOKEN_PREFIX in token or token in collided_track_ids[proposal_idx]:
                        continue

                    track_polygon = occupancy_map[token]
                    collision_type = self._classify_collision_type(
                        simulated_states[proposal_idx, time_idx],
                        ego_polygon,
                        track_polygon,
                        token,
                        scene,
                        time_idx,
                    )
                    at_fault_score = self._collision_penalty(
                        collision_type,
                        ego_areas[proposal_idx, time_idx],
                        token,
                        scene,
                    )
                    scores[proposal_idx] = min(scores[proposal_idx], at_fault_score)
                    # Official: only add to collided list if NOT at-fault
                    if at_fault_score >= 1.0:
                        collided_track_ids[proposal_idx].add(token)

        return scores

    @staticmethod
    def _drivable_area_compliance(ego_areas: np.ndarray) -> np.ndarray:
        off_road = ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA].any(axis=1)
        scores = np.ones(len(ego_areas), dtype=np.float64)
        scores[off_road] = 0.0
        return scores

    def _driving_direction_compliance(
        self,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        oncoming_progress = np.zeros(centers.shape[:2], dtype=np.float64)
        oncoming_progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)

        # Zero out progress only where not oncoming.
        not_oncoming = ~ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_progress[not_oncoming] = 0.0

        horizon_steps = max(int(round(self._config.driving_direction_horizon / self._dt(scene))), 1)
        rolling_progress = np.zeros_like(oncoming_progress)
        for time_idx in range(oncoming_progress.shape[1]):
            start_idx = max(0, time_idx - horizon_steps)
            rolling_progress[:, time_idx] = oncoming_progress[:, start_idx : time_idx + 1].sum(axis=1)

        max_progress = rolling_progress.max(axis=1)
        scores = np.ones(len(ego_coords), dtype=np.float64)
        medium_mask = (
            max_progress >= self._config.driving_direction_compliance_threshold
        ) & (max_progress < self._config.driving_direction_violation_threshold)
        severe_mask = max_progress >= self._config.driving_direction_violation_threshold
        scores[medium_mask] = 0.5
        scores[severe_mask] = 0.0
        return scores

    def _traffic_light_compliance(self, ego_polygons: np.ndarray, scene: SceneContext) -> np.ndarray:
        scores = np.ones(len(ego_polygons), dtype=np.float64)
        for time_idx in range(ego_polygons.shape[1]):
            red_light_map = self._get_red_light_map(scene, time_idx)
            if red_light_map is None:
                continue

            for proposal_idx in range(len(ego_polygons)):
                if scores[proposal_idx] == 0.0:
                    continue

                ego_polygon = ego_polygons[proposal_idx, time_idx]
                if not red_light_map.intersects(ego_polygon)[0]:
                    continue

                for token in red_light_map.get_colliding_tokens(ego_polygon):
                    if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                        scores[proposal_idx] = 0.0
                        break
        return scores

    @staticmethod
    def _progress(ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        progress = np.zeros(len(ego_coords), dtype=np.float64)
        for proposal_idx in range(len(ego_coords)):
            start_progress = scene.centerline.project(Point(*centers[proposal_idx, 0]))
            end_progress = scene.centerline.project(Point(*centers[proposal_idx, -1]))
            progress[proposal_idx] = max(0.0, end_progress - start_progress)
        return progress

    def _simulate_and_score_gt(self, scene: SceneContext) -> _GTSimResult | None:
        """Simulate GT trajectory once, compute all metrics for progress + human_penalty_filter."""
        if scene.gt_trajectory is None:
            return None

        gt_trajectory = np.asarray(scene.gt_trajectory, dtype=np.float64)
        if gt_trajectory.ndim != 2 or gt_trajectory.shape[-1] not in (2, 3) or gt_trajectory.shape[0] == 0:
            return None

        try:
            gt_waypoints = gt_trajectory[None, ...]
            if gt_waypoints.shape[1] == self.REQUIRED_NUM_WAYPOINTS:
                # Coarse 0.5s GT (if provided) should follow the same interpolation path.
                gt_proposals = self._build_proposals(gt_waypoints, scene)
            else:
                # Default cache_loader GT is dense 0.1s ego-relative poses.
                gt_proposals = self._waypoints_to_proposals(gt_waypoints, scene.ego_state)

            gt_simulated = self._simulator.simulate_proposals(
                ego_state=scene.ego_state,
                proposals=gt_proposals,
                observation=scene.observation,
            )
            gt_coords = state_to_coords(gt_simulated, self._vehicle)
            gt_polygons = coords_to_polygons(gt_coords)
            gt_areas = self._calculate_ego_areas(gt_coords, scene)

            progress = float(self._progress(gt_coords, scene)[0])

            multi = np.ones(len(MultiMetricIndex), dtype=np.float64)
            multi[MultiMetricIndex.NO_COLLISION] = self._no_at_fault_collision(
                gt_simulated, gt_polygons, gt_areas, scene,
            )[0]
            multi[MultiMetricIndex.DRIVABLE_AREA] = self._drivable_area_compliance(gt_areas)[0]
            multi[MultiMetricIndex.DRIVING_DIRECTION] = self._driving_direction_compliance(
                gt_coords, gt_areas, scene,
            )[0]
            multi[MultiMetricIndex.TRAFFIC_LIGHT] = self._traffic_light_compliance(
                gt_polygons, scene,
            )[0]

            weighted = np.ones(len(WeightedMetricIndex), dtype=np.float64)
            weighted[WeightedMetricIndex.TTC] = self._time_to_collision(
                gt_simulated, gt_coords, gt_areas, scene,
            )[0]
            weighted[WeightedMetricIndex.LANE_KEEPING] = self._lane_keeping(
                gt_coords, gt_areas, scene,
            )[0]
            weighted[WeightedMetricIndex.COMFORT] = self._history_comfort(
                gt_simulated, scene,
            )[0]

            return _GTSimResult(progress=progress, multi_metrics=multi, weighted_metrics=weighted)
        except Exception:
            return None

    _MULTI_METRIC_NAMES = {
        MultiMetricIndex.NO_COLLISION: "no_at_fault_collisions",
        MultiMetricIndex.DRIVABLE_AREA: "drivable_area_compliance",
        MultiMetricIndex.DRIVING_DIRECTION: "driving_direction_compliance",
        MultiMetricIndex.TRAFFIC_LIGHT: "traffic_light_compliance",
    }
    _WEIGHTED_METRIC_NAMES = {
        WeightedMetricIndex.PROGRESS: "ego_progress",
        WeightedMetricIndex.TTC: "time_to_collision",
        WeightedMetricIndex.LANE_KEEPING: "lane_keeping",
        WeightedMetricIndex.COMFORT: "history_comfort",
    }

    def _apply_human_penalty_from_gt(
        self,
        multi_metrics: np.ndarray,
        weighted_metrics: np.ndarray,
        gt_result: _GTSimResult,
    ) -> list[str]:
        """Use pre-computed GT metrics to forgive agent. No simulation needed."""
        overridden: list[str] = []
        for idx, name in self._MULTI_METRIC_NAMES.items():
            if gt_result.multi_metrics[idx] == 0.0:
                multi_metrics[:, idx] = 1.0
                overridden.append(name)
        for idx, name in self._WEIGHTED_METRIC_NAMES.items():
            if idx == WeightedMetricIndex.PROGRESS:
                continue  # progress is already GT-normalized
            if gt_result.weighted_metrics[idx] == 0.0:
                weighted_metrics[:, idx] = 1.0
                overridden.append(name)
        return overridden

    def _normalize_progress(
        self,
        progress_raw: np.ndarray,
        multi_metrics: np.ndarray,
        gt_progress: float | None = None,
    ) -> np.ndarray:
        # NavSim v1: masked_progress = progress * prod(multiplicative_metrics)
        if self._config.scoring_version == "v1":
            multi_prod = multi_metrics[
                :, [MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA]
            ].prod(axis=1)
        else:
            multi_prod = multi_metrics.prod(axis=1)
        masked_progress = progress_raw * multi_prod

        if gt_progress is not None and gt_progress > self._config.progress_distance_threshold:
            # Official NavSim: normalize against GT trajectory progress
            normalized = np.clip(progress_raw / gt_progress, 0.0, 1.0)
            # NavSim: zero out progress when multiplicative score is zero
            normalized[multi_prod == 0.0] = 0.0
            return normalized

        max_progress = float(masked_progress.max()) if len(masked_progress) > 0 else 0.0

        if max_progress > self._config.progress_distance_threshold:
            return np.clip(progress_raw / max_progress, 0.0, 1.0)
        else:
            # NavSim: all proposals get 1.0, except those with zero multiplicative score
            result = np.ones_like(progress_raw, dtype=np.float64)
            result[multi_prod == 0.0] = 0.0
            return result

    def _time_to_collision(
        self,
        simulated_states: np.ndarray,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
    ) -> np.ndarray:
        dt = self._dt(scene)
        total_forward_steps = max(int(round(self._config.future_collision_horizon / dt)), 1)
        sample_stride = max(int(round(0.33 / dt)), 1)
        future_offsets = np.arange(0, total_forward_steps, sample_stride, dtype=int)
        if len(future_offsets) == 0:
            future_offsets = np.asarray([0], dtype=int)

        speeds = np.hypot(
            simulated_states[..., StateIndex.VELOCITY_X],
            simulated_states[..., StateIndex.VELOCITY_Y],
        )
        dxy_per_second = np.stack(
            [
                np.cos(simulated_states[..., StateIndex.HEADING]) * speeds,
                np.sin(simulated_states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        projected_coords = np.repeat(ego_coords[:, :, None, :, :], len(future_offsets), axis=2)
        for offset_idx, future_offset in enumerate(future_offsets):
            projected_coords[:, :, offset_idx] += (
                dxy_per_second[:, :, None, :] * (float(future_offset) * dt)
            )
        projected_polygons = coords_to_polygons(projected_coords)

        scores = np.ones(len(simulated_states), dtype=np.float64)
        collided_track_ids = [set(scene.collided_track_ids) for _ in range(len(simulated_states))]
        num_time_steps = simulated_states.shape[1]

        for time_idx in range(num_time_steps):
            for offset_idx, future_offset in enumerate(future_offsets):
                current_time_idx = time_idx + future_offset
                occupancy_map = self._get_occupancy_map(scene, current_time_idx)
                if occupancy_map is None:
                    continue

                for proposal_idx in range(len(simulated_states)):
                    if scores[proposal_idx] == 0.0:
                        continue
                    if speeds[proposal_idx, time_idx] < self._config.stopped_speed_threshold:
                        continue

                    ego_polygon = projected_polygons[proposal_idx, time_idx, offset_idx]
                    if not occupancy_map.intersects(ego_polygon)[0]:
                        continue

                    for token in occupancy_map.get_colliding_tokens(ego_polygon):
                        if RED_LIGHT_TOKEN_PREFIX in token or token in collided_track_ids[proposal_idx]:
                            continue

                        ego_in_intersection = scene.drivable_area_map.is_in_layer(
                            simulated_states[proposal_idx, time_idx, [StateIndex.X, StateIndex.Y]],
                            SemanticMapLayer.INTERSECTION,
                        )
                        if self._is_ttc_violation(
                            simulated_states[proposal_idx, time_idx],
                            occupancy_map[token],
                            ego_areas[proposal_idx, time_idx],
                            ego_in_intersection,
                        ):
                            # NavSim: binary — any violation sets TTC to 0.0
                            scores[proposal_idx] = 0.0
                            collided_track_ids[proposal_idx].add(token)
                        else:
                            collided_track_ids[proposal_idx].add(token)
        return scores

    def _lane_keeping(self, ego_coords: np.ndarray, ego_areas: np.ndarray, scene: SceneContext) -> np.ndarray:
        """NavSim binary scoring: once consecutive exceeds >= threshold, score = 0.0."""
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        horizon_steps = max(int(round(self._config.lane_keeping_horizon / self._dt(scene))), 1)
        scores = np.ones(len(ego_coords), dtype=np.float64)

        for proposal_idx in range(len(ego_coords)):
            consecutive_exceeds = 0
            violated = False
            for time_idx in range(ego_coords.shape[1]):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    # Official: continue without resetting counter
                    continue
                deviation = Point(*centers[proposal_idx, time_idx]).distance(scene.centerline.linestring)
                if deviation > self._config.lane_keeping_deviation:
                    consecutive_exceeds += 1
                    if consecutive_exceeds >= horizon_steps:
                        violated = True
                        break
                else:
                    consecutive_exceeds = 0

            if violated:
                scores[proposal_idx] = 0.0

        return scores

    def _history_comfort(self, simulated_states: np.ndarray, scene: SceneContext) -> np.ndarray:
        scores = np.ones(len(simulated_states), dtype=np.float64)
        past_states = np.asarray(scene.ego_past_states, dtype=np.float64)

        # NavSim: comfort only evaluated when past_human_trajectory exists
        if len(past_states) == 0:
            return scores

        dt = self._dt(scene)
        for proposal_idx in range(len(simulated_states)):
            padded_states = np.concatenate([past_states, simulated_states[proposal_idx]], axis=0)
            time_points_s = np.arange(len(padded_states), dtype=np.float64) * dt
            scores[proposal_idx] = 1.0 if ego_is_comfortable(padded_states, time_points_s) else 0.0
        return scores

    def _aggregate(
        self,
        multi_metrics: np.ndarray,
        weighted_metrics: np.ndarray,
    ) -> np.ndarray:
        if self._config.scoring_version == "v1":
            # v1 PDMS: only NC(0) and DAC(1) are multiplicative
            multiplicative_scores = multi_metrics[
                :, [MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA]
            ].prod(axis=1)
        else:
            # v2 EPDMS: all 4 safety metrics are multiplicative
            multiplicative_scores = multi_metrics.prod(axis=1)
        weights = self._config.weighted_metrics_array
        weighted_scores = (weighted_metrics * weights[None, :]).sum(axis=1) / weights.sum()
        return multiplicative_scores * weighted_scores

    def _classify_collision_type(
        self,
        ego_state: np.ndarray,
        ego_polygon: BaseGeometry,
        track_polygon: BaseGeometry,
        token: str,
        scene: SceneContext,
        time_idx: int,
    ) -> CollisionType:
        ego_speed = np.hypot(
            ego_state[StateIndex.VELOCITY_X],
            ego_state[StateIndex.VELOCITY_Y],
        )
        if ego_speed <= _COLLISION_STOPPED_THRESHOLD:
            return CollisionType.STOPPED_EGO_OPEN

        track_speed = self._estimate_track_speed(scene, token, time_idx)
        if track_speed <= _COLLISION_STOPPED_THRESHOLD:
            return CollisionType.STOPPED_TRACK_OPEN

        if self._is_track_behind_ego(ego_state, track_polygon):
            return CollisionType.ACTIVE_REAR_BUMPER

        front_bumper = LineString(
            [
                ego_polygon.exterior.coords[0],
                ego_polygon.exterior.coords[3],
            ]
        )
        if front_bumper.intersects(track_polygon):
            return CollisionType.ACTIVE_FRONT_BUMPER

        return CollisionType.ACTIVE_LATERAL

    def _collision_penalty(
        self, collision_type: CollisionType, ego_area: np.ndarray,
        token: str, scene: SceneContext,
    ) -> float:
        # Not at-fault: rear collision or stopped ego — forgive (official behavior)
        if collision_type in (CollisionType.ACTIVE_REAR_BUMPER, CollisionType.STOPPED_EGO_OPEN):
            return 1.0
        # At-fault: penalty depends on object type (agent=0.0, static=0.5)
        is_agent = scene.track_object_types.get(token, "agent") != "static"
        at_fault_score = 0.0 if is_agent else 0.5
        if collision_type == CollisionType.ACTIVE_FRONT_BUMPER:
            return at_fault_score
        if collision_type == CollisionType.STOPPED_TRACK_OPEN:
            return at_fault_score
        # Lateral: at-fault only if in wrong area
        if ego_area[EgoAreaIndex.MULTIPLE_LANES] or ego_area[EgoAreaIndex.NON_DRIVABLE_AREA]:
            return at_fault_score
        return 1.0  # lateral in correct area = not at-fault

    def _is_ttc_violation(
        self,
        ego_state: np.ndarray,
        track_polygon: BaseGeometry,
        ego_area: np.ndarray,
        ego_in_intersection: bool,
    ) -> bool:
        agent_xy = np.asarray(track_polygon.centroid.coords[0], dtype=np.float64)
        relative_angle = self._get_agent_relative_angle(ego_state, agent_xy)
        is_ahead = relative_angle < np.deg2rad(30.0)
        is_behind = relative_angle > np.deg2rad(150.0)
        return bool(
            is_ahead
            or (
                (
                    ego_area[EgoAreaIndex.MULTIPLE_LANES]
                    or ego_area[EgoAreaIndex.NON_DRIVABLE_AREA]
                    or ego_in_intersection
                )
                and not is_behind
            )
        )

    @staticmethod
    def _get_agent_relative_angle(ego_state: np.ndarray, agent_xy: np.ndarray) -> float:
        """Angle between ego heading and ego→agent vector (nuPlan's get_agent_relative_angle)."""
        agent_vector = agent_xy - ego_state[[StateIndex.X, StateIndex.Y]]
        norm = np.linalg.norm(agent_vector)
        if norm < 1e-12:
            return 0.0
        ego_heading = float(ego_state[StateIndex.HEADING])
        ego_vector = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        dot_product = np.clip(np.dot(ego_vector, agent_vector / norm), -1.0, 1.0)
        return float(np.arccos(dot_product))

    def _estimate_track_speed(self, scene: SceneContext, token: str, time_idx: int) -> float:
        local_idx = self._local_time_idx(scene, time_idx)
        base_map = scene.observation.get_occupancy_map(local_idx)
        if base_map is None or token not in base_map.token_to_idx:
            return 0.0

        base_centroid = np.asarray(base_map[token].centroid.coords[0], dtype=np.float64)
        dt = self._dt(scene)
        num_steps = len(scene.observation.global_to_local_idcs)

        for offset in (1, -1):
            neighbor_time = time_idx + offset
            if not 0 <= neighbor_time < num_steps:
                continue
            neighbor_idx = self._local_time_idx(scene, neighbor_time)
            neighbor_map = scene.observation.get_occupancy_map(neighbor_idx)
            if neighbor_map is None or token not in neighbor_map.token_to_idx:
                continue
            neighbor_centroid = np.asarray(neighbor_map[token].centroid.coords[0], dtype=np.float64)
            return float(np.linalg.norm(neighbor_centroid - base_centroid) / dt)
        return 0.0

    @staticmethod
    def _points_in_map_tokens(
        points: np.ndarray,
        drivable_area_map: DrivableMap,
        include_layers: Iterable[SemanticMapLayer],
    ) -> np.ndarray:
        include_values = set(include_layers)
        selected_indices = [
            idx
            for idx, layer_name in enumerate(drivable_area_map.types)
            if _layer_name_to_enum(layer_name) in include_values
        ]
        if not selected_indices:
            return np.zeros((0, *points.shape[:-1]), dtype=bool)

        flat_points = points.reshape(-1, 2)
        membership = np.zeros((len(selected_indices), len(flat_points)), dtype=bool)
        x_coords = flat_points[:, 0]
        y_coords = flat_points[:, 1]
        for row_idx, polygon_idx in enumerate(selected_indices):
            membership[row_idx] = shapely.vectorized.contains(
                drivable_area_map[drivable_area_map.tokens[polygon_idx]],
                x_coords,
                y_coords,
            )
        return membership.reshape((len(selected_indices), *points.shape[:-1]))

    def _points_in_route_lanes(
        self,
        points: np.ndarray,
        drivable_area_map: DrivableMap,
        route_lane_ids: Sequence[str],
    ) -> np.ndarray:
        route_lane_id_set = set(route_lane_ids)
        if not route_lane_id_set:
            return np.zeros(points.shape[:-1], dtype=bool)

        selected_indices = [
            idx
            for idx, (token, layer_name) in enumerate(zip(drivable_area_map.tokens, drivable_area_map.types))
            if token in route_lane_id_set and _layer_name_to_enum(layer_name) in {
                SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR,
            }
        ]
        if not selected_indices:
            return np.zeros(points.shape[:-1], dtype=bool)

        flat_points = points.reshape(-1, 2)
        x_coords = flat_points[:, 0]
        y_coords = flat_points[:, 1]
        membership = np.zeros((len(selected_indices), len(flat_points)), dtype=bool)
        for row_idx, polygon_idx in enumerate(selected_indices):
            membership[row_idx] = shapely.vectorized.contains(
                drivable_area_map[drivable_area_map.tokens[polygon_idx]],
                x_coords,
                y_coords,
            )
        return membership.any(axis=0).reshape(points.shape[:-1])

    @staticmethod
    def _relative_point_in_ego_frame(ego_state: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
        dx = point_xy[0] - ego_state[StateIndex.X]
        dy = point_xy[1] - ego_state[StateIndex.Y]
        theta = float(ego_state[StateIndex.HEADING])
        return np.asarray(
            [
                dx * np.cos(theta) + dy * np.sin(theta),
                -dx * np.sin(theta) + dy * np.cos(theta),
            ],
            dtype=np.float64,
        )

    def _is_track_behind_ego(self, ego_state: np.ndarray, track_polygon: BaseGeometry) -> bool:
        relative = self._relative_point_in_ego_frame(
            ego_state,
            np.asarray(track_polygon.centroid.coords[0], dtype=np.float64),
        )
        return bool(relative[0] < 0.0)

    @staticmethod
    def _dt(scene: SceneContext) -> float:
        return float(scene.observation.interval_time)

    @staticmethod
    def _local_time_idx(scene: SceneContext, time_idx: int) -> int:
        global_to_local = scene.observation.global_to_local_idcs
        # Clamp to observation range — simulated states may exceed observation horizon
        clamped = min(max(time_idx, 0), len(global_to_local) - 1)
        return int(global_to_local[clamped])

    def _get_occupancy_map(self, scene: SceneContext, time_idx: int) -> OccupancyMap | None:
        return scene.observation.get_occupancy_map(self._local_time_idx(scene, time_idx))

    def _get_red_light_map(self, scene: SceneContext, time_idx: int) -> OccupancyMap | None:
        return scene.observation.get_red_light_map(self._local_time_idx(scene, time_idx))

    # ── RL Scoring ──────────────────────────────────────────────────────────

    def score_for_rl(
        self,
        waypoints_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
    ) -> RLScoringResult:
        return self.score_batch_for_rl(waypoints_xy[None, ...], scene, rl_config)[0]

    def score_batch_for_rl(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
    ) -> List[RLScoringResult]:
        """Score trajectories for RL reward: 8 independent sub-rewards, additive aggregation."""
        rl_config = rl_config or RLScorerConfig.v1()

        # Reuse PDMS simulation pipeline so RL stays aligned with PDMS interpolation/simulation.
        batch_waypoints = self._coerce_trajectories(trajectories_xy)
        proposals = self._build_proposals(batch_waypoints, scene)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state,
            proposals=proposals,
            observation=scene.observation,
        )
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = self._calculate_ego_areas(ego_coords, scene)

        # Safety layer: continuous or discrete
        if rl_config.safety_mode == "continuous":
            nc = self._nc_continuous(simulated_states, ego_polygons, ego_areas, scene, rl_config)
            dac = self._dac_continuous(ego_polygons, scene, rl_config)
            ddc = (
                self._ddc_continuous(ego_coords, ego_areas, scene, rl_config)
                if rl_config.ddc_weight > 0.0
                else np.ones(len(batch_waypoints), dtype=np.float64)
            )
            tlc = (
                self._tlc_continuous(ego_polygons, scene, rl_config)
                if rl_config.tlc_weight > 0.0
                else np.ones(len(batch_waypoints), dtype=np.float64)
            )
        else:
            nc = self._no_at_fault_collision(simulated_states, ego_polygons, ego_areas, scene)
            dac = self._drivable_area_compliance(ego_areas)
            ddc = (
                self._driving_direction_compliance(ego_coords, ego_areas, scene)
                if rl_config.ddc_weight > 0.0
                else np.ones(len(batch_waypoints), dtype=np.float64)
            )
            tlc = (
                self._traffic_light_compliance(ego_polygons, scene)
                if rl_config.tlc_weight > 0.0
                else np.ones(len(batch_waypoints), dtype=np.float64)
            )

        # Performance layer: always continuous
        ep = self._ep_continuous(ego_coords, scene, rl_config)
        ttc = self._ttc_continuous(simulated_states, ego_coords, ego_areas, scene, rl_config)
        lk = (
            self._lk_continuous(ego_coords, ego_areas, scene, rl_config)
            if rl_config.lk_weight > 0.0
            else np.ones(len(batch_waypoints), dtype=np.float64)
        )
        hc = self._hc_continuous(simulated_states, scene)

        # Aggregation: soft safety gate × weighted performance average
        # safety_gate = (NC × DAC × DDC × TLC) ^ alpha
        safety_product = nc * dac * ddc * tlc
        alpha = rl_config.safety_gate_alpha
        safety_gate = np.power(np.clip(safety_product, 0.0, 1.0), alpha)

        # performance = weighted_avg(EP, TTC, LK, HC)
        perf_metrics = np.stack([ep, ttc, lk, hc], axis=1)
        perf_weights = rl_config.performance_weights
        perf_sum = perf_weights.sum()
        if perf_sum > 0:
            performance = (perf_metrics * perf_weights[None, :]).sum(axis=1) / perf_sum
        else:
            performance = np.ones(len(batch_waypoints), dtype=np.float64)

        rl_scores = safety_gate * performance

        results: List[RLScoringResult] = []
        for i in range(len(batch_waypoints)):
            results.append(
                RLScoringResult(
                    rl_score=float(rl_scores[i]),
                    no_at_fault_collisions=float(nc[i]),
                    drivable_area_compliance=float(dac[i]),
                    driving_direction_compliance=float(ddc[i]),
                    traffic_light_compliance=float(tlc[i]),
                    ego_progress=float(ep[i]),
                    time_to_collision=float(ttc[i]),
                    lane_keeping=float(lk[i]),
                    history_comfort=float(hc[i]),
                )
            )
        return results

    # ── Safety layer: continuous ─────────────────────────────────────────

    def _nc_continuous(
        self,
        simulated_states: np.ndarray,
        ego_polygons: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Continuous NC: full at-fault classification with per-timestep decay.

        Each collision frame independently decays the score by (1 - severity_t),
        encoding both overlap depth and collision duration without artificial params.

        penalty = at_fault_floor + (1 - at_fault_floor) × ∏(1 - severity_t)
        where at_fault_floor = 0.0 (agent) or 0.5 (static).
        """
        del rl_config  # reserved for future tuning knobs

        batch_size = len(ego_polygons)
        num_steps = ego_polygons.shape[1]

        # Per-proposal tracking: token → (at_fault_floor, cumulative_product)
        # cumulative_product = ∏(1 - severity_t): each collision frame decays the score.
        collision_records: list[dict[str, tuple[float, float]]] = [
            {} for _ in range(batch_size)
        ]
        # Tokens already forgiven (not-at-fault or pre-existing collisions)
        forgiven: list[set[str]] = [set(scene.collided_track_ids) for _ in range(batch_size)]

        for time_idx in range(num_steps):
            occupancy_map = self._get_occupancy_map(scene, time_idx)
            if occupancy_map is None:
                continue

            for pi in range(batch_size):
                ego_poly = ego_polygons[pi, time_idx]
                if not occupancy_map.intersects(ego_poly)[0]:
                    continue

                ego_area = float(ego_poly.area)
                if ego_area < 1e-9:
                    continue

                for token in occupancy_map.get_colliding_tokens(ego_poly):
                    if RED_LIGHT_TOKEN_PREFIX in token or token in forgiven[pi]:
                        continue

                    track_polygon = occupancy_map[token]

                    # First encounter with this token: classify
                    if token not in collision_records[pi]:
                        collision_type = self._classify_collision_type(
                            simulated_states[pi, time_idx],
                            ego_poly,
                            track_polygon,
                            token,
                            scene,
                            time_idx,
                        )

                        if collision_type in (
                            CollisionType.ACTIVE_REAR_BUMPER,
                            CollisionType.STOPPED_EGO_OPEN,
                        ):
                            forgiven[pi].add(token)
                            continue

                        at_fault_floor = self._collision_penalty(
                            collision_type,
                            ego_areas[pi, time_idx],
                            token,
                            scene,
                        )
                        if at_fault_floor >= 1.0:
                            forgiven[pi].add(token)
                            continue

                        collision_records[pi][token] = (at_fault_floor, 1.0)

                    # Accumulate per-timestep decay: ∏(1 - severity_t)
                    overlap = ego_poly.intersection(track_polygon).area
                    severity = min(overlap / ego_area, 1.0)
                    floor, cum_prod = collision_records[pi][token]
                    collision_records[pi][token] = (floor, cum_prod * (1.0 - severity))

        # Compute final scores: penalty = floor + (1 - floor) × cumulative_product
        scores = np.ones(batch_size, dtype=np.float64)
        for pi in range(batch_size):
            for token, (at_fault_floor, cum_prod) in collision_records[pi].items():
                penalty = at_fault_floor + (1.0 - at_fault_floor) * cum_prod
                scores[pi] = min(scores[pi], penalty)

        return scores

    # Layers always included in DAC sweep regardless of route membership.
    # Roadblocks wrap lane polygons; intersections are shared turning areas.
    _DAC_STRUCTURAL_LAYERS = frozenset({"roadblock", "intersection"})

    def _dac_continuous(
        self,
        ego_polygons: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Per-timestep coverage product DAC: ∏ coverage(t) over all timesteps.

        Each timestep: coverage = intersection(ego_polygon, local_drivable) / ego_area.
        Product over time: brief edge-brush → mild penalty; sustained offroad → severe.

        Drivable region = route lanes + structural layers (roadblock, intersection).
        Uses STRtree for per-timestep spatial query.
        """
        del rl_config

        dm = scene.drivable_area_map
        eligible = frozenset(
            i for i, (tok, typ) in enumerate(zip(dm.tokens, dm.types))
            if tok in scene.route_lane_ids or typ in self._DAC_STRUCTURAL_LAYERS
        )

        batch_size, num_steps = ego_polygons.shape[:2]
        scores = np.ones(batch_size, dtype=np.float64)

        if dm._tree is None:
            return np.zeros(batch_size, dtype=np.float64)

        for proposal_idx in range(batch_size):
            for time_idx in range(num_steps):
                ego_poly = ego_polygons[proposal_idx, time_idx]
                ego_area = float(ego_poly.area)
                if ego_area < 1e-9:
                    continue

                nearby = dm._tree.query(ego_poly, predicate="intersects")
                local = [j for j in nearby.tolist() if j in eligible]

                if not local:
                    # Completely outside drivable — coverage = 0
                    scores[proposal_idx] = 0.0
                    break

                local_union = unary_union([dm._polygons[j] for j in local])
                inside = float(ego_poly.intersection(local_union).area)
                coverage = min(inside / ego_area, 1.0)

                scores[proposal_idx] *= coverage
                if scores[proposal_idx] < 1e-6:
                    # Already near zero, no need to continue
                    scores[proposal_idx] = 0.0
                    break

        return scores

    def _ddc_continuous(
        self,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Continuous driving direction compliance — linear interpolation between thresholds."""
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        oncoming_progress = np.zeros(centers.shape[:2], dtype=np.float64)
        oncoming_progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)

        # Zero out progress only where not oncoming.
        not_oncoming = ~ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_progress[not_oncoming] = 0.0

        horizon_steps = max(int(round(rl_config.driving_direction_compliance_threshold / self._dt(scene))), 1)
        rolling_progress = np.zeros_like(oncoming_progress)
        for time_idx in range(oncoming_progress.shape[1]):
            start_idx = max(0, time_idx - horizon_steps)
            rolling_progress[:, time_idx] = oncoming_progress[:, start_idx : time_idx + 1].sum(axis=1)

        max_progress = rolling_progress.max(axis=1)

        lo = rl_config.driving_direction_compliance_threshold
        hi = rl_config.driving_direction_violation_threshold
        # Linear interpolation: <lo → 1.0, lo..hi → 1→0, >hi → 0.0
        scores = np.clip(1.0 - (max_progress - lo) / (hi - lo), 0.0, 1.0)
        return scores

    def _tlc_continuous(
        self,
        ego_polygons: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Continuous traffic light compliance based on distance to red-light zones.

        Only checks distance when ego is near red-light zones (within margin buffer).
        Uses buffered intersection to avoid scanning all tokens.
        """
        batch_size = len(ego_polygons)
        min_distances = np.full(batch_size, np.inf, dtype=np.float64)
        margin = rl_config.tlc_margin

        for time_idx in range(ego_polygons.shape[1]):
            red_light_map = self._get_red_light_map(scene, time_idx)
            if red_light_map is None:
                continue

            for proposal_idx in range(batch_size):
                if min_distances[proposal_idx] == 0.0:
                    continue  # already violated

                ego_poly = ego_polygons[proposal_idx, time_idx]

                # Direct intersection → violation
                if red_light_map.intersects(ego_poly)[0]:
                    for token in red_light_map.get_colliding_tokens(ego_poly):
                        if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                            min_distances[proposal_idx] = 0.0
                            break
                else:
                    # Only check distance with buffered polygon (avoids scanning all tokens)
                    buffered = ego_poly.buffer(margin)
                    if red_light_map.intersects(buffered)[0]:
                        for token in red_light_map.get_colliding_tokens(buffered):
                            if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                                dist = ego_poly.distance(red_light_map[token])
                                min_distances[proposal_idx] = min(min_distances[proposal_idx], dist)

        # No red lights nearby → score 1.0
        min_distances = np.where(np.isinf(min_distances), margin * 2.0, min_distances)
        scores = np.clip(min_distances / margin, 0.0, 1.0)
        return scores

    # ── Performance layer: continuous ────────────────────────────────────

    def _ep_continuous(
        self,
        ego_coords: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Continuous ego progress with GT-progress normalization when available."""
        raw_progress = self._progress(ego_coords, scene)
        gt_progress = scene.gt_progress

        if gt_progress is not None and gt_progress > rl_config.progress_distance_threshold:
            return np.clip(raw_progress / gt_progress, 0.0, 1.0)

        return np.clip(raw_progress / rl_config.progress_distance_threshold, 0.0, 1.0)

    def _ttc_continuous(
        self,
        simulated_states: np.ndarray,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Continuous TTC: normalized time to first violation [0, 1]."""
        dt = self._dt(scene)
        total_forward_steps = max(int(round(rl_config.future_collision_horizon / dt)), 1)
        sample_stride = max(int(round(0.33 / dt)), 1)
        future_offsets = np.arange(0, total_forward_steps, sample_stride, dtype=int)
        if len(future_offsets) == 0:
            future_offsets = np.asarray([0], dtype=int)

        speeds = np.hypot(
            simulated_states[..., StateIndex.VELOCITY_X],
            simulated_states[..., StateIndex.VELOCITY_Y],
        )
        dxy_per_second = np.stack(
            [
                np.cos(simulated_states[..., StateIndex.HEADING]) * speeds,
                np.sin(simulated_states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )

        projected_coords = np.repeat(ego_coords[:, :, None, :, :], len(future_offsets), axis=2)
        for offset_idx, future_offset in enumerate(future_offsets):
            projected_coords[:, :, offset_idx] += (
                dxy_per_second[:, :, None, :] * (float(future_offset) * dt)
            )
        projected_polygons = coords_to_polygons(projected_coords)

        # Track first violation time for each proposal
        batch_size = len(simulated_states)
        first_violation_time = np.full(batch_size, np.inf, dtype=np.float64)
        collided_track_ids = [set(scene.collided_track_ids) for _ in range(batch_size)]
        num_time_steps = simulated_states.shape[1]

        for time_idx in range(num_time_steps):
            for offset_idx, future_offset in enumerate(future_offsets):
                current_time_idx = time_idx + future_offset
                occupancy_map = self._get_occupancy_map(scene, current_time_idx)
                if occupancy_map is None:
                    continue

                for proposal_idx in range(batch_size):
                    if first_violation_time[proposal_idx] < np.inf:
                        continue
                    if speeds[proposal_idx, time_idx] < rl_config.stopped_speed_threshold:
                        continue

                    ego_polygon = projected_polygons[proposal_idx, time_idx, offset_idx]
                    if not occupancy_map.intersects(ego_polygon)[0]:
                        continue

                    for token in occupancy_map.get_colliding_tokens(ego_polygon):
                        if RED_LIGHT_TOKEN_PREFIX in token or token in collided_track_ids[proposal_idx]:
                            continue

                        ego_in_intersection = scene.drivable_area_map.is_in_layer(
                            simulated_states[proposal_idx, time_idx, [StateIndex.X, StateIndex.Y]],
                            SemanticMapLayer.INTERSECTION,
                        )
                        if self._is_ttc_violation(
                            simulated_states[proposal_idx, time_idx],
                            occupancy_map[token],
                            ego_areas[proposal_idx, time_idx],
                            ego_in_intersection,
                        ):
                            violation_time = float(time_idx) * dt
                            first_violation_time[proposal_idx] = violation_time
                        else:
                            collided_track_ids[proposal_idx].add(token)

        # Normalize: ttc / horizon → [0, 1]. No violation = 1.0
        ttc_seconds = np.where(np.isinf(first_violation_time), rl_config.ttc_horizon, first_violation_time)
        scores = np.clip(ttc_seconds / rl_config.ttc_horizon, 0.0, 1.0)
        return scores

    def _lk_continuous(
        self,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> np.ndarray:
        """Continuous lane keeping: 1.0 = on center, decays with mean deviation."""
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        batch_size = len(ego_coords)
        scores = np.ones(batch_size, dtype=np.float64)

        for proposal_idx in range(batch_size):
            deviations = []
            for time_idx in range(ego_coords.shape[1]):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    continue
                deviation = Point(*centers[proposal_idx, time_idx]).distance(scene.centerline.linestring)
                deviations.append(deviation)

            if not deviations:
                continue

            mean_dev = float(np.mean(deviations))
            lo = rl_config.lane_keeping_deviation
            hi = rl_config.lane_keeping_max_deviation
            if mean_dev <= lo:
                scores[proposal_idx] = 1.0
            elif mean_dev >= hi:
                scores[proposal_idx] = 0.0
            else:
                scores[proposal_idx] = 1.0 - (mean_dev - lo) / (hi - lo)

        return scores

    def _hc_continuous(
        self,
        simulated_states: np.ndarray,
        scene: SceneContext,
    ) -> np.ndarray:
        """Continuous comfort metric using max violation ratio."""
        scores = np.ones(len(simulated_states), dtype=np.float64)
        past_states = np.asarray(scene.ego_past_states, dtype=np.float64)

        if len(past_states) == 0:
            return scores

        dt = self._dt(scene)
        for proposal_idx in range(len(simulated_states)):
            padded_states = np.concatenate([past_states, simulated_states[proposal_idx]], axis=0)
            time_points_s = np.arange(len(padded_states), dtype=np.float64) * dt
            scores[proposal_idx] = ego_comfort_violation(padded_states, time_points_s)
        return scores


def _layer_name_to_enum(layer_name: str) -> SemanticMapLayer | None:
    return _LAYER_NAME_TO_ENUM.get(_normalize_layer_name(layer_name))

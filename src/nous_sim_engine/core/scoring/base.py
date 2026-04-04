from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import numpy.typing as npt
import shapely.vectorized
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..comfort import ego_comfort_violation, ego_is_comfortable
from ..enums import (
    BBCoordsIndex,
    CollisionType,
    DRIVABLE_LAYERS,
    EgoAreaIndex,
    MultiMetricIndex,
    SemanticMapLayer,
    StateIndex,
    WeightedMetricIndex,
)
from ..geometry import coords_to_polygons, normalize_angle, state_to_coords
from ..occupancy import DrivableMap, OccupancyMap, _LAYER_NAME_TO_ENUM, _normalize_layer_name
from ..simulator import PDMSimulator
from ..types import SceneContext, VehicleParams


RED_LIGHT_TOKEN_PREFIX = "red_light"

# Official NavSim uses 5e-2 for collision stopped classification
_COLLISION_STOPPED_THRESHOLD = 5e-2


@dataclass(frozen=True)
class PDMScorerConfig:
    scoring_version: str = "v1"
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
    """

    ep_weight: float = 5.0
    ttc_weight: float = 5.0
    hc_weight: float = 2.0
    lk_weight: float = 0.0

    safety_gate_alpha: float = 0.5
    safety_mode: str = "continuous"

    nc_weight: float = 5.0
    dac_weight: float = 3.0
    ddc_weight: float = 0.0
    tlc_weight: float = 0.0

    collision_distance_scale: float = 2.0
    dac_margin: float = 2.0
    tlc_margin: float = 1.0

    progress_distance_threshold: float = 5.0
    ttc_horizon: float = 4.0  # match 4s simulation horizon
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
        return cls(ddc_weight=0.0, tlc_weight=0.0, lk_weight=0.0)

    @property
    def weights_array(self) -> npt.NDArray[np.float64]:
        return np.array(
            [self.nc_weight, self.dac_weight, self.ddc_weight, self.tlc_weight,
             self.ep_weight, self.ttc_weight, self.lk_weight, self.hc_weight],
            dtype=np.float64,
        )

    @property
    def performance_weights(self) -> npt.NDArray[np.float64]:
        return np.array(
            [self.ep_weight, self.ttc_weight, self.lk_weight, self.hc_weight],
            dtype=np.float64,
        )


@dataclass
class _GTSimResult:
    """Cached GT simulation result, shared by progress normalization and human_penalty_filter."""
    progress: float
    multi_metrics: np.ndarray    # (len(MultiMetricIndex),)
    weighted_metrics: np.ndarray  # (len(WeightedMetricIndex),)


class ScorerBase:
    """Shared infrastructure for all scorer versions.

    Provides coordinate transforms, collision classification, spatial queries,
    and simulation pipeline. Subclasses implement version-specific scoring logic.
    """

    REQUIRED_NUM_WAYPOINTS = 8

    def __init__(
        self,
        vehicle: VehicleParams | None = None,
        simulator: PDMSimulator | None = None,
        discretization_time: float = 0.1,
    ) -> None:
        self._vehicle = vehicle or VehicleParams()
        self._simulator = simulator or PDMSimulator(
            discretization_time=discretization_time,
            vehicle=self._vehicle,
        )

    # ── Trajectory preprocessing ──────────────────────────────────────

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
        if T != ScorerBase.REQUIRED_NUM_WAYPOINTS:
            raise ValueError(
                f"Input must be exactly {ScorerBase.REQUIRED_NUM_WAYPOINTS} waypoints "
                f"at 0.5s intervals (4s horizon), got {T} waypoints. "
                f"Expected shape: [B, {ScorerBase.REQUIRED_NUM_WAYPOINTS}, 2|3]"
            )
        return trajectory_array

    @staticmethod
    def _derive_relative_headings(waypoints_xy: np.ndarray) -> np.ndarray:
        dxy = np.diff(waypoints_xy, axis=1)
        raw_fwd = np.arctan2(dxy[..., 1], dxy[..., 0])
        headings = np.concatenate([raw_fwd, raw_fwd[:, -1:]], axis=1)
        dist = np.linalg.norm(dxy, axis=-1)
        dist_full = np.concatenate([dist, dist[:, -1:]], axis=1)
        near_zero = dist_full < 1e-6
        if np.any(near_zero):
            for i in range(headings.shape[1]):
                if i > 0:
                    headings[:, i] = np.where(near_zero[:, i], headings[:, i - 1], headings[:, i])
                else:
                    headings[:, i] = np.where(near_zero[:, i], 0.0, headings[:, i])
        return headings

    @staticmethod
    def _ego_to_global(waypoints_xy: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
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
            else ScorerBase._derive_relative_headings(relative_xy)
        )
        headings = normalize_angle(relative_heading + ego_heading)
        return np.concatenate([global_xy, headings[..., None]], axis=-1)

    def _build_proposals(
        self, waypoints: np.ndarray, scene: SceneContext, input_interval: float = 0.5,
    ) -> np.ndarray:
        ego_state = scene.ego_state
        global_coarse = self._ego_to_global(waypoints, ego_state)

        sim_dt = float(scene.observation.interval_time)
        if sim_dt <= 0 or abs(input_interval - sim_dt) < 1e-6:
            return global_coarse

        ratio = round(input_interval / sim_dt)
        if ratio <= 1:
            return global_coarse

        batch_size, num_coarse, _ = global_coarse.shape
        ego_pose = np.array(
            [ego_state[StateIndex.X], ego_state[StateIndex.Y], ego_state[StateIndex.HEADING]],
            dtype=np.float64,
        )
        extended = np.concatenate(
            [np.broadcast_to(ego_pose, (batch_size, 1, 3)), global_coarse], axis=1,
        )
        extended_heading = np.unwrap(extended[..., 2], axis=1)

        num_extended = num_coarse + 1
        num_fine = (num_extended - 1) * ratio + 1

        alphas = np.arange(ratio, dtype=np.float64) / ratio
        start_xy = extended[:, :-1, :2]
        end_xy = extended[:, 1:, :2]
        start_h = extended_heading[:, :-1]
        end_h = extended_heading[:, 1:]

        a = alphas[None, None, :, None]
        interp_xy = start_xy[:, :, None, :] * (1 - a) + end_xy[:, :, None, :] * a
        a_h = alphas[None, None, :]
        interp_h = start_h[:, :, None] * (1 - a_h) + end_h[:, :, None] * a_h

        fine_xy = np.concatenate(
            [interp_xy.reshape(batch_size, -1, 2), extended[:, -1:, :2]], axis=1,
        )
        fine_heading = np.concatenate(
            [interp_h.reshape(batch_size, -1), extended_heading[:, -1:]], axis=1,
        )

        proposals = np.zeros((batch_size, num_fine, 3), dtype=np.float64)
        proposals[..., :2] = fine_xy
        proposals[..., 2] = normalize_angle(fine_heading)
        return proposals[:, 1:, :]

    def _waypoints_to_proposals(self, waypoints_xy: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
        return self._ego_to_global(waypoints_xy, ego_state)

    # ── Ego area computation ──────────────────────────────────────────

    def _calculate_ego_areas(self, ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
        batch_size, horizon, _, _ = ego_coords.shape
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        corners = ego_coords[:, :, :4, :]

        ego_areas = np.zeros((batch_size, horizon, len(EgoAreaIndex)), dtype=bool)

        lane_membership = self._points_in_map_tokens(
            corners, scene.drivable_area_map,
            {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR},
        )
        if len(lane_membership) > 0:
            lane_corner_counts = lane_membership.sum(axis=-1)
            multiple_lane_mask = (lane_corner_counts > 0).sum(axis=0) > 1
            single_lane_mask = np.any(lane_corner_counts == 4, axis=0)
            ego_areas[:, :, EgoAreaIndex.MULTIPLE_LANES] = multiple_lane_mask & ~single_lane_mask

        center_membership = scene.drivable_area_map.points_in_polygons(centers)
        corner_membership = scene.drivable_area_map.points_in_polygons(corners)
        drivable_layers_all = list(DRIVABLE_LAYERS)
        corner_in_drivable = corner_membership[..., drivable_layers_all[0]]
        for layer in drivable_layers_all[1:]:
            corner_in_drivable = corner_in_drivable | corner_membership[..., layer]
        ego_areas[:, :, EgoAreaIndex.NON_DRIVABLE_AREA] = ~corner_in_drivable.all(axis=-1)

        route_layers = self._points_in_route_lanes(centers, scene.drivable_area_map, scene.route_lane_ids)
        ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC] = ~route_layers
        ego_areas[:, :, EgoAreaIndex.IN_INTERSECTION] = center_membership[..., SemanticMapLayer.INTERSECTION]

        return ego_areas

    # ── Progress ──────────────────────────────────────────────────────

    @staticmethod
    def _progress(ego_coords: np.ndarray, scene: SceneContext) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        progress = np.zeros(len(ego_coords), dtype=np.float64)
        for proposal_idx in range(len(ego_coords)):
            start_progress = scene.centerline.project(Point(*centers[proposal_idx, 0]))
            end_progress = scene.centerline.project(Point(*centers[proposal_idx, -1]))
            progress[proposal_idx] = max(0.0, end_progress - start_progress)
        return progress

    # ── Collision classification ──────────────────────────────────────

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
            ego_state[StateIndex.VELOCITY_X], ego_state[StateIndex.VELOCITY_Y],
        )
        if ego_speed <= _COLLISION_STOPPED_THRESHOLD:
            return CollisionType.STOPPED_EGO_OPEN

        track_speed = self._estimate_track_speed(scene, token, time_idx)
        if track_speed <= _COLLISION_STOPPED_THRESHOLD:
            return CollisionType.STOPPED_TRACK_OPEN

        if self._is_track_behind_ego(ego_state, track_polygon):
            return CollisionType.ACTIVE_REAR_BUMPER

        front_bumper = LineString([
            ego_polygon.exterior.coords[0], ego_polygon.exterior.coords[3],
        ])
        if front_bumper.intersects(track_polygon):
            return CollisionType.ACTIVE_FRONT_BUMPER

        return CollisionType.ACTIVE_LATERAL

    def _collision_penalty(
        self,
        collision_type: CollisionType,
        ego_area: np.ndarray,
        token: str,
        scene: SceneContext,
    ) -> float:
        """At-fault collision penalty.

        Uses scene.track_object_types for object type lookup (v2 behavior).
        V1 override: _collision_penalty_from_observation.
        """
        if collision_type in (CollisionType.ACTIVE_REAR_BUMPER, CollisionType.STOPPED_EGO_OPEN):
            return 1.0

        is_agent = scene.track_object_types.get(token, "agent") != "static"
        at_fault_score = 0.0 if is_agent else 0.5

        if collision_type == CollisionType.ACTIVE_FRONT_BUMPER:
            return at_fault_score
        if collision_type == CollisionType.STOPPED_TRACK_OPEN:
            return at_fault_score
        if ego_area[EgoAreaIndex.MULTIPLE_LANES] or ego_area[EgoAreaIndex.NON_DRIVABLE_AREA]:
            return at_fault_score
        return 1.0

    @staticmethod
    def _collision_penalty_from_observation(
        collision_type: CollisionType,
        ego_area: np.ndarray,
        token: str,
        scene: SceneContext,
    ) -> float:
        """V1 collision penalty: reads object type from scene.track_object_types.

        Equivalent to recogdrive's observation.unique_objects[token].tracked_object_type.
        In our architecture, track_object_types is extracted from the same source.
        agent → 0.0, static → 0.5 (matches recogdrive).
        """
        if collision_type in (CollisionType.ACTIVE_REAR_BUMPER, CollisionType.STOPPED_EGO_OPEN):
            return 1.0

        # Matches recogdrive: tracked_object_type in AGENT_TYPES → 0.0, else 0.5
        is_agent = scene.track_object_types.get(token, "agent") != "static"
        at_fault_score = 0.0 if is_agent else 0.5

        if collision_type == CollisionType.ACTIVE_FRONT_BUMPER:
            return at_fault_score
        if collision_type == CollisionType.STOPPED_TRACK_OPEN:
            return at_fault_score
        if ego_area[EgoAreaIndex.MULTIPLE_LANES] or ego_area[EgoAreaIndex.NON_DRIVABLE_AREA]:
            return at_fault_score
        return 1.0

    # ── PDMS discrete metrics ─────────────────────────────────────────

    def _no_at_fault_collision(
        self,
        simulated_states: np.ndarray,
        ego_polygons: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        *,
        use_observation_types: bool = False,
    ) -> np.ndarray:
        scores = np.ones(len(simulated_states), dtype=np.float64)
        collided_track_ids = [set(scene.collided_track_ids) for _ in range(len(simulated_states))]

        penalty_fn = self._collision_penalty_from_observation if use_observation_types else self._collision_penalty

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
                        simulated_states[proposal_idx, time_idx], ego_polygon,
                        track_polygon, token, scene, time_idx,
                    )
                    at_fault_score = penalty_fn(
                        collision_type, ego_areas[proposal_idx, time_idx], token, scene,
                    )
                    scores[proposal_idx] = min(scores[proposal_idx], at_fault_score)
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
        *,
        horizon: float = 1.0,
        compliance_threshold: float = 2.0,
        violation_threshold: float = 6.0,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        oncoming_progress = np.zeros(centers.shape[:2], dtype=np.float64)
        oncoming_progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)

        not_oncoming = ~ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_progress[not_oncoming] = 0.0

        horizon_steps = max(int(round(horizon / self._dt(scene))), 1)
        rolling_progress = np.zeros_like(oncoming_progress)
        for time_idx in range(oncoming_progress.shape[1]):
            start_idx = max(0, time_idx - horizon_steps)
            rolling_progress[:, time_idx] = oncoming_progress[:, start_idx: time_idx + 1].sum(axis=1)

        max_progress = rolling_progress.max(axis=1)
        scores = np.ones(len(ego_coords), dtype=np.float64)
        medium_mask = (max_progress >= compliance_threshold) & (max_progress < violation_threshold)
        severe_mask = max_progress >= violation_threshold
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

    def _time_to_collision(
        self,
        simulated_states: np.ndarray,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        future_collision_horizon: float = 1.0,
        stopped_speed_threshold: float = 5e-3,
    ) -> np.ndarray:
        dt = self._dt(scene)
        total_forward_steps = max(int(round(future_collision_horizon / dt)), 1)
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
                    if speeds[proposal_idx, time_idx] < stopped_speed_threshold:
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
                            scores[proposal_idx] = 0.0
                            collided_track_ids[proposal_idx].add(token)
                        else:
                            collided_track_ids[proposal_idx].add(token)
        return scores

    def _lane_keeping(
        self,
        ego_coords: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        *,
        deviation: float = 0.5,
        horizon: float = 2.0,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        horizon_steps = max(int(round(horizon / self._dt(scene))), 1)
        scores = np.ones(len(ego_coords), dtype=np.float64)

        for proposal_idx in range(len(ego_coords)):
            consecutive_exceeds = 0
            violated = False
            for time_idx in range(ego_coords.shape[1]):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    continue
                dev = Point(*centers[proposal_idx, time_idx]).distance(scene.centerline.linestring)
                if dev > deviation:
                    consecutive_exceeds += 1
                    if consecutive_exceeds >= horizon_steps:
                        violated = True
                        break
                else:
                    consecutive_exceeds = 0
            if violated:
                scores[proposal_idx] = 0.0
        return scores

    def _history_comfort(
        self,
        simulated_states: np.ndarray,
        scene: SceneContext,
        *,
        use_past_states: bool = True,
    ) -> np.ndarray:
        """Discrete comfort: 1.0 if comfortable, 0.0 if not.

        Args:
            use_past_states: V2 prepends ego_past_states. V1 uses simulated states only.
        """
        scores = np.ones(len(simulated_states), dtype=np.float64)
        if use_past_states:
            past_states = np.asarray(scene.ego_past_states, dtype=np.float64)
            if len(past_states) == 0:
                return scores

        dt = self._dt(scene)
        for proposal_idx in range(len(simulated_states)):
            if use_past_states:
                padded = np.concatenate([past_states, simulated_states[proposal_idx]], axis=0)
            else:
                padded = simulated_states[proposal_idx]
            time_points_s = np.arange(len(padded), dtype=np.float64) * dt
            scores[proposal_idx] = 1.0 if ego_is_comfortable(padded, time_points_s) else 0.0
        return scores

    # ── GT simulation ─────────────────────────────────────────────────

    def _simulate_and_score_gt(
        self,
        scene: SceneContext,
        *,
        multi_indices: list[int] | None = None,
        weighted_indices: list[int] | None = None,
    ) -> _GTSimResult | None:
        """Simulate GT trajectory once, compute all metrics."""
        if scene.gt_trajectory is None:
            return None

        gt_trajectory = np.asarray(scene.gt_trajectory, dtype=np.float64)
        if gt_trajectory.ndim != 2 or gt_trajectory.shape[-1] not in (2, 3) or gt_trajectory.shape[0] == 0:
            return None

        try:
            gt_waypoints = gt_trajectory[None, ...]
            if gt_waypoints.shape[1] == self.REQUIRED_NUM_WAYPOINTS:
                gt_proposals = self._build_proposals(gt_waypoints, scene)
            else:
                gt_proposals = self._waypoints_to_proposals(gt_waypoints, scene.ego_state)

            gt_simulated = self._simulator.simulate_proposals(
                ego_state=scene.ego_state, proposals=gt_proposals, observation=scene.observation,
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

    # ── Human penalty filter ──────────────────────────────────────────

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
        overridden: list[str] = []
        for idx, name in self._MULTI_METRIC_NAMES.items():
            if gt_result.multi_metrics[idx] == 0.0:
                multi_metrics[:, idx] = 1.0
                overridden.append(name)
        for idx, name in self._WEIGHTED_METRIC_NAMES.items():
            if idx == WeightedMetricIndex.PROGRESS:
                continue
            if gt_result.weighted_metrics[idx] == 0.0:
                weighted_metrics[:, idx] = 1.0
                overridden.append(name)
        return overridden

    # ── RL continuous: safety layer ───────────────────────────────────

    def _nc_continuous(
        self,
        simulated_states: np.ndarray,
        ego_polygons: np.ndarray,
        ego_areas: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig,
    ) -> np.ndarray:
        del rl_config

        batch_size = len(ego_polygons)
        num_steps = ego_polygons.shape[1]

        collision_records: list[dict[str, tuple[float, float]]] = [{} for _ in range(batch_size)]
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

                    if token not in collision_records[pi]:
                        collision_type = self._classify_collision_type(
                            simulated_states[pi, time_idx], ego_poly,
                            track_polygon, token, scene, time_idx,
                        )
                        if collision_type in (CollisionType.ACTIVE_REAR_BUMPER, CollisionType.STOPPED_EGO_OPEN):
                            forgiven[pi].add(token)
                            continue
                        at_fault_floor = self._collision_penalty(
                            collision_type, ego_areas[pi, time_idx], token, scene,
                        )
                        if at_fault_floor >= 1.0:
                            forgiven[pi].add(token)
                            continue
                        collision_records[pi][token] = (at_fault_floor, 1.0)

                    overlap = ego_poly.intersection(track_polygon).area
                    severity = min(overlap / ego_area, 1.0)
                    floor, cum_prod = collision_records[pi][token]
                    collision_records[pi][token] = (floor, cum_prod * (1.0 - severity))

        scores = np.ones(batch_size, dtype=np.float64)
        for pi in range(batch_size):
            for token, (at_fault_floor, cum_prod) in collision_records[pi].items():
                penalty = at_fault_floor + (1.0 - at_fault_floor) * cum_prod
                scores[pi] = min(scores[pi], penalty)
        return scores

    _DAC_STRUCTURAL_LAYERS = frozenset({"roadblock", "intersection"})

    def _dac_continuous(
        self, ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
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
                    scores[proposal_idx] = 0.0
                    break

                local_union = unary_union([dm._polygons[j] for j in local])
                inside = float(ego_poly.intersection(local_union).area)
                coverage = min(inside / ego_area, 1.0)

                scores[proposal_idx] *= coverage
                if scores[proposal_idx] < 1e-6:
                    scores[proposal_idx] = 0.0
                    break
        return scores

    def _ddc_continuous(
        self, ego_coords: np.ndarray, ego_areas: np.ndarray,
        scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        oncoming_progress = np.zeros(centers.shape[:2], dtype=np.float64)
        oncoming_progress[:, 1:] = np.linalg.norm(centers[:, 1:] - centers[:, :-1], axis=-1)

        not_oncoming = ~ego_areas[:, :, EgoAreaIndex.ONCOMING_TRAFFIC]
        oncoming_progress[not_oncoming] = 0.0

        horizon_steps = max(
            int(round(rl_config.driving_direction_compliance_threshold / self._dt(scene))), 1,
        )
        rolling_progress = np.zeros_like(oncoming_progress)
        for time_idx in range(oncoming_progress.shape[1]):
            start_idx = max(0, time_idx - horizon_steps)
            rolling_progress[:, time_idx] = oncoming_progress[:, start_idx: time_idx + 1].sum(axis=1)

        max_progress = rolling_progress.max(axis=1)
        lo = rl_config.driving_direction_compliance_threshold
        hi = rl_config.driving_direction_violation_threshold
        return np.clip(1.0 - (max_progress - lo) / (hi - lo), 0.0, 1.0)

    def _tlc_continuous(
        self, ego_polygons: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        batch_size = len(ego_polygons)
        min_distances = np.full(batch_size, np.inf, dtype=np.float64)
        margin = rl_config.tlc_margin

        for time_idx in range(ego_polygons.shape[1]):
            red_light_map = self._get_red_light_map(scene, time_idx)
            if red_light_map is None:
                continue
            for proposal_idx in range(batch_size):
                if min_distances[proposal_idx] == 0.0:
                    continue
                ego_poly = ego_polygons[proposal_idx, time_idx]
                if red_light_map.intersects(ego_poly)[0]:
                    for token in red_light_map.get_colliding_tokens(ego_poly):
                        if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                            min_distances[proposal_idx] = 0.0
                            break
                else:
                    buffered = ego_poly.buffer(margin)
                    if red_light_map.intersects(buffered)[0]:
                        for token in red_light_map.get_colliding_tokens(buffered):
                            if token.startswith(RED_LIGHT_TOKEN_PREFIX):
                                dist = ego_poly.distance(red_light_map[token])
                                min_distances[proposal_idx] = min(min_distances[proposal_idx], dist)

        min_distances = np.where(np.isinf(min_distances), margin * 2.0, min_distances)
        return np.clip(min_distances / margin, 0.0, 1.0)

    # ── RL continuous: performance layer ──────────────────────────────

    def _ep_continuous(
        self, ego_coords: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
        *,
        gt_masked_progress: float | None = None,
    ) -> np.ndarray:
        """Continuous ego progress, normalized by GT masked progress.

        Fallback chain:
            1. gt_masked_progress param (from caller / online computation)
            2. scene.gt_masked_progress (precomputed in warmup)
            3. progress_distance_threshold (5m, no GT available)
        """
        raw_progress = self._progress(ego_coords, scene)
        denominator = gt_masked_progress or scene.gt_masked_progress
        if denominator is not None and denominator > rl_config.progress_distance_threshold:
            return np.clip(raw_progress / denominator, 0.0, 1.0)
        return np.clip(raw_progress / rl_config.progress_distance_threshold, 0.0, 1.0)

    def _ttc_continuous(
        self, simulated_states: np.ndarray, ego_coords: np.ndarray,
        ego_areas: np.ndarray, scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
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
                            first_violation_time[proposal_idx] = float(time_idx) * dt
                        else:
                            collided_track_ids[proposal_idx].add(token)

        ttc_seconds = np.where(np.isinf(first_violation_time), rl_config.ttc_horizon, first_violation_time)
        return np.clip(ttc_seconds / rl_config.ttc_horizon, 0.0, 1.0)

    def _lk_continuous(
        self, ego_coords: np.ndarray, ego_areas: np.ndarray,
        scene: SceneContext, rl_config: RLScorerConfig,
    ) -> np.ndarray:
        centers = ego_coords[:, :, BBCoordsIndex.CENTER, :]
        batch_size = len(ego_coords)
        scores = np.ones(batch_size, dtype=np.float64)

        for proposal_idx in range(batch_size):
            deviations = []
            for time_idx in range(ego_coords.shape[1]):
                if ego_areas[proposal_idx, time_idx, EgoAreaIndex.IN_INTERSECTION]:
                    continue
                dev = Point(*centers[proposal_idx, time_idx]).distance(scene.centerline.linestring)
                deviations.append(dev)
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
        self, simulated_states: np.ndarray, scene: SceneContext,
    ) -> np.ndarray:
        """Continuous comfort metric using max violation ratio.

        Aligned with V1: uses only simulated states, no past_states.
        """
        scores = np.ones(len(simulated_states), dtype=np.float64)
        dt = self._dt(scene)
        for proposal_idx in range(len(simulated_states)):
            states = simulated_states[proposal_idx]
            time_points_s = np.arange(len(states), dtype=np.float64) * dt
            scores[proposal_idx] = ego_comfort_violation(states, time_points_s)
        return scores

    # ── Geometric / time utilities ────────────────────────────────────

    def _is_ttc_violation(
        self, ego_state: np.ndarray, track_polygon: BaseGeometry,
        ego_area: np.ndarray, ego_in_intersection: bool,
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
            idx for idx, layer_name in enumerate(drivable_area_map.types)
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
                x_coords, y_coords,
            )
        return membership.reshape((len(selected_indices), *points.shape[:-1]))

    def _points_in_route_lanes(
        self, points: np.ndarray, drivable_area_map: DrivableMap,
        route_lane_ids: Sequence[str],
    ) -> np.ndarray:
        route_lane_id_set = set(route_lane_ids)
        if not route_lane_id_set:
            return np.zeros(points.shape[:-1], dtype=bool)
        selected_indices = [
            idx for idx, (token, layer_name) in enumerate(
                zip(drivable_area_map.tokens, drivable_area_map.types)
            )
            if token in route_lane_id_set
            and _layer_name_to_enum(layer_name) in {SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR}
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
                x_coords, y_coords,
            )
        return membership.any(axis=0).reshape(points.shape[:-1])

    @staticmethod
    def _relative_point_in_ego_frame(ego_state: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
        dx = point_xy[0] - ego_state[StateIndex.X]
        dy = point_xy[1] - ego_state[StateIndex.Y]
        theta = float(ego_state[StateIndex.HEADING])
        return np.asarray(
            [dx * np.cos(theta) + dy * np.sin(theta), -dx * np.sin(theta) + dy * np.cos(theta)],
            dtype=np.float64,
        )

    def _is_track_behind_ego(self, ego_state: np.ndarray, track_polygon: BaseGeometry) -> bool:
        relative = self._relative_point_in_ego_frame(
            ego_state, np.asarray(track_polygon.centroid.coords[0], dtype=np.float64),
        )
        return bool(relative[0] < 0.0)

    @staticmethod
    def _dt(scene: SceneContext) -> float:
        return float(scene.observation.interval_time)

    @staticmethod
    def _local_time_idx(scene: SceneContext, time_idx: int) -> int:
        global_to_local = scene.observation.global_to_local_idcs
        clamped = min(max(time_idx, 0), len(global_to_local) - 1)
        return int(global_to_local[clamped])

    def _get_occupancy_map(self, scene: SceneContext, time_idx: int) -> OccupancyMap | None:
        return scene.observation.get_occupancy_map(self._local_time_idx(scene, time_idx))

    def _get_red_light_map(self, scene: SceneContext, time_idx: int) -> OccupancyMap | None:
        return scene.observation.get_red_light_map(self._local_time_idx(scene, time_idx))


def _layer_name_to_enum(layer_name: str) -> SemanticMapLayer | None:
    return _LAYER_NAME_TO_ENUM.get(_normalize_layer_name(layer_name))

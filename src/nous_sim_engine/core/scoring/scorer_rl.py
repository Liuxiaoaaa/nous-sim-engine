from __future__ import annotations

from typing import List

import numpy as np

from ..enums import MultiMetricIndex
from ..geometry import state_to_coords, coords_to_polygons
from ..types import SceneContext, RLScoringResult
from .base import ScorerBase, RLScorerConfig


class RLScorer(ScorerBase):
    """RL continuous reward scorer with 8 independent sub-rewards.

    Aggregation: safety_gate ^ alpha × weighted_avg(EP, TTC, LK, HC)
    where safety_gate = NC × DAC × DDC × TLC
    """

    def score(
        self,
        waypoints_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
    ) -> RLScoringResult:
        return self.score_batch(waypoints_xy[None, ...], scene, rl_config)[0]

    def _resolve_gt_masked_progress(self, scene: SceneContext) -> float | None:
        """Get gt_masked_progress: precomputed (warmup) or online fallback.

        Fallback chain:
            1. scene.gt_masked_progress (precomputed in warmup)
            2. Online GT simulation → gt_progress × gt_NC × gt_DAC
            3. None (no GT trajectory available)
        """
        if scene.gt_masked_progress is not None:
            return scene.gt_masked_progress

        # Online fallback: simulate GT and compute masked progress
        gt_result = self._simulate_and_score_gt(scene)
        if gt_result is None:
            return None

        gt_nc = float(gt_result.multi_metrics[MultiMetricIndex.NO_COLLISION])
        gt_dac = float(gt_result.multi_metrics[MultiMetricIndex.DRIVABLE_AREA])
        gt_masked = gt_result.progress * gt_nc * gt_dac

        # Cache back to scene for subsequent calls in same request
        scene.gt_masked_progress = gt_masked
        if scene.gt_progress is None:
            scene.gt_progress = gt_result.progress

        return gt_masked

    def score_batch(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
        *,
        include_ego: bool = False,
    ) -> List[RLScoringResult]:
        rl_config = rl_config or RLScorerConfig.v1()

        batch_waypoints = self._coerce_trajectories(trajectories_xy)
        proposals = self._build_proposals(batch_waypoints, scene, include_ego=include_ego)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state, proposals=proposals, observation=scene.observation,
        )
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = self._calculate_ego_areas(ego_coords, scene)

        # Safety layer: continuous or discrete
        if rl_config.safety_mode == "continuous":
            collision_result = self._collision_metrics(simulated_states, ego_polygons, ego_areas, scene, rl_config)
            nc = collision_result["nc"]
            max_collision_overlap = collision_result["max_collision_overlap"]
            max_collision_penetration_distance = collision_result["max_collision_penetration_distance"]
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
            # Raw physics: obstacle and boundary distances + normalization denominators
            min_obstacle_dist = self._min_obstacle_distance(ego_polygons, scene, rl_config)
            min_boundary_dist = self._min_boundary_distance(ego_coords, scene, rl_config)
            mean_obstacle_dist_5m = self._mean_obstacle_distance_within(ego_polygons, scene, rl_config)
            half_lane_w = self._half_lane_width(scene)
        else:
            nc = self._no_at_fault_collision(simulated_states, ego_polygons, ego_areas, scene)
            dac = self._drivable_area_compliance(ego_areas)
            max_collision_overlap = np.where(nc < 1.0, 1.0 - nc, 0.0)
            max_collision_penetration_distance = np.zeros(len(batch_waypoints), dtype=np.float64)
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
            min_obstacle_dist = np.zeros(len(batch_waypoints), dtype=np.float64)
            min_boundary_dist = np.zeros(len(batch_waypoints), dtype=np.float64)
            mean_obstacle_dist_5m = np.full(len(batch_waypoints), 5.0, dtype=np.float64)
            half_lane_w = 2.0

        # Performance layer: always continuous
        gt_masked = self._resolve_gt_masked_progress(scene)
        ep = self._ep_continuous(ego_coords, scene, rl_config, gt_masked_progress=gt_masked)
        ttc = self._ttc_continuous(simulated_states, ego_coords, ego_areas, scene, rl_config)
        lk = (
            self._lk_continuous(ego_coords, ego_areas, scene, rl_config)
            if rl_config.lk_weight > 0.0
            else np.ones(len(batch_waypoints), dtype=np.float64)
        )
        hc = self._hc_continuous(simulated_states, scene)
        lat_offset_signed = self._lateral_offset_signed(ego_coords, ego_areas, scene)
        lat_offset_change = self._lateral_offset_change(ego_coords, ego_areas, scene)
        centerline_geom = self._centerline_geometry(ego_coords, ego_areas, scene)
        boundary_geom = self._boundary_geometry(ego_coords, scene, rl_config)
        topology_geom = self._topology_occupancy(ego_areas)

        # Discrete mode: raw progress (meters) + safety gate + PDMS monitoring
        if rl_config.safety_mode == "discrete":
            raw_progress_meters = self._progress(ego_coords, scene)
            # V1 safety gate: NC × DAC only
            safety_gate_arr = nc * dac
            # Quality-modulated progress: raw_meters × TTC × HC, gated by safety
            raw_progress_gated = raw_progress_meters * safety_gate_arr * ttc * hc
            # PDMS v1 monitoring: (NC × DAC) × (5*EP + 5*TTC + 2*HC) / 12
            perf_weighted = (5.0 * ep + 5.0 * ttc + 2.0 * hc) / 12.0
            pdms_arr = safety_gate_arr * perf_weighted

        # Aggregation: soft safety gate × weighted performance average
        safety_product = nc * dac * ddc * tlc
        alpha = rl_config.safety_gate_alpha
        safety_gate = np.power(np.clip(safety_product, 0.0, 1.0), alpha)

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
            result_kwargs = dict(
                rl_score=float(rl_scores[i]),
                no_at_fault_collisions=float(nc[i]),
                drivable_area_compliance=float(dac[i]),
                driving_direction_compliance=float(ddc[i]),
                traffic_light_compliance=float(tlc[i]),
                ego_progress=float(ep[i]),
                time_to_collision=float(ttc[i]),
                lane_keeping=float(lk[i]),
                history_comfort=float(hc[i]),
                max_collision_overlap=float(max_collision_overlap[i]),
                max_collision_penetration_distance=float(max_collision_penetration_distance[i]),
                min_obstacle_distance=float(min_obstacle_dist[i]),
                min_boundary_distance=float(min_boundary_dist[i]),
                mean_obstacle_distance_5m=float(mean_obstacle_dist_5m[i]),
                half_lane_width=float(half_lane_w),
                lateral_offset_signed=float(lat_offset_signed[i]),
                lateral_offset_change=float(lat_offset_change[i]),
                centerline_lateral_offset_start_signed=float(centerline_geom["start_signed"][i]),
                centerline_lateral_offset_end_signed=float(centerline_geom["end_signed"][i]),
                centerline_distance_mean=float(centerline_geom["mean_distance"][i]),
                centerline_distance_max=float(centerline_geom["max_distance"][i]),
                local_centerline_points=centerline_geom["local_centerline_points"],
                boundary_distance_start=float(boundary_geom["start"][i]),
                boundary_distance_end=float(boundary_geom["end"][i]),
                boundary_distance_min=float(boundary_geom["min"][i]),
                boundary_distance_mean=float(boundary_geom["mean"][i]),
                boundary_distances=[float(x) for x in boundary_geom["distances"][i].tolist()],
                boundary_side=boundary_geom["side"][i],
                in_intersection_fraction=float(topology_geom["in_intersection_fraction"][i]),
                oncoming_fraction=float(topology_geom["oncoming_fraction"][i]),
                non_drivable_fraction=float(topology_geom["non_drivable_fraction"][i]),
                multiple_lanes_fraction=float(topology_geom["multiple_lanes_fraction"][i]),
                in_intersection_flags=[bool(x) for x in topology_geom["in_intersection_flags"][i].tolist()],
                oncoming_flags=[bool(x) for x in topology_geom["oncoming_flags"][i].tolist()],
                non_drivable_flags=[bool(x) for x in topology_geom["non_drivable_flags"][i].tolist()],
                multiple_lanes_flags=[bool(x) for x in topology_geom["multiple_lanes_flags"][i].tolist()],
            )
            if rl_config.safety_mode == "discrete":
                result_kwargs.update(
                    safety_gate=float(safety_gate_arr[i]),
                    raw_progress=float(raw_progress_gated[i]),
                    pdms_score=float(pdms_arr[i]),
                )
            results.append(RLScoringResult(**result_kwargs))
        return results

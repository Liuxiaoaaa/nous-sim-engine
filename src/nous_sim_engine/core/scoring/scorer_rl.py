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
    ) -> List[RLScoringResult]:
        rl_config = rl_config or RLScorerConfig.v1()

        batch_waypoints = self._coerce_trajectories(trajectories_xy)
        proposals = self._build_proposals(batch_waypoints, scene)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state, proposals=proposals, observation=scene.observation,
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
        gt_masked = self._resolve_gt_masked_progress(scene)
        ep = self._ep_continuous(ego_coords, scene, rl_config, gt_masked_progress=gt_masked)
        ttc = self._ttc_continuous(simulated_states, ego_coords, ego_areas, scene, rl_config)
        lk = (
            self._lk_continuous(ego_coords, ego_areas, scene, rl_config)
            if rl_config.lk_weight > 0.0
            else np.ones(len(batch_waypoints), dtype=np.float64)
        )
        hc = self._hc_continuous(simulated_states, scene)

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
            results.append(RLScoringResult(
                rl_score=float(rl_scores[i]),
                no_at_fault_collisions=float(nc[i]),
                drivable_area_compliance=float(dac[i]),
                driving_direction_compliance=float(ddc[i]),
                traffic_light_compliance=float(tlc[i]),
                ego_progress=float(ep[i]),
                time_to_collision=float(ttc[i]),
                lane_keeping=float(lk[i]),
                history_comfort=float(hc[i]),
            ))
        return results

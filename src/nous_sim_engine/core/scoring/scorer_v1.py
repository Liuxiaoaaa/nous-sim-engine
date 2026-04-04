from __future__ import annotations

from typing import List

import numpy as np

from ..enums import MultiMetricIndex, WeightedMetricIndex
from ..geometry import state_to_coords, coords_to_polygons
from ..types import SceneContext, ScoringResult
from .base import ScorerBase


class PDMScorerV1(ScorerBase):
    """NavSim v1 PDMS scorer, strictly aligned with recogdrive/navsim.

    Key differences from V2:
    - NC uses observation-based collision type (treats all objects as agents)
    - No TLC, no LK metrics
    - Comfort uses simulated states only (no ego_past_states)
    - Progress: batch max normalization on masked_progress (raw * multi)
    - No human penalty filter
    - Aggregation: (NC * DAC) * (5*EP + 5*TTC + 2*HC) / 12
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._progress_distance_threshold = 5.0

    def score(self, waypoints_xy: np.ndarray, scene: SceneContext) -> ScoringResult:
        return self.score_batch(waypoints_xy[None, ...], scene)[0]

    def score_batch(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
    ) -> List[ScoringResult]:
        batch_waypoints = self._coerce_trajectories(trajectories_xy)
        proposals = self._build_proposals(batch_waypoints, scene)
        simulated_states = self._simulator.simulate_proposals(
            ego_state=scene.ego_state, proposals=proposals, observation=scene.observation,
        )
        ego_coords = state_to_coords(simulated_states, self._vehicle)
        ego_polygons = coords_to_polygons(ego_coords)
        ego_areas = self._calculate_ego_areas(ego_coords, scene)

        # V1: only NC * DAC (2 multiplicative metrics)
        multi_metrics = np.ones((len(batch_waypoints), len(MultiMetricIndex)), dtype=np.float64)
        weighted_metrics = np.ones((len(batch_waypoints), len(WeightedMetricIndex)), dtype=np.float64)

        # NC: use observation-based collision types (recogride behavior)
        multi_metrics[:, MultiMetricIndex.NO_COLLISION] = self._no_at_fault_collision(
            simulated_states, ego_polygons, ego_areas, scene, use_observation_types=True,
        )
        multi_metrics[:, MultiMetricIndex.DRIVABLE_AREA] = self._drivable_area_compliance(ego_areas)
        # DDC computed but NOT used in aggregation (weight=0)
        multi_metrics[:, MultiMetricIndex.DRIVING_DIRECTION] = self._driving_direction_compliance(
            ego_coords, ego_areas, scene,
        )
        # TLC computed but NOT used in aggregation
        multi_metrics[:, MultiMetricIndex.TRAFFIC_LIGHT] = self._traffic_light_compliance(
            ego_polygons, scene,
        )

        progress_raw = self._progress(ego_coords, scene)

        # GT masked progress for normalization
        # recogdrive puts GT in batch; max(masked) = gt_raw * gt_NC * gt_DAC
        gt_result = self._simulate_and_score_gt(scene)
        gt_masked_progress: float | None = None
        if gt_result is not None:
            gt_multi = float(
                gt_result.multi_metrics[MultiMetricIndex.NO_COLLISION]
                * gt_result.multi_metrics[MultiMetricIndex.DRIVABLE_AREA]
            )
            gt_masked_progress = gt_result.progress * gt_multi

        weighted_metrics[:, WeightedMetricIndex.PROGRESS] = self._normalize_progress_v1(
            progress_raw, multi_metrics, gt_masked_progress=gt_masked_progress,
        )
        weighted_metrics[:, WeightedMetricIndex.TTC] = self._time_to_collision(
            simulated_states, ego_coords, ego_areas, scene,
        )
        weighted_metrics[:, WeightedMetricIndex.LANE_KEEPING] = self._lane_keeping(
            ego_coords, ego_areas, scene,
        )
        # V1 comfort: only simulated states, no past_states
        weighted_metrics[:, WeightedMetricIndex.COMFORT] = self._history_comfort(
            simulated_states, scene, use_past_states=False,
        )

        scores = self._aggregate_v1(multi_metrics, weighted_metrics)

        results: List[ScoringResult] = []
        for proposal_idx, score in enumerate(scores):
            results.append(ScoringResult(
                scoring_version="v1",
                pdm_score=float(score),
                no_at_fault_collisions=float(multi_metrics[proposal_idx, MultiMetricIndex.NO_COLLISION]),
                drivable_area_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.DRIVABLE_AREA]),
                driving_direction_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.DRIVING_DIRECTION]),
                traffic_light_compliance=float(multi_metrics[proposal_idx, MultiMetricIndex.TRAFFIC_LIGHT]),
                ego_progress=float(weighted_metrics[proposal_idx, WeightedMetricIndex.PROGRESS]),
                time_to_collision=float(weighted_metrics[proposal_idx, WeightedMetricIndex.TTC]),
                lane_keeping=float(weighted_metrics[proposal_idx, WeightedMetricIndex.LANE_KEEPING]),
                history_comfort=float(weighted_metrics[proposal_idx, WeightedMetricIndex.COMFORT]),
            ))
        return results

    def _normalize_progress_v1(
        self,
        progress_raw: np.ndarray,
        multi_metrics: np.ndarray,
        *,
        gt_masked_progress: float | None = None,
    ) -> np.ndarray:
        """V1 progress normalization aligned with recogdrive.

        recogdrive logic (GT is in batch at pred_idx=0):
            masked = raw * multi           (multi = NC * DAC for v1)
            max_masked = max(masked)        (= gt_raw * gt_NC * gt_DAC)
            if max_masked > 5m:
                norm = masked / max_masked
            else:
                norm = 1.0 where multi > 0, else 0.0

        gt_masked_progress = gt_raw * gt_NC * gt_DAC, computed from GT simulation.
        """
        multi_prod = multi_metrics[
            :, [MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA]
        ].prod(axis=1)
        masked_progress = progress_raw * multi_prod

        # Prefer GT masked progress (matches recogdrive: max(batch_masked) ≈ gt_masked)
        if gt_masked_progress is not None and gt_masked_progress > self._progress_distance_threshold:
            normalized = np.clip(masked_progress / gt_masked_progress, 0.0, 1.0)
        else:
            # Fallback: batch max
            max_masked = float(masked_progress.max()) if len(masked_progress) > 0 else 0.0
            if max_masked > self._progress_distance_threshold:
                normalized = np.clip(masked_progress / max_masked, 0.0, 1.0)
            else:
                normalized = np.ones_like(progress_raw, dtype=np.float64)
                normalized[multi_prod == 0.0] = 0.0

        return normalized

    @staticmethod
    def _aggregate_v1(multi_metrics: np.ndarray, weighted_metrics: np.ndarray) -> np.ndarray:
        """V1 aggregation: (NC * DAC) * (5*EP + 5*TTC + 2*HC) / 12"""
        multiplicative_scores = multi_metrics[
            :, [MultiMetricIndex.NO_COLLISION, MultiMetricIndex.DRIVABLE_AREA]
        ].prod(axis=1)

        weights = np.array([5.0, 5.0, 0.0, 2.0], dtype=np.float64)  # EP, TTC, LK, HC
        weighted_scores = (weighted_metrics * weights[None, :]).sum(axis=1) / weights.sum()
        return multiplicative_scores * weighted_scores

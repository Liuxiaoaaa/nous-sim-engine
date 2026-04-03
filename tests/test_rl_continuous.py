"""Tests for continuous RL scoring: NC (collision), DAC (sweep area), EP (GT progress).

Uses synthetic scenes from conftest — no MetricCache dependency.
All trajectories are 8 waypoints at 0.5s spacing (ego-relative xy).
"""
from __future__ import annotations

import numpy as np
import pytest

from nous_sim_engine.core.enums import SemanticMapLayer, StateIndex
from nous_sim_engine.core.geometry import PDMPath
from nous_sim_engine.core.observation import PDMObservation
from nous_sim_engine.core.occupancy import DrivableMap
from nous_sim_engine.core.scorer import PDMScorer, RLScorerConfig
from nous_sim_engine.core.types import SceneContext


# ── Helpers ──────────────────────────────────────────────────────────────

NUM_WP = 8  # 8 waypoints @ 0.5s = 4s horizon


def _traj(xs: list[float], ys: list[float]) -> np.ndarray:
    """Build (8, 2) trajectory from x/y lists."""
    assert len(xs) == len(ys) == NUM_WP
    return np.column_stack([xs, ys])


def _straight(dx_per_step: float = 2.5, y: float = 0.0) -> np.ndarray:
    """Straight trajectory along x-axis."""
    xs = [dx_per_step * i for i in range(1, NUM_WP + 1)]
    return _traj(xs, [y] * NUM_WP)


def _build_scene(
    *,
    ego_speed: float = 5.0,
    obstacle_x: float = 30.0,
    obstacle_y: float = 0.0,
    obstacle_vx: float = 0.0,
    road_half_width: float = 4.0,
    road_length: float = 100.0,
    static_obstacle: bool = False,
    no_obstacle: bool = False,
    num_steps: int = 50,
    gt_progress: float | None = None,
) -> SceneContext:
    """Minimal scene builder with configurable obstacles."""
    from shapely.geometry import box

    ego_state = np.zeros(StateIndex.size(), dtype=np.float64)
    ego_state[StateIndex.VELOCITY_X] = ego_speed

    ego_past_states = np.zeros((20, StateIndex.size()), dtype=np.float64)
    for i in range(20):
        t = -(20 - i) * 0.1
        ego_past_states[i, StateIndex.X] = ego_speed * t
        ego_past_states[i, StateIndex.VELOCITY_X] = ego_speed

    road_polygon = box(-10.0, -road_half_width, road_length, road_half_width)
    drivable_map = DrivableMap(
        tokens=["lane_0"],
        types=[SemanticMapLayer.LANE],
        polygons=np.array([road_polygon], dtype=object),
    )
    centerline = PDMPath(np.array([[-10.0, 0.0], [road_length, 0.0]], dtype=np.float64))
    observation = PDMObservation(num_steps=num_steps, interval_time=0.1)

    dynamic_coords = np.empty((0, 4, 2), dtype=np.float64)
    dynamic_tokens: list[str] = []
    dynamic_vels = np.empty((0, 2), dtype=np.float64)
    static_coords = np.empty((0, 4, 2), dtype=np.float64)
    static_tokens: list[str] = []
    track_types: dict[str, str] = {}

    if not no_obstacle:
        obs_corners = np.array([
            [obstacle_x - 2.0, obstacle_y + 1.0],
            [obstacle_x + 2.0, obstacle_y + 1.0],
            [obstacle_x + 2.0, obstacle_y - 1.0],
            [obstacle_x - 2.0, obstacle_y - 1.0],
        ], dtype=np.float64)[None, :, :]

        if static_obstacle:
            static_coords = obs_corners
            static_tokens = ["static_1"]
            track_types["static_1"] = "static"
        else:
            dynamic_coords = obs_corners
            dynamic_tokens = ["vehicle_1"]
            dynamic_vels = np.array([[obstacle_vx, 0.0]], dtype=np.float64)
            track_types["vehicle_1"] = "agent"

    observation.update(
        static_coords=static_coords,
        static_tokens=static_tokens,
        dynamic_coords=dynamic_coords,
        dynamic_tokens=dynamic_tokens,
        dynamic_velocities=dynamic_vels,
        red_light_coords=np.empty((0, 4, 2), dtype=np.float64),
        red_light_tokens=[],
    )

    ctx = SceneContext(
        scene_token="test",
        log_name="test",
        ego_state=ego_state,
        ego_past_states=ego_past_states,
        observation=observation,
        drivable_area_map=drivable_map,
        route_lane_ids={"lane_0"},
        centerline=centerline,
        track_object_types=track_types,
    )
    if gt_progress is not None:
        ctx.gt_progress = gt_progress
    return ctx


def _score_rl(scene: SceneContext, traj: np.ndarray, config: RLScorerConfig | None = None):
    scorer = PDMScorer()
    return scorer.score_batch_for_rl(traj[None], scene, config)[0]


# ═════════════════════════════════════════════════════════════════════════
# 1. NC Continuous — collision classification + overlap severity
# ═════════════════════════════════════════════════════════════════════════


class TestNCContinuous:
    """Test continuous no-collision metric with full at-fault classification."""

    def test_no_collision_returns_1(self):
        """No collision → NC = 1.0"""
        scene = _build_scene(obstacle_x=50.0)
        result = _score_rl(scene, _straight(2.0))
        assert result.no_at_fault_collisions == 1.0

    def test_head_on_collision_returns_below_1(self):
        """Head-on into obstacle → at-fault, NC < 1.0"""
        scene = _build_scene(obstacle_x=10.0, obstacle_vx=0.0)
        traj = _straight(3.0)  # drives fast toward obstacle at x=10
        result = _score_rl(scene, traj)
        assert result.no_at_fault_collisions < 1.0, (
            f"Expected NC penalty for head-on collision, got {result.no_at_fault_collisions}"
        )

    def test_static_obstacle_max_penalty_is_05(self):
        """Static obstacle → at-fault max penalty capped at 0.5"""
        scene = _build_scene(obstacle_x=15.0, static_obstacle=True)
        traj = _straight(2.5)
        result = _score_rl(scene, traj)
        assert result.no_at_fault_collisions >= 0.4, (
            f"Static obstacle NC should have floor ~0.5, got {result.no_at_fault_collisions}"
        )

    def test_nc_is_monotonic_with_overlap(self):
        """Larger overlap → lower NC: direct hit worse than glancing."""
        scene = _build_scene(obstacle_x=15.0, obstacle_y=0.0)
        # Direct: y=0 → full overlap
        direct = _score_rl(scene, _straight(2.5, y=0.0))
        # Glancing: y=1.5 → partial overlap
        glancing = _score_rl(scene, _straight(2.5, y=1.5))
        assert glancing.no_at_fault_collisions >= direct.no_at_fault_collisions, (
            f"Glancing ({glancing.no_at_fault_collisions}) should be >= direct ({direct.no_at_fault_collisions})"
        )

    def test_nc_discrete_is_binary(self):
        """Discrete NC must be exactly 0.0 or 0.5 or 1.0."""
        config = RLScorerConfig(safety_mode="discrete")
        scene = _build_scene(obstacle_x=15.0)
        result = _score_rl(scene, _straight(2.5), config)
        assert result.no_at_fault_collisions in (0.0, 0.5, 1.0), (
            f"Discrete NC should be binary-ish, got {result.no_at_fault_collisions}"
        )

    def test_nc_range_is_01(self):
        """NC must be in [0, 1] for all trajectories."""
        scene = _build_scene(obstacle_x=15.0)
        for y_off in np.linspace(-3.0, 3.0, 7):
            result = _score_rl(scene, _straight(2.5, y=float(y_off)))
            assert 0.0 <= result.no_at_fault_collisions <= 1.0, (
                f"NC out of range for y_off={y_off}: {result.no_at_fault_collisions}"
            )


# ═════════════════════════════════════════════════════════════════════════
# 2. DAC Continuous — sweep area ratio
# ═════════════════════════════════════════════════════════════════════════


class TestDACContinuous:
    """Test continuous drivable area compliance with sweep-area algorithm."""

    def test_on_road_returns_1(self):
        """Fully on-road trajectory → DAC = 1.0"""
        scene = _build_scene(no_obstacle=True)
        result = _score_rl(scene, _straight(2.0, y=0.0))
        assert result.drivable_area_compliance == 1.0

    def test_fully_offroad_returns_low(self):
        """Trajectory far off road → DAC < 0.5"""
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        # y offset = 20m → mostly outside road [-4, 4]
        # Note: LQR starts from y=0, so early sweep is on-road; expect DAC < 0.5
        traj = _straight(2.0, y=20.0)
        result = _score_rl(scene, traj)
        assert result.drivable_area_compliance < 0.5, (
            f"Expected low DAC for far offroad, got {result.drivable_area_compliance}"
        )

    def test_partial_offroad_is_intermediate(self):
        """Trajectory with edge off road → 0 < DAC < 1"""
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        # y=3.5 → vehicle corner brushes the edge (vehicle ~2m wide)
        traj = _straight(2.0, y=3.5)
        result = _score_rl(scene, traj)
        assert 0.0 < result.drivable_area_compliance < 1.0, (
            f"Expected intermediate DAC for edge case, got {result.drivable_area_compliance}"
        )

    def test_dac_is_monotonic_with_offroad_distance(self):
        """More off-road → lower DAC."""
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        dac_near = _score_rl(scene, _straight(2.0, y=3.0)).drivable_area_compliance
        dac_far = _score_rl(scene, _straight(2.0, y=6.0)).drivable_area_compliance
        assert dac_near > dac_far, f"Near ({dac_near}) should be > far ({dac_far})"

    def test_dac_is_continuous(self):
        """Small y change → small DAC change (no large jumps)."""
        scene = _build_scene(road_half_width=4.0, no_obstacle=True)
        y_values = np.linspace(0.0, 6.0, 13)
        dac_values = []
        for y in y_values:
            r = _score_rl(scene, _straight(2.0, y=float(y)))
            dac_values.append(r.drivable_area_compliance)

        # Check no jump > 0.5 between adjacent y values (0.5m spacing)
        for i in range(1, len(dac_values)):
            diff = abs(dac_values[i] - dac_values[i - 1])
            assert diff < 0.5, (
                f"DAC jump too large between y={y_values[i-1]:.1f} ({dac_values[i-1]:.3f}) "
                f"and y={y_values[i]:.1f} ({dac_values[i]:.3f}), diff={diff:.3f}"
            )

    def test_dac_range_is_01(self):
        """DAC must be in [0, 1] for all trajectories."""
        scene = _build_scene(no_obstacle=True)
        for y in [-10.0, -4.0, 0.0, 4.0, 10.0]:
            result = _score_rl(scene, _straight(2.0, y=y))
            assert 0.0 <= result.drivable_area_compliance <= 1.0


# ═════════════════════════════════════════════════════════════════════════
# 3. EP Continuous — GT-progress normalization
# ═════════════════════════════════════════════════════════════════════════


class TestEPContinuous:
    """Test continuous ego progress with GT normalization."""

    def test_gt_normalized_progress(self):
        """When gt_progress is set, EP = raw_progress / gt_progress."""
        scene = _build_scene(no_obstacle=True, gt_progress=20.0)
        result = _score_rl(scene, _straight(2.0))
        # Trajectory advances ~16m in 4s → EP ≈ 16/20 = 0.8
        assert 0.5 < result.ego_progress <= 1.0, (
            f"Expected EP ~0.8 with GT norm, got {result.ego_progress}"
        )

    def test_fallback_to_threshold_when_no_gt(self):
        """Without gt_progress, EP = raw_progress / threshold (5.0)."""
        scene = _build_scene(no_obstacle=True, gt_progress=None)
        result = _score_rl(scene, _straight(2.0))
        # Trajectory advances >>5m, so EP clipped to 1.0
        assert result.ego_progress == 1.0

    def test_ep_clipped_to_01(self):
        """EP must be in [0, 1] even with gt_progress < raw progress."""
        scene = _build_scene(no_obstacle=True, gt_progress=1.0)
        result = _score_rl(scene, _straight(2.0))
        assert 0.0 <= result.ego_progress <= 1.0

    def test_slow_progress_is_lower(self):
        """Slower trajectory → lower EP."""
        scene = _build_scene(no_obstacle=True, gt_progress=20.0)
        fast = _score_rl(scene, _straight(3.0)).ego_progress
        slow = _score_rl(scene, _straight(1.0)).ego_progress
        assert fast > slow, f"Fast EP ({fast}) should be > slow EP ({slow})"


# ═════════════════════════════════════════════════════════════════════════
# 4. Config alignment: v1 defaults
# ═════════════════════════════════════════════════════════════════════════


class TestRLConfig:
    """Test RLScorerConfig.v1() defaults and weight-zero gating."""

    def test_v1_inactive_weights_are_zero(self):
        cfg = RLScorerConfig.v1()
        assert cfg.ddc_weight == 0.0
        assert cfg.tlc_weight == 0.0
        assert cfg.lk_weight == 0.0

    def test_v1_active_weights_match_plan(self):
        cfg = RLScorerConfig.v1()
        assert cfg.nc_weight == 5.0
        assert cfg.dac_weight == 3.0
        assert cfg.ep_weight == 5.0
        assert cfg.ttc_weight == 5.0
        assert cfg.hc_weight == 2.0

    def test_weight_zero_skips_computation(self):
        """DDC/TLC/LK = 1.0 when weight = 0 (skipped)."""
        scene = _build_scene(no_obstacle=True)
        config = RLScorerConfig.v1()
        result = _score_rl(scene, _straight(2.0), config)
        assert result.driving_direction_compliance == 1.0
        assert result.traffic_light_compliance == 1.0
        assert result.lane_keeping == 1.0

    def test_default_uses_v1(self):
        """Default RLScorerConfig matches v1() for inactive weights."""
        default = RLScorerConfig()
        assert default.ddc_weight == 0.0
        assert default.tlc_weight == 0.0
        assert default.lk_weight == 0.0


# ═════════════════════════════════════════════════════════════════════════
# 5. Pipeline alignment: RL uses same simulation as PDMS
# ═════════════════════════════════════════════════════════════════════════


class TestPipelineAlignment:
    """Verify RL scoring uses the same _build_proposals path as PDMS."""

    def test_discrete_nc_matches_pdms(self):
        """Discrete RL NC should match PDMS NC (same simulation pipeline)."""
        scene = _build_scene(obstacle_x=50.0)
        traj = _straight(2.0)
        scorer = PDMScorer()

        pdms = scorer.score(traj, scene)
        rl_discrete = scorer.score_batch_for_rl(
            traj[None], scene, RLScorerConfig(safety_mode="discrete")
        )[0]

        assert abs(pdms.no_at_fault_collisions - rl_discrete.no_at_fault_collisions) < 1e-6
        assert abs(pdms.drivable_area_compliance - rl_discrete.drivable_area_compliance) < 1e-6

    def test_all_sub_rewards_in_01(self):
        """All sub-rewards must be in [0, 1] for any trajectory."""
        scene = _build_scene()
        for y in [0.0, 2.0, -2.0, 5.0, -5.0]:
            result = _score_rl(scene, _straight(2.0, y=float(y)))
            for key, val in result.sub_rewards().items():
                assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] for y={y}"

    def test_rl_score_is_soft_gated(self):
        """rl_score = (NC*DAC*DDC*TLC)^alpha × weighted_avg(EP, TTC, LK, HC)."""
        scene = _build_scene(no_obstacle=True)
        config = RLScorerConfig.v1()
        result = _score_rl(scene, _straight(2.0), config)

        sub = result.sub_rewards()
        safety = sub["nc"] * sub["dac"] * sub["ddc"] * sub["tlc"]
        safety_gate = safety ** config.safety_gate_alpha

        perf_w = config.performance_weights
        perf_vals = np.array([sub["ep"], sub["ttc"], sub["lk"], sub["hc"]])
        performance = (perf_vals * perf_w).sum() / perf_w.sum()

        expected = safety_gate * performance
        assert abs(result.rl_score - expected) < 1e-6, (
            f"rl_score ({result.rl_score}) != soft_gate ({expected})"
        )

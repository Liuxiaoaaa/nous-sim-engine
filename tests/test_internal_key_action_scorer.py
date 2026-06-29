from __future__ import annotations

import copy

import numpy as np
import pytest

from nous_sim_engine.core.scoring import InternalKeyActionScorer
from nous_sim_engine.core.types import KeyActionObstacle, SceneContext


def _box(center_x: float, center_y: float, length: float = 4.0, width: float = 2.0) -> np.ndarray:
    half_l = length / 2.0
    half_w = width / 2.0
    return np.asarray(
        [
            [center_x + half_l, center_y + half_w],
            [center_x - half_l, center_y + half_w],
            [center_x - half_l, center_y - half_w],
            [center_x + half_l, center_y - half_w],
        ],
        dtype=np.float64,
    )


def _with_key_obstacles(
    scene: SceneContext,
    obstacles: list[tuple[str, int, float]],
) -> SceneContext:
    copied = copy.deepcopy(scene)
    copied.key_action_obstacles = [
        KeyActionObstacle(
            token=token,
            label=label,
            polygon_coords=_box(x, 0.0),
            object_type="static",
        )
        for token, label, x in obstacles
    ]
    copied.candidate_progress = 25.0
    copied.gt_progress = 10.0
    return copied


def test_internal_key_action_all_nudge_passes(straight_road_scene):
    scene = _with_key_obstacles(straight_road_scene, [("cone_1", 1, 12.0)])
    trajectory = np.asarray([[2.5 * t, 0.0] for t in range(1, 9)], dtype=np.float64)

    result = InternalKeyActionScorer().score(trajectory, scene)

    assert result.sample_valid is True
    assert result.num_relevant_labeled == 1
    assert result.num_key_actions == 1
    assert result.num_key_actions_passed == 1
    assert result.key_action_score == pytest.approx(1.0)
    assert result.progress_norm == pytest.approx(25.0)
    assert result.progress_norm_source == "candidate+logged:candidate"
    assert result.internal_score > 0.0


def test_internal_key_action_no_nudge_gate_blocks_overrun(straight_road_scene):
    scene = _with_key_obstacles(
        straight_road_scene,
        [
            ("cone_1", 1, 12.0),
            ("blocked_car", 0, 24.0),
        ],
    )
    trajectory = np.asarray([[3.75 * t, 0.0] for t in range(1, 9)], dtype=np.float64)

    result = InternalKeyActionScorer().score(trajectory, scene)

    assert result.sample_valid is True
    assert result.first_no_nudge_upper_bound == pytest.approx(20.0)
    assert result.overrun_no_nudge_gate is True
    assert result.key_action_score == pytest.approx(0.0)
    assert result.progress_score == pytest.approx(0.0)


def test_internal_key_action_only_no_nudge_valid_when_stays_behind(straight_road_scene):
    scene = _with_key_obstacles(straight_road_scene, [("blocked_car", 0, 24.0)])
    trajectory = np.asarray([[1.0 * t, 0.0] for t in range(1, 9)], dtype=np.float64)

    result = InternalKeyActionScorer().score(trajectory, scene)

    assert result.sample_valid is True
    assert result.num_key_actions == 0
    assert result.overrun_no_nudge_gate is False
    assert result.key_action_score == pytest.approx(1.0)
    assert 0.0 < result.progress_score < 1.0


def test_internal_key_action_missing_annotations_invalid(straight_road_scene):
    result = InternalKeyActionScorer().score(
        np.asarray([[1.0 * t, 0.0] for t in range(1, 9)], dtype=np.float64),
        straight_road_scene,
    )

    assert result.sample_valid is False
    assert result.invalid_reason == "missing_key_action_obstacles"
    assert result.internal_score == pytest.approx(0.0)

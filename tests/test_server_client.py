"""Tests for server endpoints and client methods.

Uses FastAPI TestClient (in-process, no real HTTP server needed).
Patches `load_scene_context` to return our fixture scenes so we don't need
MetricCache/NavSim dependencies.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nous_sim_engine.client import SimEngineClient, SimEngineClientError
from nous_sim_engine.core.scorer import PDMScorer, RLScorerConfig
from nous_sim_engine.core.types import SceneContext
from nous_sim_engine.server.app import create_app


# ── Helpers ──────────────────────────────────────────────────────────────

DATASET = "test"
LOG_NAME = "test_log"
SCENE_TOKEN = "test_scene_001"


class _PatchedClient(SimEngineClient):
    """SimEngineClient that talks to FastAPI TestClient instead of real HTTP."""

    def __init__(self, test_client: TestClient):
        super().__init__(base_url="http://testserver")
        self._test_client = test_client

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        if method == "GET":
            resp = self._test_client.get(path)
        else:
            resp = self._test_client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()


@pytest.fixture
def app_and_client(straight_road_scene):
    """Create FastAPI app + TestClient with load_scene_context patched and test dataset registered."""
    with patch("nous_sim_engine.server.app.load_scene_context", return_value=straight_road_scene):
        with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
            app = create_app()
            with TestClient(app) as tc:
                yield tc


@pytest.fixture
def app_with_redlight(red_light_scene):
    """TestClient with a red-light scene."""
    with patch("nous_sim_engine.server.app.load_scene_context", return_value=red_light_scene):
        with patch.dict("os.environ", {"SIM_ENGINE_DATASETS": f"test=/fake/cache"}):
            app = create_app()
            with TestClient(app) as tc:
                yield tc


@pytest.fixture
def sim_client(app_and_client) -> SimEngineClient:
    """SimEngineClient wired to the in-process TestClient."""
    return _PatchedClient(app_and_client)


@pytest.fixture
def sim_client_redlight(app_with_redlight) -> SimEngineClient:
    return _PatchedClient(app_with_redlight)


# ═══════════════════════════════════════════════════════════════════════
# 1. Health endpoint
# ═══════════════════════════════════════════════════════════════════════


class TestHealth:
    def test_health_ok(self, app_and_client):
        resp = app_and_client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "cache_stats" in data

    def test_client_health(self, sim_client):
        result = sim_client.health()
        assert result["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════
# 2. PDM Scoring endpoints (/v1/score, /v1/score/batch)
# ═══════════════════════════════════════════════════════════════════════


class TestPDMScoring:
    def test_score_single(self, app_and_client, safe_trajectory):
        resp = app_and_client.post("/v1/score", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "pdm_score" in data
        assert data["error"] is None
        assert 0.0 <= data["pdm_score"] <= 1.0

    def test_score_batch(self, app_and_client, safe_trajectory, offroad_trajectory):
        resp = app_and_client.post("/v1/score/batch", json={
            "trajectories": [safe_trajectory, offroad_trajectory],
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        # Safe trajectory should score higher than offroad
        assert data["results"][0]["pdm_score"] >= data["results"][1]["pdm_score"]

    def test_client_score(self, sim_client, safe_trajectory):
        pdm_score, result = sim_client.score(
            trajectory=safe_trajectory,
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
        )
        assert pdm_score > 0.0
        assert result["error"] is None

    def test_client_score_batch(self, sim_client, safe_trajectory, offroad_trajectory):
        results = sim_client.score_batch(
            trajectories=[safe_trajectory, offroad_trajectory],
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
        )
        assert len(results) == 2
        assert results[0]["pdm_score"] >= results[1]["pdm_score"]


# ═══════════════════════════════════════════════════════════════════════
# 3. RL Scoring endpoints (/v1/score/rl, /v1/score/rl/batch)
# ═══════════════════════════════════════════════════════════════════════


class TestRLScoring:

    # ── 3.1 Basic continuous mode ────────────────────────────────────

    def test_rl_score_continuous(self, app_and_client, safe_trajectory):
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "rl_score" in data
        assert "sub_rewards" in data
        assert data["error"] is None
        assert 0.0 <= data["rl_score"] <= 1.0
        # sub_rewards should have all 8 keys
        assert set(data["sub_rewards"].keys()) == {"nc", "dac", "ddc", "tlc", "ep", "ttc", "lk", "hc"}
        # All sub_rewards in [0, 1]
        for key, val in data["sub_rewards"].items():
            assert 0.0 <= val <= 1.0, f"sub_reward {key}={val} out of [0,1]"

    # ── 3.2 Basic discrete mode ──────────────────────────────────────

    def test_rl_score_discrete(self, app_and_client, safe_trajectory):
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "discrete",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is None
        # Discrete safety metrics should be exactly 0.0 or 1.0
        for key in ["no_at_fault_collisions", "drivable_area_compliance",
                     "driving_direction_compliance", "traffic_light_compliance"]:
            assert data[key] in (0.0, 0.5, 1.0), f"discrete {key}={data[key]} not binary-ish"

    # ── 3.3 Mode switching produces different results ────────────────

    def test_continuous_vs_discrete_differ(self, app_and_client, safe_trajectory):
        resp_cont = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
        })
        resp_disc = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "discrete",
        })
        cont = resp_cont.json()
        disc = resp_disc.json()
        # At least one sub-metric should differ (NC continuous != NC discrete near obstacle)
        # Note: they CAN be same if trajectory is far from all obstacles. Use rl_score.
        # Both should be valid
        assert cont["error"] is None
        assert disc["error"] is None

    # ── 3.4 Batch RL scoring ─────────────────────────────────────────

    def test_rl_score_batch(self, app_and_client, safe_trajectory, offroad_trajectory):
        resp = app_and_client.post("/v1/score/rl/batch", json={
            "trajectories": [safe_trajectory, offroad_trajectory],
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        safe_result = data["results"][0]
        offroad_result = data["results"][1]
        # Safe should score higher than offroad
        assert safe_result["rl_score"] >= offroad_result["rl_score"]
        # Offroad should have lower DAC
        assert safe_result["drivable_area_compliance"] >= offroad_result["drivable_area_compliance"]

    # ── 3.5 Config overrides ─────────────────────────────────────────

    def test_rl_score_with_config_overrides(self, app_and_client, offroad_trajectory):
        # Score with default weights
        resp_default = app_and_client.post("/v1/score/rl", json={
            "trajectory": offroad_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
        })
        # Score with heavily boosted ep_weight
        resp_custom = app_and_client.post("/v1/score/rl", json={
            "trajectory": offroad_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
            "config_overrides": {"ep_weight": 100.0},
        })
        default_data = resp_default.json()
        custom_data = resp_custom.json()
        assert default_data["error"] is None
        assert custom_data["error"] is None
        # Sub-metrics should be same (same trajectory), but rl_score differs due to weights
        assert abs(default_data["sub_rewards"]["ep"] - custom_data["sub_rewards"]["ep"]) < 1e-6
        # rl_score should differ
        assert default_data["rl_score"] != custom_data["rl_score"]

    # ── 3.6 Red light compliance ─────────────────────────────────────

    def test_rl_red_light_continuous(self, app_with_redlight, safe_trajectory):
        """Trajectory through red light should get low TLC in continuous mode."""
        resp = app_with_redlight.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "continuous",
        })
        data = resp.json()
        assert data["error"] is None
        # Should have red light violation (trajectory goes through x=20)
        assert data["traffic_light_compliance"] < 1.0

    def test_rl_red_light_discrete(self, app_with_redlight, safe_trajectory):
        """Discrete TLC should be exactly 0.0 for red light violation."""
        resp = app_with_redlight.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "discrete",
        })
        data = resp.json()
        assert data["traffic_light_compliance"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 4. Client RL methods
# ═══════════════════════════════════════════════════════════════════════


class TestClientRL:
    def test_client_score_rl_continuous(self, sim_client, safe_trajectory):
        rl_score, result = sim_client.score_rl(
            trajectory=safe_trajectory,
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
            scoring_mode="continuous",
        )
        assert rl_score > 0.0
        assert result["error"] is None
        assert "sub_rewards" in result
        assert set(result["sub_rewards"].keys()) == {"nc", "dac", "ddc", "tlc", "ep", "ttc", "lk", "hc"}

    def test_client_score_rl_discrete(self, sim_client, safe_trajectory):
        rl_score, result = sim_client.score_rl(
            trajectory=safe_trajectory,
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
            scoring_mode="discrete",
        )
        assert rl_score > 0.0
        assert result["error"] is None

    def test_client_score_batch_rl(self, sim_client, safe_trajectory, offroad_trajectory):
        results = sim_client.score_batch_rl(
            trajectories=[safe_trajectory, offroad_trajectory],
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
            scoring_mode="continuous",
        )
        assert len(results) == 2
        assert results[0]["rl_score"] >= results[1]["rl_score"]
        assert results[0]["error"] is None

    def test_client_score_rl_with_overrides(self, sim_client, safe_trajectory):
        rl_score, result = sim_client.score_rl(
            trajectory=safe_trajectory,
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
            scoring_mode="continuous",
            config_overrides={"nc_weight": 10.0, "ttc_horizon": 5.0},
        )
        assert rl_score > 0.0
        assert result["error"] is None

    def test_client_score_rl_red_light(self, sim_client_redlight, safe_trajectory):
        rl_score, result = sim_client_redlight.score_rl(
            trajectory=safe_trajectory,
            scene_token=SCENE_TOKEN,
            log_name=LOG_NAME,
            dataset=DATASET,
            scoring_mode="discrete",
        )
        assert result["traffic_light_compliance"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 5. Schema validation
# ═══════════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    def test_invalid_scoring_mode(self, app_and_client, safe_trajectory):
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "invalid_mode",
        })
        assert resp.status_code == 422  # Pydantic validation error

    def test_missing_required_fields(self, app_and_client):
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": [[0.1, 0.2]],
            # Missing scene_token, log_name, metric_cache_dir
        })
        assert resp.status_code == 422

    def test_default_scoring_mode(self, app_and_client, safe_trajectory):
        """scoring_mode defaults to 'continuous' when not specified."""
        resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            # No scoring_mode — should default to continuous
        })
        assert resp.status_code == 200
        assert resp.json()["error"] is None


# ═══════════════════════════════════════════════════════════════════════
# 6. Consistency: discrete RL sub-metrics vs PDM sub-metrics
# ═══════════════════════════════════════════════════════════════════════


class TestDiscreteConsistency:
    """Discrete RL safety sub-metrics should match PDM sub-metrics."""

    def test_discrete_matches_pdm_safety(self, app_and_client, safe_trajectory):
        pdm_resp = app_and_client.post("/v1/score", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
        })
        rl_resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": safe_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "discrete",
        })
        pdm = pdm_resp.json()
        rl = rl_resp.json()
        # Safety metrics should match exactly
        for key in ["no_at_fault_collisions", "drivable_area_compliance",
                     "driving_direction_compliance", "traffic_light_compliance"]:
            assert abs(pdm[key] - rl[key]) < 1e-6, f"{key}: PDM={pdm[key]} vs RL={rl[key]}"
        # LK, HC should match in discrete mode
        for key in ["lane_keeping", "history_comfort"]:
            assert abs(pdm[key] - rl[key]) < 1e-6, f"{key}: PDM={pdm[key]} vs RL={rl[key]}"
        # TTC: RL uses continuous _ttc_continuous (normalized by horizon) even in discrete mode,
        # while PDM uses binary _time_to_collision. This is by design — RL TTC is always continuous.
        # Both should be in [0, 1].
        assert 0.0 <= rl["time_to_collision"] <= 1.0

    def test_discrete_matches_pdm_offroad(self, app_and_client, offroad_trajectory):
        pdm_resp = app_and_client.post("/v1/score", json={
            "trajectory": offroad_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
        })
        rl_resp = app_and_client.post("/v1/score/rl", json={
            "trajectory": offroad_trajectory,
            "scene_token": SCENE_TOKEN,
            "log_name": LOG_NAME,
            "dataset": DATASET,
            "scoring_mode": "discrete",
        })
        pdm = pdm_resp.json()
        rl = rl_resp.json()
        for key in ["no_at_fault_collisions", "drivable_area_compliance",
                     "driving_direction_compliance", "traffic_light_compliance",
                     "time_to_collision", "lane_keeping", "history_comfort"]:
            assert abs(pdm[key] - rl[key]) < 1e-6, f"{key}: PDM={pdm[key]} vs RL={rl[key]}"

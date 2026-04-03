from __future__ import annotations

import json
import socket
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SimEngineClientError(RuntimeError):
    """Raised when the sim engine returns an invalid HTTP response."""


class SimEngineClient:
    """Lightweight urllib-based HTTP client for the nous-sim-engine service."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ── PDM Scoring (NavSim-compatible) ─────────────────────────────────

    def score(
        self,
        trajectory: List[List[float]],
        scene_token: str,
        log_name: str,
        dataset: str,
    ) -> Tuple[float, dict[str, Any]]:
        """Score a single trajectory and return ``(pdm_score, full_result_dict)``."""
        try:
            response = self._request_json(
                method="POST",
                path="/v1/score",
                payload={
                    "trajectory": trajectory,
                    "scene_token": scene_token,
                    "log_name": log_name,
                    "dataset": dataset,
                },
            )
        except SimEngineClientError as exc:
            error_result = self._error_result(str(exc))
            return 0.0, error_result

        result = self._normalize_result(response, endpoint="/v1/score")
        pdm_score = self._result_score(result)
        if result.get("error"):
            return 0.0, result
        return pdm_score, result

    def score_batch(
        self,
        trajectories: List[List[List[float]]],
        scene_token: str,
        log_name: str,
        dataset: str,
    ) -> List[dict[str, Any]]:
        """Score a batch of trajectories and return the raw result dicts."""
        try:
            response = self._request_json(
                method="POST",
                path="/v1/score/batch",
                payload={
                    "trajectories": trajectories,
                    "scene_token": scene_token,
                    "log_name": log_name,
                    "dataset": dataset,
                },
            )
        except SimEngineClientError as exc:
            return [self._error_result(str(exc)) for _ in trajectories]

        if not isinstance(response, dict):
            return [
                self._error_result("Invalid JSON payload from /v1/score/batch: expected object")
                for _ in trajectories
            ]

        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(trajectories):
            return [
                self._error_result("Invalid JSON payload from /v1/score/batch: malformed results")
                for _ in trajectories
            ]

        return [self._normalize_result(item, endpoint="/v1/score/batch") for item in results]

    # ── RL Scoring (continuous / discrete) ──────────────────────────────

    def score_rl(
        self,
        trajectory: List[List[float]],
        scene_token: str,
        log_name: str,
        dataset: str,
        scoring_mode: Literal["continuous", "discrete"] = "continuous",
        config_overrides: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, dict[str, Any]]:
        """Score a single trajectory with RL reward and return ``(rl_score, full_result_dict)``."""
        payload: dict[str, Any] = {
            "trajectory": trajectory,
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
            "scoring_mode": scoring_mode,
        }
        if config_overrides:
            payload["config_overrides"] = config_overrides

        try:
            response = self._request_json(method="POST", path="/v1/score/rl", payload=payload)
        except SimEngineClientError as exc:
            error_result = self._rl_error_result(str(exc))
            return 0.0, error_result

        result = self._normalize_rl_result(response, endpoint="/v1/score/rl")
        rl_score = self._rl_result_score(result)
        if result.get("error"):
            return 0.0, result
        return rl_score, result

    def score_batch_rl(
        self,
        trajectories: List[List[List[float]]],
        scene_token: str,
        log_name: str,
        dataset: str,
        scoring_mode: Literal["continuous", "discrete"] = "continuous",
        config_overrides: Optional[Dict[str, float]] = None,
    ) -> List[dict[str, Any]]:
        """Score a batch of trajectories with RL reward."""
        payload: dict[str, Any] = {
            "trajectories": trajectories,
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
            "scoring_mode": scoring_mode,
        }
        if config_overrides:
            payload["config_overrides"] = config_overrides

        try:
            response = self._request_json(method="POST", path="/v1/score/rl/batch", payload=payload)
        except SimEngineClientError as exc:
            return [self._rl_error_result(str(exc)) for _ in trajectories]

        if not isinstance(response, dict):
            return [
                self._rl_error_result("Invalid JSON payload from /v1/score/rl/batch: expected object")
                for _ in trajectories
            ]

        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(trajectories):
            return [
                self._rl_error_result("Invalid JSON payload from /v1/score/rl/batch: malformed results")
                for _ in trajectories
            ]

        return [self._normalize_rl_result(item, endpoint="/v1/score/rl/batch") for item in results]

    # ── Dataset Management ──────────────────────────────────────────────

    def list_datasets(self) -> Dict[str, str]:
        """List all registered datasets on the server."""
        try:
            response = self._request_json(method="GET", path="/v1/datasets")
        except SimEngineClientError as exc:
            return {"error": str(exc)}
        if isinstance(response, dict):
            return response.get("datasets", response)
        return {"error": "Invalid response"}

    def register_dataset(self, name: str, path: str) -> dict:
        """Register a new dataset on the server."""
        return self._request_json(
            method="POST",
            path="/v1/datasets",
            payload={"name": name, "path": path},
        )

    def unregister_dataset(self, name: str) -> dict:
        """Unregister a dataset from the server."""
        return self._request_json(method="DELETE", path=f"/v1/datasets/{name}")

    # ── Health ──────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Return health metadata or an error payload when the service is unreachable."""
        try:
            response = self._request_json(method="GET", path="/v1/health")
        except SimEngineClientError as exc:
            return {"status": "error", "error": str(exc)}

        if isinstance(response, dict):
            return response
        return {"status": "error", "error": "Invalid JSON payload from /v1/health: expected object"}

    # ── Internal ────────────────────────────────────────────────────────

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            url=f"{self._base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw_body = response.read()
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            detail = f": {body}" if body else ""
            raise SimEngineClientError(f"HTTP {exc.code} from {path}{detail}") from exc
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                raise SimEngineClientError(f"Request to {path} timed out after {self._timeout}s") from exc
            raise SimEngineClientError(f"Request to {path} failed: {reason}") from exc
        except socket.timeout as exc:
            raise SimEngineClientError(f"Request to {path} timed out after {self._timeout}s") from exc
        except OSError as exc:
            raise SimEngineClientError(f"Request to {path} failed: {exc}") from exc

        try:
            return json.loads(raw_body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SimEngineClientError(f"Response from {path} is not valid UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise SimEngineClientError(f"Response from {path} is not valid JSON: {exc.msg}") from exc

    def _normalize_result(self, response: Any, endpoint: str) -> dict[str, Any]:
        if not isinstance(response, dict):
            return self._error_result(f"Invalid JSON payload from {endpoint}: expected object")

        result = self._error_result("")
        result.update(response)
        return result

    def _normalize_rl_result(self, response: Any, endpoint: str) -> dict[str, Any]:
        if not isinstance(response, dict):
            return self._rl_error_result(f"Invalid JSON payload from {endpoint}: expected object")

        result = self._rl_error_result("")
        result.update(response)
        return result

    def _result_score(self, result: dict[str, Any]) -> float:
        try:
            return float(result.get("pdm_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _rl_result_score(self, result: dict[str, Any]) -> float:
        try:
            return float(result.get("rl_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _error_result(self, message: str) -> dict[str, Any]:
        return {
            "pdm_score": 0.0,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
            "ego_progress": 0.0,
            "time_to_collision": 1.0,
            "lane_keeping": 1.0,
            "history_comfort": 1.0,
            "error": message or None,
        }

    def _rl_error_result(self, message: str) -> dict[str, Any]:
        return {
            "rl_score": 0.0,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
            "ego_progress": 0.0,
            "time_to_collision": 1.0,
            "lane_keeping": 1.0,
            "history_comfort": 1.0,
            "sub_rewards": {},
            "error": message or None,
        }


__all__ = ["SimEngineClient", "SimEngineClientError"]

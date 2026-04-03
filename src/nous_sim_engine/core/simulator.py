from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .bicycle_model import BatchKinematicBicycleModel
from .enums import StateIndex
from .lqr_tracker import BatchLQRTracker
from .types import VehicleParams

if TYPE_CHECKING:
    from .observation import PDMObservation


class PDMSimulator:
    """Batch closed-loop proposal simulator matching NavSim's PDM rollout logic."""

    def __init__(
        self,
        discretization_time: float = 0.1,
        vehicle: VehicleParams | None = None,
    ) -> None:
        if discretization_time <= 0.0:
            raise ValueError(f"discretization_time must be positive, got {discretization_time}")

        self._discretization_time = float(discretization_time)
        self._vehicle = vehicle or VehicleParams()
        self._motion_model = BatchKinematicBicycleModel(vehicle=self._vehicle)
        self._tracker = BatchLQRTracker(
            discretization_time=self._discretization_time,
            vehicle=self._vehicle,
        )

    @staticmethod
    def _extract_pose(state: np.ndarray) -> np.ndarray:
        return np.asarray(
            state[[StateIndex.X, StateIndex.Y, StateIndex.HEADING]],
            dtype=np.float64,
        )

    def _resolve_dt(self, observation: "PDMObservation" | None) -> float:
        if observation is None:
            return self._discretization_time

        interval_time = float(observation.interval_time)
        if interval_time <= 0.0:
            raise ValueError(f"observation interval_time must be positive, got {interval_time}")
        return interval_time

    def _prepare_reference_trajectory(
        self,
        ego_state: np.ndarray,
        proposals: np.ndarray,
    ) -> np.ndarray:
        ego_pose = self._extract_pose(ego_state)
        first_pose = proposals[:, 0, :]

        if np.allclose(first_pose, ego_pose[None, :], atol=1e-6, rtol=1e-6):
            return proposals

        return np.concatenate(
            [np.repeat(ego_pose[None, None, :], proposals.shape[0], axis=0), proposals],
            axis=1,
        )

    def simulate_proposals(
        self,
        ego_state: np.ndarray,
        proposals: np.ndarray,
        observation: "PDMObservation" | None = None,
    ) -> np.ndarray:
        ego_state = np.asarray(ego_state, dtype=np.float64)
        proposals = np.asarray(proposals, dtype=np.float64)

        if ego_state.shape != (StateIndex.size(),):
            raise ValueError(f"ego_state must have shape [{StateIndex.size()}], got {ego_state.shape}")
        if proposals.ndim != 3 or proposals.shape[-1] != 3:
            raise ValueError(f"proposals must have shape [B, T, 3], got {proposals.shape}")
        if proposals.shape[1] == 0:
            raise ValueError("proposals must contain at least one pose")

        dt = self._resolve_dt(observation)
        self._tracker.discretization_time = dt

        reference_trajectory = self._prepare_reference_trajectory(ego_state, proposals)
        batch_size, num_steps, _ = reference_trajectory.shape

        simulated_states = np.zeros((batch_size, num_steps, StateIndex.size()), dtype=np.float64)
        simulated_states[:, 0, :] = ego_state[None, :]
        self._tracker.update(reference_trajectory)

        for time_idx in range(num_steps - 1):
            accelerations, steering_rates = self._tracker.track_trajectory(
                current_state=simulated_states[:, time_idx, :],
                time_idx=time_idx,
            )
            simulated_states[:, time_idx + 1, :] = self._motion_model.propagate_state(
                states=simulated_states[:, time_idx, :],
                accel_cmds=accelerations,
                steering_rate_cmds=steering_rates,
                dt=dt,
            )

        return simulated_states

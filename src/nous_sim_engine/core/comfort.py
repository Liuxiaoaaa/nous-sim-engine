from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .enums import StateIndex

COMFORT_THRESHOLDS = {
    "max_abs_mag_jerk": 8.37,
    "max_abs_lat_accel": 4.89,
    "max_lon_accel": 2.40,
    "min_lon_accel": -4.05,
    "max_abs_yaw_accel": 1.93,
    "max_abs_lon_jerk": 4.13,
    "max_abs_yaw_rate": 0.95,
}

# Default rear-axle to center-of-vehicle offset (Pacifica)
_DEFAULT_REAR_AXLE_TO_CENTER = 1.461


def _safe_savgol(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Apply savgol_filter with safe window/polyorder handling for short sequences."""
    n = len(values)
    if n < 3:
        return values.astype(np.float64, copy=True)
    w = min(window_length, n if n % 2 == 1 else n - 1)
    if w < 3:
        return values.astype(np.float64, copy=True)
    p = min(polyorder, w - 1)
    return savgol_filter(values, window_length=w, polyorder=p, mode="interp")


def _savgol_derivative(values: np.ndarray, dt: float, window_length: int, polyorder: int) -> np.ndarray:
    """Compute 1st derivative using savgol_filter (matches official NavSim approach)."""
    n = len(values)
    if n < 3:
        return np.zeros_like(values, dtype=np.float64)
    w = min(window_length, n if n % 2 == 1 else n - 1)
    if w < 3:
        return np.gradient(values, dt)
    p = min(polyorder, w - 1)
    return savgol_filter(values, window_length=w, polyorder=p, deriv=1, delta=dt, mode="interp")


def _rear_axle_to_center(states: np.ndarray, rear_axle_to_center: float) -> tuple[np.ndarray, np.ndarray]:
    """Shift positions from rear-axle to center-of-vehicle.

    Returns center (x, y) arrays.
    """
    headings = states[:, StateIndex.HEADING]
    center_x = states[:, StateIndex.X] + rear_axle_to_center * np.cos(headings)
    center_y = states[:, StateIndex.Y] + rear_axle_to_center * np.sin(headings)
    return center_x, center_y


def _compute_comfort_signals(
    states: np.ndarray,
    time_points_s: np.ndarray,
    rear_axle_to_center: float = _DEFAULT_REAR_AXLE_TO_CENTER,
) -> dict[str, np.ndarray]:
    """Compute all comfort signals aligned with official NavSim.

    Official approach:
    - lon/lat acceleration: from center-of-vehicle coordinates, savgol smoothed (window=8, poly=2)
    - lon jerk: derivative of center-state lon acceleration (savgol window=15, poly=2)
    - magnitude jerk: from rear-axle acceleration magnitude (savgol window=15, poly=2)
    - yaw rate/accel: from heading (savgol derivative window=15, poly=2)
    """
    dt = float(np.median(np.diff(time_points_s)))

    # Center-of-vehicle positions for lon/lat acceleration
    center_x, center_y = _rear_axle_to_center(states, rear_axle_to_center)
    headings = np.unwrap(states[:, StateIndex.HEADING])

    # Velocity from center positions (savgol derivative, window=8, poly=2)
    vx = _savgol_derivative(center_x, dt, window_length=8, polyorder=2)
    vy = _savgol_derivative(center_y, dt, window_length=8, polyorder=2)

    # Project to lon/lat frame
    cos_h = np.cos(headings)
    sin_h = np.sin(headings)
    lon_vel = cos_h * vx + sin_h * vy
    lat_vel = -sin_h * vx + cos_h * vy

    # Acceleration from center-state velocity (savgol derivative, window=8, poly=2)
    lon_accel = _savgol_derivative(lon_vel, dt, window_length=8, polyorder=2)
    lat_accel = _savgol_derivative(lat_vel, dt, window_length=8, polyorder=2)

    # Lon jerk from center-state (savgol derivative, window=15, poly=2)
    lon_jerk = _savgol_derivative(lon_accel, dt, window_length=15, polyorder=2)

    # Magnitude jerk from rear-axle acceleration (official uses rear-axle for magnitude)
    rear_ax = _safe_savgol(states[:, StateIndex.ACCELERATION_X], window_length=8, polyorder=2)
    rear_ay = _safe_savgol(states[:, StateIndex.ACCELERATION_Y], window_length=8, polyorder=2)
    accel_magnitude = np.hypot(rear_ax, rear_ay)
    jerk_magnitude = _savgol_derivative(accel_magnitude, dt, window_length=15, polyorder=2)

    # Yaw rate and acceleration (savgol derivative, window=15, poly=2)
    yaw_rate = _savgol_derivative(headings, dt, window_length=15, polyorder=2)
    yaw_accel = _savgol_derivative(yaw_rate, dt, window_length=15, polyorder=2)

    return {
        "lon_accel": lon_accel,
        "lat_accel": lat_accel,
        "lon_jerk": lon_jerk,
        "jerk_magnitude": jerk_magnitude,
        "yaw_rate": yaw_rate,
        "yaw_accel": yaw_accel,
    }


def ego_is_comfortable(
    states: np.ndarray,
    time_points_s: np.ndarray,
    rear_axle_to_center: float = _DEFAULT_REAR_AXLE_TO_CENTER,
) -> bool:
    """Evaluate comfort thresholds on a single state trajectory (NavSim-aligned)."""
    states = np.asarray(states, dtype=np.float64)
    time_points_s = np.asarray(time_points_s, dtype=np.float64)

    if states.ndim != 2 or states.shape[1] != StateIndex.size():
        raise ValueError(f"states must have shape [T, {StateIndex.size()}], got {states.shape}")
    if len(time_points_s) < 3:
        return True

    s = _compute_comfort_signals(states, time_points_s, rear_axle_to_center)

    return bool(
        np.all(np.abs(s["jerk_magnitude"]) <= COMFORT_THRESHOLDS["max_abs_mag_jerk"])
        and np.all(np.abs(s["lat_accel"]) <= COMFORT_THRESHOLDS["max_abs_lat_accel"])
        and np.all(s["lon_accel"] <= COMFORT_THRESHOLDS["max_lon_accel"])
        and np.all(s["lon_accel"] >= COMFORT_THRESHOLDS["min_lon_accel"])
        and np.all(np.abs(s["yaw_accel"]) <= COMFORT_THRESHOLDS["max_abs_yaw_accel"])
        and np.all(np.abs(s["lon_jerk"]) <= COMFORT_THRESHOLDS["max_abs_lon_jerk"])
        and np.all(np.abs(s["yaw_rate"]) <= COMFORT_THRESHOLDS["max_abs_yaw_rate"])
    )


def ego_comfort_violation(
    states: np.ndarray,
    time_points_s: np.ndarray,
    rear_axle_to_center: float = _DEFAULT_REAR_AXLE_TO_CENTER,
) -> float:
    """Continuous comfort metric: 1.0 = fully comfortable, 0.0 = severely violated.

    Computes the max violation ratio across all 7 thresholds and all timesteps,
    then maps to [0, 1] via ``1 - clip(max_ratio, 0, 1)``.
    """
    states = np.asarray(states, dtype=np.float64)
    time_points_s = np.asarray(time_points_s, dtype=np.float64)

    if len(time_points_s) < 3:
        return 1.0

    s = _compute_comfort_signals(states, time_points_s, rear_axle_to_center)

    violation_ratios = [
        np.max(np.abs(s["jerk_magnitude"]) / COMFORT_THRESHOLDS["max_abs_mag_jerk"] - 1.0),
        np.max(np.abs(s["lat_accel"]) / COMFORT_THRESHOLDS["max_abs_lat_accel"] - 1.0),
        np.max(s["lon_accel"] / COMFORT_THRESHOLDS["max_lon_accel"] - 1.0),
        np.max(-s["lon_accel"] / (-COMFORT_THRESHOLDS["min_lon_accel"]) - 1.0),
        np.max(np.abs(s["yaw_accel"]) / COMFORT_THRESHOLDS["max_abs_yaw_accel"] - 1.0),
        np.max(np.abs(s["lon_jerk"]) / COMFORT_THRESHOLDS["max_abs_lon_jerk"] - 1.0),
        np.max(np.abs(s["yaw_rate"]) / COMFORT_THRESHOLDS["max_abs_yaw_rate"] - 1.0),
    ]

    max_violation = max(0.0, float(max(violation_ratios)))
    return 1.0 - min(max_violation, 1.0)

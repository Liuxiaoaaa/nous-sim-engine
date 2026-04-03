#!/usr/bin/env python3
"""Validate RL scoring against NavSim-compatible discrete scoring on mini_set."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from nous_sim_engine.adapters.navsim.cache_loader import (
    load_metric_cache,
    metric_cache_to_scene_context,
)
from nous_sim_engine.core.scorer import PDMScorer, PDMScorerConfig, RLScorerConfig

METRIC_CACHE_DIR = "/home/liuxiao34/data_augmentation_agents/mini_set_closed_loop/metric_cache"
LOG_NAME = "2021.06.14.14.25.15_veh-26_04936_05073"


def _extract_trajectory(metric_cache, scene_context):
    """Extract ego-relative trajectory from MetricCache."""
    traj = metric_cache.trajectory
    states = list(traj.get_sampled_trajectory())
    global_xy = np.array(
        [[s.rear_axle.x, s.rear_axle.y] for s in states], dtype=np.float64
    )
    ego_x, ego_y = scene_context.ego_state[0], scene_context.ego_state[1]
    ego_heading = scene_context.ego_state[2]
    cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
    dx = global_xy[:, 0] - ego_x
    dy = global_xy[:, 1] - ego_y
    local_xy = np.column_stack([dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h])
    return local_xy[1:51]


def main():
    cache_dir = Path(METRIC_CACHE_DIR)
    log_dir = cache_dir / LOG_NAME / "unknown"
    scene_tokens = sorted([d.name for d in log_dir.iterdir() if d.is_dir()])
    print(f"Found {len(scene_tokens)} scenes")

    scorer = PDMScorer(config=PDMScorerConfig(human_penalty_filter=False))
    rl_config_discrete = RLScorerConfig(safety_mode="discrete")
    rl_config_continuous = RLScorerConfig(safety_mode="continuous")

    pdm_times, rl_disc_times, rl_cont_times = [], [], []

    print(f"\n{'Token':<20} {'PDM':>6} | {'RL-disc':>7} {'RL-cont':>7} | "
          f"{'NC-d':>5} {'NC-c':>5} | {'DAC-d':>5} {'DAC-c':>5} | "
          f"{'DDC-d':>5} {'DDC-c':>5} | {'TLC-d':>5} {'TLC-c':>5} | "
          f"{'EP-d':>5} {'EP-c':>5} | {'TTC-d':>5} {'TTC-c':>5} | "
          f"{'LK-d':>5} {'LK-c':>5} | {'HC-d':>5} {'HC-c':>5}")
    print("-" * 180)

    sub_diffs = {k: [] for k in ["nc", "dac", "ddc", "tlc", "ep", "ttc", "lk", "hc"]}

    for token in scene_tokens:
        mc = load_metric_cache(str(cache_dir), LOG_NAME, token)
        scene = metric_cache_to_scene_context(mc, token)
        waypoints = _extract_trajectory(mc, scene)

        # NavSim-compatible PDM scoring
        t0 = time.time()
        pdm_result = scorer.score(waypoints, scene)
        pdm_times.append(time.time() - t0)

        # RL discrete mode
        t0 = time.time()
        rl_disc = scorer.score_for_rl(waypoints, scene, rl_config_discrete)
        rl_disc_times.append(time.time() - t0)

        # RL continuous mode
        t0 = time.time()
        rl_cont = scorer.score_for_rl(waypoints, scene, rl_config_continuous)
        rl_cont_times.append(time.time() - t0)

        # Compare discrete RL sub-metrics vs PDM sub-metrics
        pdm_d = pdm_result.to_dict()
        disc_d = rl_disc.to_dict()
        cont_d = rl_cont.to_dict()

        # Track sub-metric differences (discrete RL vs PDM)
        for key, short in [
            ("no_at_fault_collisions", "nc"),
            ("drivable_area_compliance", "dac"),
            ("driving_direction_compliance", "ddc"),
            ("traffic_light_compliance", "tlc"),
            ("time_to_collision", "ttc"),
            ("lane_keeping", "lk"),
            ("history_comfort", "hc"),
        ]:
            sub_diffs[short].append(disc_d[key] - pdm_d[key])
        # EP uses different normalization, so just track the value
        sub_diffs["ep"].append(disc_d["ego_progress"] - pdm_d["ego_progress"])

        print(
            f"{token[:18]:<20} {pdm_d['pdm_score']:6.3f} | "
            f"{disc_d['rl_score']:7.3f} {cont_d['rl_score']:7.3f} | "
            f"{disc_d['no_at_fault_collisions']:5.2f} {cont_d['no_at_fault_collisions']:5.2f} | "
            f"{disc_d['drivable_area_compliance']:5.2f} {cont_d['drivable_area_compliance']:5.2f} | "
            f"{disc_d['driving_direction_compliance']:5.2f} {cont_d['driving_direction_compliance']:5.2f} | "
            f"{disc_d['traffic_light_compliance']:5.2f} {cont_d['traffic_light_compliance']:5.2f} | "
            f"{disc_d['ego_progress']:5.2f} {cont_d['ego_progress']:5.2f} | "
            f"{disc_d['time_to_collision']:5.2f} {cont_d['time_to_collision']:5.2f} | "
            f"{disc_d['lane_keeping']:5.2f} {cont_d['lane_keeping']:5.2f} | "
            f"{disc_d['history_comfort']:5.2f} {cont_d['history_comfort']:5.2f}"
        )

    print(f"\n{'=' * 80}")
    print("Timing (ms):")
    print(f"  PDM:           mean={np.mean(pdm_times)*1000:.1f}  total={sum(pdm_times)*1000:.0f}")
    print(f"  RL (discrete): mean={np.mean(rl_disc_times)*1000:.1f}  total={sum(rl_disc_times)*1000:.0f}")
    print(f"  RL (continuous): mean={np.mean(rl_cont_times)*1000:.1f}  total={sum(rl_cont_times)*1000:.0f}")
    overhead = (np.mean(rl_cont_times) / np.mean(pdm_times) - 1) * 100 if np.mean(pdm_times) > 0 else 0
    print(f"  Continuous overhead: {overhead:+.1f}%")

    print(f"\nDiscrete RL vs PDM sub-metric consistency:")
    for key in ["nc", "dac", "ddc", "tlc", "ttc", "lk", "hc"]:
        diffs = sub_diffs[key]
        max_diff = max(abs(d) for d in diffs)
        all_match = all(abs(d) < 1e-6 for d in diffs)
        status = "MATCH" if all_match else f"DIFF (max={max_diff:.4f})"
        print(f"  {key:>4}: {status}")
    ep_diffs = sub_diffs["ep"]
    print(f"  ep  : max_diff={max(abs(d) for d in ep_diffs):.4f} (expected: different normalization)")


if __name__ == "__main__":
    main()

"""Shared actor diagnostics from an already computed historical reward result."""

from trackmaniarl.trackmania.reward_types import RewardResult


def reward_result_info(result: RewardResult) -> dict[str, float | bool]:
    """Expose the same measured terms for native and Openplanet environments."""
    return {
        "reward_time": result.time_reward,
        "reward_pbrs": result.pbrs_reward,
        "reward_progress": result.progress_reward,
        "reward_projected_velocity": result.projected_velocity_reward,
        "reward_projected_speed": result.projected_speed_reward,
        "reward_steering_delta": result.steering_delta_reward,
        "reward_time_attack_terminal": result.time_attack_terminal_reward,
        "reward_pace": result.pace_reward,
        "reward_collision": result.collision_reward,
        "collision": result.collided,
        "collision_detected": result.collision_detected,
        "reward_terminal": result.terminal_reward,
        "potential_progress": result.potential_progress,
        "reference_time_s": result.reference_time_s,
        "time_debt_s": result.time_debt_s,
        "nearest_distance_m": result.nearest_distance_m,
        "accepted_progress_delta_m": result.accepted_progress_delta_m,
        "window_progress_m": result.window_progress_m,
        "steps_since_progress": result.steps_since_progress,
        "projected_velocity_mps": result.projected_velocity_mps,
        "projected_velocity_ratio": result.projected_velocity_ratio,
    }

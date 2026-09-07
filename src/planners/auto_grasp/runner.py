"""Environment loop shared by stateful automatic grasp planners."""

from __future__ import annotations

import time


def run_environment_planner(
    config,
    planner_type,
    success_label: str,
    *,
    realtime: bool = False,
) -> bool:
    import gymnasium as gym
    import envs  # noqa: F401  # Register Gymnasium environments.

    env = gym.make(config.env_id, render_mode=config.render_mode)
    planner = planner_type(config, env=env)
    try:
        while not planner.is_done and planner.step_index < planner.max_steps:
            viewer = getattr(env.unwrapped, "viewer", None)
            if viewer and not viewer.is_running():
                break

            started_at = time.perf_counter()
            action = planner.compute_action()
            transition = env.step(action)
            planner.update(*transition)
            if realtime:
                elapsed = time.perf_counter() - started_at
                time.sleep(max(0.0, 1.0 / planner.control_fps - elapsed))

        if planner.is_success:
            print(f"[{planner.step_index:04d}] SUCCESS: {success_label}.")
            return True
        if planner.step_index >= planner.max_steps:
            print(f"FAILED: max_steps={planner.max_steps} reached.")
        else:
            reason = planner.failure_reason or "planner stopped"
            print(f"[{planner.step_index:04d}] FAILED: {reason}.")
        return False
    except KeyboardInterrupt:
        print("Interrupted.")
        return False
    finally:
        planner.close()
        env.close()

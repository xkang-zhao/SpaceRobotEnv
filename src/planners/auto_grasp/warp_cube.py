"""Run the existing Cube teacher independently in each CUDA vector world."""

from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .cube import CubeAutoGraspConfig, CubeAutoGraspPlanner, action_dict_to_vector


def run_vector_cube(num_envs: int = 4, *, seed: int = 7, max_steps: int = 400,
                    device: str = "cuda:0", ik_backend: str = "cpu") -> dict:
    from envs.space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    env = SpaceUR10eWarpVectorEnv(num_envs, device=device, ik_backend=ik_backend)
    try:
        obs, _ = env.reset(seed=seed)
        planners, views = [], []
        for w in range(num_envs):
            planner = CubeAutoGraspPlanner(CubeAutoGraspConfig(render_mode="none"))
            # Minimal view required by the existing Cube planner: commanded
            # tool orientation and current pinch position. No full MjData copy.
            view = SimpleNamespace(controller=env.controllers[w], pinch_site_id=0,
                                   data=SimpleNamespace(site_xpos=np.zeros((1, 3))))
            planner.bind_env_base(view)
            planner.set_reference_pose(obs["ee_pose"][w])
            planners.append(planner)
            views.append(view)
        results = [None] * num_envs
        accepted_count = fallback_count = 0
        shadow_max = 0.0
        start = perf_counter()
        for step in range(1, max_steps + 1):
            teacher = env.get_planner_state()
            actions = np.zeros((num_envs, 7), dtype=np.float32)
            for w, planner in enumerate(planners):
                views[w].data.site_xpos[0] = teacher["pinch_pos"][w]
                planner_obs = {key: value[w] for key, value in obs.items()}
                planner_obs["target_pose"] = teacher["target_pose"][w]
                actions[w] = action_dict_to_vector(planner.get_action(planner_obs, episode_success=False))
            obs, reward, terminated, truncated, info = env.step(actions)
            accepted_count += int(np.sum(info.get("ik_gpu_accepted", 0)))
            fallback_count += int(np.sum(info.get("ik_cpu_fallback", 0)))
            if ik_backend == "shadow" and info["ik_gpu_accepted"].any():
                shadow_max = max(shadow_max, float(np.max(info["ik_shadow_joint_max_abs"][info["ik_gpu_accepted"]])))
            done = terminated | truncated
            for w in np.flatnonzero(done):
                if results[w] is None:
                    results[w] = {"world": int(w), "seed": seed + int(w), "steps": step,
                                  "success": bool(terminated[w] and info["is_success"][w]),
                                  "success_counter": int(info["success_counter"][w]),
                                  "left_contact": bool(info["left_contact"][w]),
                                  "right_contact": bool(info["right_contact"][w])}
            if all(result is not None for result in results):
                break
            if done.any():
                # Finished worlds start disposable episodes while other worlds
                # finish their first attempt. Never reset unfinished worlds.
                obs, _ = env.reset(seed=seed, options={"reset_mask": done})
                for w in np.flatnonzero(done):
                    planners[w].reset_episode()
                    planners[w].set_reference_pose(obs["ee_pose"][w])
        for w in range(num_envs):
            if results[w] is None:
                results[w] = {"world": w, "seed": seed + w, "steps": max_steps,
                              "success": False, "failure_reason": "step limit"}
        return {"num_envs": num_envs, "ik_backend": ik_backend,
                "gpu_accepted_world_steps": accepted_count, "cpu_fallback_world_steps": fallback_count,
                "total_world_steps": step * num_envs, "shadow_max_accepted_joint_difference_rad": shadow_max,
                "seconds_including_cpu_ik_and_partial_resets": perf_counter() - start,
                "results": results}
    finally:
        env.close()

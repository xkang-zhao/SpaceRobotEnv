"""Headless Cube grasp benchmark for the single-world physics backends."""

from __future__ import annotations

from time import perf_counter

import numpy as np


def benchmark_cube(backend: str, *, seed: int = 7, max_steps: int = 400,
                   device: str = "cuda:0") -> dict:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    import gymnasium as gym
    import envs  # noqa: F401
    from planners.auto_grasp.tasks import AutoGraspSession

    env = gym.make("SpaceUR10e-Cube-v0", observation_mode="state",
                   physics_backend=backend, warp_device=device)
    base = env.unwrapped
    durations = {"ik": [], "physics_and_sync": [], "env_step": []}

    def timed(fn, key):
        def call(*args, **kwargs):
            start = perf_counter()
            result = fn(*args, **kwargs)
            durations[key].append(perf_counter() - start)
            return result
        return call

    base.kinematics.ik = timed(base.kinematics.ik, "ik")
    base._physics.step = timed(base._physics.step, "physics_and_sync")
    session = AutoGraspSession("cube", env)
    try:
        start = perf_counter()
        obs = session.reset(seed)
        reset_seconds = perf_counter() - start
        # Warm graph capture and step kernels without including them in timings.
        env.step(np.zeros(7, dtype=np.float32))
        warmup_seconds = perf_counter() - start - reset_seconds
        obs = session.reset(seed)
        # Reset invalidates graphs. Capture without advancing the episode.
        if backend == "warp":
            base._physics.prepare_steps(100)
        for values in durations.values():
            values.clear()
        info = {}
        max_contacts = max_constraints = 0
        start = perf_counter()
        for _ in range(max_steps):
            # The scripted teacher needs privileged target pose; keep it out
            # of the public state observation used by policies.
            planner_obs = {**obs, "target_pose": base._get_target_pose()}
            action = session.compute_action(planner_obs)
            step_start = perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            durations["env_step"].append(perf_counter() - step_start)
            max_contacts = max(max_contacts, base.data.ncon)
            max_constraints = max(max_constraints, base.data.nefc)
            if not np.isfinite(base.data.qpos).all() or not np.isfinite(base.data.qvel).all():
                raise RuntimeError(f"Cube {backend}, seed={seed}: non-finite simulation state")
            session.update(obs, reward, terminated, truncated, info)
            if session.is_done:
                break
        elapsed = perf_counter() - start
        return {
            "backend": backend, "seed": seed,
            "steps": len(durations["env_step"]),
            "success": session.is_success,
            "failure_reason": session.failure_reason or (None if session.is_success else "step limit"),
            "success_counter": int(info.get("success_counter", 0)),
            "left_contact": bool(info.get("left_contact", False)),
            "right_contact": bool(info.get("right_contact", False)),
            "max_contacts_at_control_boundary": int(max_contacts),
            "max_constraints_at_control_boundary": int(max_constraints),
            "reset_seconds": reset_seconds, "warmup_seconds": warmup_seconds,
            "episode_seconds": elapsed, "simulation_seconds": float(base.data.time),
            "mean_ms": {key: 1000 * float(np.mean(values)) for key, values in durations.items()},
        }
    finally:
        env.close()

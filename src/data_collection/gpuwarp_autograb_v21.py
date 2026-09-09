"""LeRobot v2.1 collection on the CUDA Cube VectorEnv.

The physics and optional IK candidate solve are batched on CUDA.  RGB frames
are rendered one world at a time from the read-back MuJoCo state because the
current VectorEnv deliberately exposes state observations only.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from data_collection.autograb_v21 import (
    CollectionConfig,
    _validate_observation,
    validate_collection_config,
)
from data_collection.lerobot_v21 import (
    LeRobotV21EpisodeBuffer,
    commit_staging_dir,
    finalize_dataset,
    prepare_staging_dir,
    validate_output_target,
)
from mujoco_robot.robot_sensor import RobotSensor
from planners.auto_grasp.cube import (
    CubeAutoGraspConfig,
    CubeAutoGraspPlanner,
    action_dict_to_vector,
)


@dataclass
class _Attempt:
    episode_index: int
    attempt_index: int
    seed: int
    planner: CubeAutoGraspPlanner
    view: SimpleNamespace
    buffer: LeRobotV21EpisodeBuffer
    obs: dict


class _WorldRenderer:
    """Render RGB observations from one world state without a second physics step."""

    def __init__(self, model):
        self.data = mujoco.MjData(model)
        self.sensor = RobotSensor(model, self.data)

    def render(self, qpos: np.ndarray, obs: dict, camera_map: dict[str, str]) -> dict:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.sensor.model, self.data)
        images = self.sensor.update_camera_view()
        rendered = {key: np.asarray(value).copy() for key, value in obs.items()}
        for obs_key in camera_map.values():
            try:
                rendered[obs_key] = images[obs_key]
            except KeyError as exc:
                available = ", ".join(sorted(images))
                raise KeyError(
                    f"GPU Warp renderer has no camera {obs_key!r}; available: {available}"
                ) from exc
        return rendered

    def close(self) -> None:
        self.sensor.close()


def validate_gpuwarp_config(config: CollectionConfig, *, ik_backend: str) -> None:
    validate_collection_config(config)
    if config.task_name != "cube":
        raise ValueError(
            "GPU Warp collection currently supports only task_name='cube'; "
            "the VectorEnv has no Satellite or Debris scenes."
        )
    if config.render or config.enable_rerun:
        raise ValueError("GPU Warp collection does not support --render or --enable-rerun.")
    if ik_backend not in {"cpu", "gpu", "shadow"}:
        raise ValueError("ik_backend must be cpu, gpu, or shadow")


def _attempt_seed(config: CollectionConfig, episode_index: int, attempt_index: int) -> int:
    return config.seed + episode_index * config.max_attempts_per_episode + attempt_index


def _attempt_dir(staging_dir: Path, episode_index: int, attempt_index: int) -> Path:
    return (
        staging_dir
        / ".attempts"
        / f"gpuwarp_{os.getpid()}"
        / f"episode_{episode_index:06d}_attempt_{attempt_index:03d}"
    )


def _new_attempt(
    *,
    env,
    renderer: _WorldRenderer,
    config: CollectionConfig,
    staging_dir: Path,
    world: int,
    episode_index: int,
    attempt_index: int,
    vector_obs: dict,
) -> _Attempt:
    seed = _attempt_seed(config, episode_index, attempt_index)
    obs = renderer.render(env._qpos[world], {key: value[world] for key, value in vector_obs.items()}, config.camera_map)
    _validate_observation(obs, camera_map=config.camera_map, image_size=config.image_size)
    planner = CubeAutoGraspPlanner(CubeAutoGraspConfig(render_mode="none", debug=config.debug_planner))
    view = SimpleNamespace(
        controller=env.controllers[world],
        pinch_site_id=0,
        data=SimpleNamespace(site_xpos=np.zeros((1, 3))),
    )
    planner.bind_env_base(view)
    planner.set_reference_pose(obs["ee_pose"])
    return _Attempt(
        episode_index=episode_index,
        attempt_index=attempt_index,
        seed=seed,
        planner=planner,
        view=view,
        buffer=LeRobotV21EpisodeBuffer(
            _attempt_dir(staging_dir, episode_index, attempt_index),
            camera_map=config.camera_map,
            image_size=config.image_size,
            fps=config.fps,
        ),
        obs=obs,
    )


def _make_result(attempt: _Attempt, *, success: bool, failure_reason: str | None) -> dict:
    return {
        "episode_index": attempt.episode_index,
        "seed": attempt.seed,
        "attempts": attempt.attempt_index + 1,
        "episode": attempt.buffer.to_episode(),
        "is_success": success,
        "failure_reason": failure_reason,
        "worker_pid": os.getpid(),
    }


def collect_gpuwarp_episode_results(
    config: CollectionConfig,
    staging_dir: Path,
    *,
    ik_backend: str,
    device: str,
) -> tuple[list[dict], dict]:
    """Collect target episodes while resetting only worlds that finish an attempt."""
    from envs.space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv

    world_count = min(config.num_envs, config.episodes)
    env = SpaceUR10eWarpVectorEnv(world_count, device=device, ik_backend=ik_backend)
    renderer = _WorldRenderer(env.model)
    attempts: list[_Attempt | None] = [None] * world_count
    results: list[dict] = []
    next_episode = 0
    gpu_accepted = cpu_fallback = 0
    try:
        first_mask = np.ones(world_count, dtype=bool)
        first_seeds = [None] * world_count
        assignments: list[tuple[int, int, int]] = []
        for world in range(world_count):
            assignments.append((world, next_episode, 0))
            first_seeds[world] = _attempt_seed(config, next_episode, 0)
            next_episode += 1
        vector_obs, _ = env.reset(seed=first_seeds, options={"reset_mask": first_mask})
        for world, episode_index, attempt_index in assignments:
            attempts[world] = _new_attempt(
                env=env, renderer=renderer, config=config, staging_dir=staging_dir,
                world=world, episode_index=episode_index, attempt_index=attempt_index,
                vector_obs=vector_obs,
            )

        while len(results) < config.episodes:
            teacher = env.get_planner_state()
            actions = np.zeros((world_count, 7), dtype=np.float32)
            for world, attempt in enumerate(attempts):
                if attempt is None:
                    continue
                attempt.view.data.site_xpos[0] = teacher["pinch_pos"][world]
                planner_obs = {**attempt.obs, "target_pose": teacher["target_pose"][world]}
                action = np.asarray(
                    action_dict_to_vector(attempt.planner.get_action(planner_obs, episode_success=False)),
                    dtype=np.float32,
                )
                attempt.buffer.append_pre_step(attempt.obs, action)
                actions[world] = action

            vector_obs, reward, terminated, truncated, info = env.step(actions)
            gpu_accepted += int(np.sum(info.get("ik_gpu_accepted", 0)))
            cpu_fallback += int(np.sum(info.get("ik_cpu_fallback", 0)))
            reset_mask = np.zeros(world_count, dtype=bool)
            replacements: list[tuple[int, int, int]] = []

            for world, attempt in enumerate(attempts):
                if attempt is None:
                    continue
                confirmed = bool(terminated[world] and info["is_success"][world])
                attempt.buffer.set_last_success(confirmed)
                attempt.obs = renderer.render(
                    env._qpos[world],
                    {key: value[world] for key, value in vector_obs.items()},
                    config.camera_map,
                )
                if confirmed:
                    results.append(_make_result(attempt, success=True, failure_reason=None))
                    attempts[world] = None
                elif truncated[world] or attempt.planner.should_rerecord_episode():
                    reason = (
                        "environment truncated" if truncated[world]
                        else "planner timed out without confirmed success"
                    )
                    if config.keep_failed:
                        results.append(_make_result(attempt, success=False, failure_reason=reason))
                        attempts[world] = None
                    else:
                        attempt.buffer.discard()
                        if attempt.attempt_index + 1 >= config.max_attempts_per_episode:
                            raise RuntimeError(
                                f"GPU Warp collection failed episode {attempt.episode_index} after "
                                f"{config.max_attempts_per_episode} attempts: {reason}"
                            )
                        replacements.append((world, attempt.episode_index, attempt.attempt_index + 1))
                        attempts[world] = None

                if attempts[world] is None and len(results) < config.episodes:
                    reset_mask[world] = True

            if len(results) >= config.episodes:
                break
            for world in np.flatnonzero(reset_mask):
                if not any(item[0] == world for item in replacements) and next_episode < config.episodes:
                    replacements.append((int(world), next_episode, 0))
                    next_episode += 1
            reset_seeds = [None] * world_count
            for world, episode_index, attempt_index in replacements:
                reset_seeds[world] = _attempt_seed(config, episode_index, attempt_index)
            if reset_mask.any():
                vector_obs, _ = env.reset(seed=reset_seeds, options={"reset_mask": reset_mask})
                for world, episode_index, attempt_index in replacements:
                    attempts[world] = _new_attempt(
                        env=env, renderer=renderer, config=config, staging_dir=staging_dir,
                        world=world, episode_index=episode_index, attempt_index=attempt_index,
                        vector_obs=vector_obs,
                    )

        return sorted(results, key=lambda item: item["episode_index"]), {
            "gpu_accepted_world_steps": gpu_accepted,
            "cpu_fallback_world_steps": cpu_fallback,
        }
    except BaseException:
        for attempt in attempts:
            if attempt is not None:
                attempt.buffer.discard()
        raise
    finally:
        renderer.close()
        env.close()


def collect_gpuwarp_dataset(
    config: CollectionConfig,
    *,
    ik_backend: str = "gpu",
    device: str = "cuda:0",
) -> dict:
    """Collect a Cube dataset and transactionally publish it after validation."""
    validate_gpuwarp_config(config, ik_backend=ik_backend)
    target = validate_output_target(config.output_dir)
    staging_dir = prepare_staging_dir(target, overwrite=config.overwrite)
    committed = False
    try:
        results, runtime = collect_gpuwarp_episode_results(
            config, staging_dir, ik_backend=ik_backend, device=device,
        )
        summary = finalize_dataset(
            staging_dir,
            results=results,
            expected_episodes=config.episodes,
            instruction=config.instruction,
            camera_map=config.camera_map,
            fps=config.fps,
            image_size=config.image_size,
        )
        commit_staging_dir(staging_dir, target, overwrite=config.overwrite)
        committed = True
        summary.update(runtime, output_dir=str(target), ik_backend=ik_backend, device=device)
        return summary
    finally:
        if not committed and staging_dir.exists():
            shutil.rmtree(staging_dir)

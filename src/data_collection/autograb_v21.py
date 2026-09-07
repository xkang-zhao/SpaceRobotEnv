"""Automatic grasp collection orchestration for LeRobot v2.1."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from planners.auto_grasp.tasks import AutoGraspSession, get_task_spec

from .lerobot_v21 import (
    LeRobotV21EpisodeBuffer,
    commit_staging_dir,
    finalize_dataset,
    prepare_staging_dir,
    validate_output_target,
)


FPS = 20
IMAGE_SIZE = (640, 480)
DEFAULT_CAMERA_MAP = {
    "cam1": "third_left_camera",
    "cam2": "third_right_camera",
    "cam3": "left_wrist_camera",
}
CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class CollectionConfig:
    task_name: str
    output_dir: Path
    episodes: int
    num_envs: int
    instruction: str
    seed: int
    max_attempts_per_episode: int
    camera_map: dict[str, str]
    keep_failed: bool = False
    overwrite: bool = False
    render: bool = False
    enable_rerun: bool = False
    debug_planner: bool = False
    fps: int = FPS
    image_size: tuple[int, int] = IMAGE_SIZE


def parse_camera_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(
            f"Camera mapping must use NAME=OBS_KEY, got {value!r}."
        )
    name, obs_key = (part.strip() for part in value.split("=", 1))
    if not name or not CAMERA_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Camera NAME must contain only letters, digits, '_' or '-': "
            f"{name!r}."
        )
    if not obs_key:
        raise ValueError("Camera OBS_KEY must not be empty.")
    return name, obs_key


def build_camera_map(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_CAMERA_MAP)
    camera_map: dict[str, str] = {}
    used_obs_keys: set[str] = set()
    for value in values:
        name, obs_key = parse_camera_mapping(value)
        if name in camera_map:
            raise ValueError(f"Duplicate camera name: {name!r}.")
        if obs_key in used_obs_keys:
            raise ValueError(
                f"Duplicate camera observation key: {obs_key!r}."
            )
        camera_map[name] = obs_key
        used_obs_keys.add(obs_key)
    return camera_map


def validate_collection_config(config: CollectionConfig) -> None:
    get_task_spec(config.task_name)
    if config.episodes <= 0:
        raise ValueError("episodes must be positive.")
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if config.max_attempts_per_episode <= 0:
        raise ValueError("max_attempts_per_episode must be positive.")
    if config.fps != FPS:
        raise ValueError(
            f"Collection FPS must remain {FPS}, got {config.fps}."
        )
    if tuple(config.image_size) != IMAGE_SIZE:
        raise ValueError(
            f"Image size must remain {IMAGE_SIZE}, "
            f"got {config.image_size}."
        )
    if not config.instruction.strip():
        raise ValueError("instruction must not be empty.")
    if not config.camera_map:
        raise ValueError("At least one RGB camera must be selected.")
    used_obs_keys: set[str] = set()
    for name, obs_key in config.camera_map.items():
        if not CAMERA_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid camera name: {name!r}.")
        if not obs_key:
            raise ValueError(
                f"Camera observation key for {name!r} must not be empty."
            )
        if obs_key in used_obs_keys:
            raise ValueError(
                f"Duplicate camera observation key: {obs_key!r}."
            )
        used_obs_keys.add(obs_key)
    if config.render and config.num_envs != 1:
        raise ValueError("--render requires --num-envs 1.")
    if config.enable_rerun and config.num_envs != 1:
        raise ValueError("--enable-rerun requires --num-envs 1.")
    validate_output_target(config.output_dir)


def _make_env(config: CollectionConfig):
    import gymnasium as gym
    import envs  # noqa: F401

    task = get_task_spec(config.task_name)
    env = gym.make(
        task.env_id,
        render_mode="human" if config.render else None,
    )
    metadata_fps = int(env.metadata.get("render_fps", 0))
    if metadata_fps != config.fps:
        env.close()
        raise ValueError(
            f"Environment {task.env_id} reports {metadata_fps} FPS, "
            f"expected {config.fps}."
        )
    return env


def _validate_observation(
    obs: dict,
    *,
    camera_map: dict[str, str],
    image_size: tuple[int, int],
) -> None:
    required = {"joint_pos", "ee_pose", *camera_map.values()}
    missing = sorted(required.difference(obs))
    if missing:
        available = ", ".join(sorted(obs))
        raise KeyError(
            f"Missing observation keys: {missing}. "
            f"Available keys: {available}"
        )
    if np.asarray(obs["joint_pos"]).reshape(-1).shape != (6,):
        raise ValueError("joint_pos must contain exactly 6 values.")
    if np.asarray(obs["ee_pose"]).reshape(-1).shape != (7,):
        raise ValueError("ee_pose must contain exactly 7 values.")
    width, height = image_size
    expected = (height, width, 3)
    for name, obs_key in camera_map.items():
        image = np.asarray(obs[obs_key])
        if image.shape != expected or image.dtype != np.uint8:
            raise ValueError(
                f"Camera {name!r} ({obs_key!r}) must be uint8 "
                f"{expected}, got {image.dtype} {image.shape}."
            )


def _attempt_dir(
    staging_dir: Path,
    episode_index: int,
    attempt_index: int,
) -> Path:
    return (
        Path(staging_dir)
        / ".attempts"
        / f"worker_{os.getpid()}"
        / (
            f"episode_{episode_index:06d}_"
            f"attempt_{attempt_index:03d}"
        )
    )


def _log_rerun(obs: dict, action: np.ndarray, camera_map: dict) -> None:
    import rerun as rr

    rr.log("action", rr.BarChart(action))
    for name, obs_key in camera_map.items():
        rr.log(f"observation/images/{name}", rr.Image(obs[obs_key]))


def collect_target_episode(
    env,
    config: CollectionConfig,
    episode_index: int,
    staging_dir: Path,
) -> dict:
    """Collect one target episode, retrying rejected attempts."""
    for attempt_index in range(config.max_attempts_per_episode):
        seed = (
            config.seed
            + episode_index * config.max_attempts_per_episode
            + attempt_index
        )
        attempt_dir = _attempt_dir(
            staging_dir,
            episode_index,
            attempt_index,
        )
        buffer = LeRobotV21EpisodeBuffer(
            attempt_dir,
            camera_map=config.camera_map,
            image_size=config.image_size,
            fps=config.fps,
        )
        session = AutoGraspSession(
            config.task_name,
            env,
            debug=config.debug_planner,
        )
        try:
            obs = session.reset(seed)
            _validate_observation(
                obs,
                camera_map=config.camera_map,
                image_size=config.image_size,
            )
            while (
                not session.is_done
                and session.step_index < session.max_steps
            ):
                viewer = getattr(env.unwrapped, "viewer", None)
                if viewer is not None and not viewer.is_running():
                    raise KeyboardInterrupt

                action = session.compute_action(obs)
                buffer.append_pre_step(obs, action)
                if config.enable_rerun:
                    _log_rerun(obs, action, config.camera_map)

                transition = env.step(action)
                next_obs, reward, terminated, truncated, info = transition
                _validate_observation(
                    next_obs,
                    camera_map=config.camera_map,
                    image_size=config.image_size,
                )
                confirmed = bool(
                    terminated and info.get("is_success", False)
                )
                session.update(
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                )
                buffer.set_last_success(confirmed)
                obs = next_obs

                if confirmed:
                    episode = buffer.to_episode()
                    return {
                        "episode_index": episode_index,
                        "seed": seed,
                        "attempts": attempt_index + 1,
                        "episode": episode,
                        "is_success": True,
                        "failure_reason": None,
                        "worker_pid": os.getpid(),
                    }
                if truncated or session.is_done:
                    break

            failure_reason = (
                session.failure_reason
                or (
                    f"max_steps={session.max_steps} reached"
                    if session.step_index >= session.max_steps
                    else "planner stopped without confirmed success"
                )
            )
            if config.keep_failed:
                episode = buffer.to_episode()
                return {
                    "episode_index": episode_index,
                    "seed": seed,
                    "attempts": attempt_index + 1,
                    "episode": episode,
                    "is_success": False,
                    "failure_reason": failure_reason,
                    "worker_pid": os.getpid(),
                }
            buffer.discard()
            print(
                "[LEROBOT21] discard failed attempt: "
                f"episode={episode_index}, seed={seed}, "
                f"reason={failure_reason}",
                flush=True,
            )
        except BaseException:
            buffer.discard()
            raise

    raise RuntimeError(
        f"Failed to collect episode {episode_index} after "
        f"{config.max_attempts_per_episode} attempts."
    )


_WORKER_ENV = None
_WORKER_CONFIG = None
_WORKER_STAGING_DIR = None


def _close_worker_env() -> None:
    global _WORKER_ENV
    if _WORKER_ENV is not None:
        _WORKER_ENV.close()
        _WORKER_ENV = None


def _initialize_worker(config_values: dict, staging_dir: str) -> None:
    global _WORKER_ENV, _WORKER_CONFIG, _WORKER_STAGING_DIR
    values = dict(config_values)
    values["output_dir"] = Path(values["output_dir"])
    values["image_size"] = tuple(values["image_size"])
    _WORKER_CONFIG = CollectionConfig(**values)
    _WORKER_STAGING_DIR = Path(staging_dir)
    _WORKER_ENV = _make_env(_WORKER_CONFIG)
    atexit.register(_close_worker_env)


def _collect_worker_job(episode_index: int) -> dict:
    if (
        _WORKER_ENV is None
        or _WORKER_CONFIG is None
        or _WORKER_STAGING_DIR is None
    ):
        raise RuntimeError("Worker environment is not initialized.")
    return collect_target_episode(
        _WORKER_ENV,
        _WORKER_CONFIG,
        episode_index,
        _WORKER_STAGING_DIR,
    )


def format_collected_message(result: dict) -> str:
    return (
        "[LEROBOT21] collected "
        f"episode={result['episode_index']}, seed={result['seed']}, "
        f"is_success={bool(result['is_success'])}, "
        f"attempts={result['attempts']}, "
        f"worker_pid={result['worker_pid']}"
    )


def collect_episode_results(
    config: CollectionConfig,
    staging_dir: Path,
) -> list[dict]:
    worker_count = min(config.num_envs, config.episodes)
    if worker_count == 1:
        env = _make_env(config)
        if config.enable_rerun:
            try:
                import rerun as rr
            except ImportError as exc:
                env.close()
                raise RuntimeError(
                    "Rerun is not installed. Install it or remove "
                    "--enable-rerun."
                ) from exc
            rr.init("autograb_lerobot_v21_collection", spawn=True)
        try:
            results = []
            for episode_index in range(config.episodes):
                result = collect_target_episode(
                    env,
                    config,
                    episode_index,
                    staging_dir,
                )
                results.append(result)
                print(format_collected_message(result), flush=True)
            return results
        finally:
            env.close()

    values = asdict(config)
    values["output_dir"] = str(values["output_dir"])
    context = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(values, str(staging_dir)),
    ) as executor:
        futures = {
            executor.submit(_collect_worker_job, episode_index):
            episode_index
            for episode_index in range(config.episodes)
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(format_collected_message(result), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return sorted(results, key=lambda item: item["episode_index"])


def collect_dataset(config: CollectionConfig) -> dict:
    """Collect and transactionally install one v2.1 dataset."""
    validate_collection_config(config)
    target = validate_output_target(config.output_dir)
    staging_dir = prepare_staging_dir(
        target,
        overwrite=config.overwrite,
    )
    committed = False
    try:
        results = collect_episode_results(config, staging_dir)
        summary = finalize_dataset(
            staging_dir,
            results=results,
            expected_episodes=config.episodes,
            instruction=config.instruction,
            camera_map=config.camera_map,
            fps=config.fps,
            image_size=config.image_size,
        )
        commit_staging_dir(
            staging_dir,
            target,
            overwrite=config.overwrite,
        )
        committed = True
        summary["output_dir"] = str(target)
        return summary
    finally:
        if not committed and staging_dir.exists():
            shutil.rmtree(staging_dir)

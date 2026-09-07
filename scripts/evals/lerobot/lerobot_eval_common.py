from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


ACTION_ORDER = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "delta_gripper",
)

ACTION_FEATURES = {name: float for name in ACTION_ORDER}

CAMERA_NAME_MAP = {
    "third_left_camera": "third_left_0_rgb",
    "left_wrist_camera": "left_wrist_0_rgb",
    "third_right_camera": "third_right_0_rgb",
    "left_wrist_camera_depth": "left_wrist_0_depth",
}

RESOLUTION = (640, 480)
OBSERVATION_FEATURES = {
    "base_pos_x": float,
    "base_pos_y": float,
    "base_pos_z": float,
    "base_quat_w": float,
    "base_quat_x": float,
    "base_quat_y": float,
    "base_quat_z": float,
    "shoulder_pan_joint.pos": float,
    "shoulder_lift_joint.pos": float,
    "elbow_joint.pos": float,
    "wrist_1_joint.pos": float,
    "wrist_2_joint.pos": float,
    "wrist_3_joint.pos": float,
    "gripper_joint.pos": float,
    "third_left_0_rgb": (RESOLUTION[1], RESOLUTION[0], 3),
    "left_wrist_0_rgb": (RESOLUTION[1], RESOLUTION[0], 3),
    "third_right_0_rgb": (RESOLUTION[1], RESOLUTION[0], 3),
    "left_wrist_0_depth": (RESOLUTION[1], RESOLUTION[0], 3),
}


@dataclass(frozen=True)
class EvalConfig:
    model_path: str
    policy_type: str = "pi0"
    policy_class: str | None = None
    device: str = "cuda"
    task: str = "Grasp this red cube in space."
    robot_type: str = ""
    action_chunk_size: int = 1
    max_steps_per_episode: int = 500
    render_mode: str | None = None
    env_id: str = "SpaceUR10e-Cube-v0"
    scene_path: str | None = None
    arm_path: str | None = None
    frame_name: str | None = None
    use_depth: bool = True
    seed: int | None = None


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    success: bool
    steps: int
    reason: str
    worker_id: int
    final_distance: float | None


def env_obs_preprocess(obs: dict[str, Any], camera_name_map: dict[str, str]) -> dict[str, Any]:
    obs["base_pos_x"] = obs["base_pose"][0]
    obs["base_pos_y"] = obs["base_pose"][1]
    obs["base_pos_z"] = obs["base_pose"][2]
    obs["base_quat_w"] = obs["base_pose"][3]
    obs["base_quat_x"] = obs["base_pose"][4]
    obs["base_quat_y"] = obs["base_pose"][5]
    obs["base_quat_z"] = obs["base_pose"][6]

    obs["shoulder_pan_joint.pos"] = obs["joint_pos"][0]
    obs["shoulder_lift_joint.pos"] = obs["joint_pos"][1]
    obs["elbow_joint.pos"] = obs["joint_pos"][2]
    obs["wrist_1_joint.pos"] = obs["joint_pos"][3]
    obs["wrist_2_joint.pos"] = obs["joint_pos"][4]
    obs["wrist_3_joint.pos"] = obs["joint_pos"][5]
    obs["gripper_joint.pos"] = 0.0

    for cam_name, map_name in camera_name_map.items():
        img = obs[cam_name]
        if "depth" in map_name and len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
        obs[map_name] = img

    return obs


def import_hw_to_dataset_features():
    try:
        from lerobot.datasets.utils import hw_to_dataset_features
    except ImportError:
        from lerobot.datasets.feature_utils import hw_to_dataset_features

    return hw_to_dataset_features


def build_dataset_features() -> dict[str, Any]:
    hw_to_dataset_features = import_hw_to_dataset_features()
    action_features = hw_to_dataset_features(ACTION_FEATURES, "action")
    obs_features = hw_to_dataset_features(OBSERVATION_FEATURES, "observation")
    return {**action_features, **obs_features}


def import_policy_class(policy_type: str, policy_class: str | None):
    if policy_class:
        module_name, class_name = policy_class.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    policy_imports = {
        "pi0": ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
        "pi05": ("lerobot.policies.pi05.modeling_pi05", "PI05Policy"),
        "smolvla": ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
        "act": ("lerobot.policies.act.modeling_act", "ACTPolicy"),
        "myvla": ("lerobot_policy_myvla", "MyVLAPolicy"),
    }
    if policy_type not in policy_imports:
        raise ValueError(
            f"Unsupported policy type: {policy_type}. "
            f"Use one of {sorted(policy_imports)} or pass --policy-class module:ClassName."
        )

    module_name, class_name = policy_imports[policy_type]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _value_to_1d_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


def action_dict_to_env_actions(action_dict: dict[str, Any]) -> np.ndarray:
    arrays = {name: _value_to_1d_numpy(action_dict[name]) for name in ACTION_ORDER}
    chunk_len = max(len(value) for value in arrays.values())

    columns = []
    for name in ACTION_ORDER:
        value = arrays[name]
        if len(value) == 1 and chunk_len > 1:
            value = np.repeat(value, chunk_len)
        if len(value) != chunk_len:
            raise ValueError(f"Action field {name!r} has length {len(value)}, expected {chunk_len}.")
        columns.append(value)

    return np.stack(columns, axis=1).astype(np.float32, copy=False)


def array_like_to_env_actions(action: Any) -> np.ndarray | None:
    if not hasattr(action, "detach") and not isinstance(action, (list, tuple, np.ndarray)):
        return None
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()

    actions = np.asarray(action, dtype=np.float32)
    if actions.ndim == 1 and actions.shape[0] == len(ACTION_ORDER):
        return actions.reshape(1, len(ACTION_ORDER))
    if actions.ndim == 2 and actions.shape[1] == len(ACTION_ORDER):
        return actions
    if actions.ndim == 3 and actions.shape[0] == 1 and actions.shape[2] == len(ACTION_ORDER):
        return actions[0]
    return None


def policy_output_to_env_actions(
    action: Any,
    dataset_features: dict[str, Any],
    make_robot_action_func: Callable[[Any, dict[str, Any]], dict[str, Any]],
) -> np.ndarray:
    if isinstance(action, dict) and all(name in action for name in ACTION_ORDER):
        return action_dict_to_env_actions(action)

    env_actions = array_like_to_env_actions(action)
    if env_actions is not None:
        return env_actions.astype(np.float32, copy=False)

    action_dict = make_robot_action_func(action, dataset_features)
    return action_dict_to_env_actions(action_dict)


class LeRobotPolicyRunner:
    def __init__(self, config: EvalConfig):
        import torch
        from lerobot.policies.factory import make_pre_post_processors

        self.config = config
        self.device = torch.device(config.device)
        self.dataset_features = build_dataset_features()

        policy_cls = import_policy_class(config.policy_type, config.policy_class)
        self.model = policy_cls.from_pretrained(config.model_path)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.model.config,
            config.model_path,
            preprocessor_overrides={"device_processor": {"device": config.device}},
        )

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()

    def select_action_chunk(
        self,
        observation: dict[str, Any],
        task: str | None = None,
        robot_type: str | None = None,
        action_chunk_size: int | None = None,
    ) -> np.ndarray:
        import torch
        from lerobot.policies.utils import build_inference_frame, make_robot_action

        chunk_size = action_chunk_size or self.config.action_chunk_size
        policy_obs = env_obs_preprocess(observation.copy(), CAMERA_NAME_MAP)
        obs_frame = build_inference_frame(
            observation=policy_obs,
            ds_features=self.dataset_features,
            device=self.device,
            task=self.config.task if task is None else task,
            robot_type=self.config.robot_type if robot_type is None else robot_type,
        )

        with torch.inference_mode():
            policy_input = self.preprocess(obs_frame)
            if chunk_size > 1 and hasattr(self.model, "predict_action_chunk"):
                action = self.model.predict_action_chunk(policy_input)
            else:
                action = self.model.select_action(policy_input)
            action = self.postprocess(action)

        actions = policy_output_to_env_actions(action, self.dataset_features, make_robot_action)
        if chunk_size > 0:
            actions = actions[:chunk_size]
        return actions


def make_env(config: EvalConfig):
    import gymnasium as gym
    import envs  # noqa: F401

    kwargs: dict[str, Any] = {
        "render_mode": config.render_mode,
        "use_depth": config.use_depth,
    }
    for name in ("scene_path", "arm_path", "frame_name"):
        value = getattr(config, name)
        if value is not None:
            kwargs[name] = value

    return gym.make(config.env_id, **kwargs)


def validate_eval_config(config: EvalConfig) -> None:
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path not found: {config.model_path}")
    if config.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be greater than 0.")
    if config.action_chunk_size <= 0:
        raise ValueError("action_chunk_size must be greater than 0.")
    if config.policy_class is not None and ":" not in config.policy_class:
        raise ValueError("policy_class must use module:ClassName format.")


def run_one_episode(
    episode_index: int,
    worker_id: int,
    env,
    runner: LeRobotPolicyRunner,
    config: EvalConfig,
) -> EpisodeResult:
    if config.seed is not None:
        np.random.seed(config.seed + episode_index)

    runner.reset()
    obs, _ = env.reset(seed=None if config.seed is None else config.seed + episode_index)
    steps = 0
    success = False
    reason = "max_steps"
    final_distance = None

    while steps < config.max_steps_per_episode:
        actions = runner.select_action_chunk(obs)
        if len(actions) == 0:
            reason = "empty_action_chunk"
            break

        for env_action in actions:
            obs, _reward, terminated, truncated, info = env.step(env_action)
            steps += 1
            final_distance = info.get("distance")

            if terminated or info.get("is_success", False):
                success = True
                reason = "success"
                break
            if truncated:
                reason = "truncated"
                break
            if steps >= config.max_steps_per_episode:
                break

        if success or reason == "truncated" or steps >= config.max_steps_per_episode:
            break

    return EpisodeResult(
        episode_index=episode_index,
        success=success,
        steps=steps,
        reason=reason,
        worker_id=worker_id,
        final_distance=None if final_distance is None else float(final_distance),
    )


def run_worker(worker_id: int, episode_indices: list[int], config: EvalConfig) -> list[EpisodeResult]:
    runner = LeRobotPolicyRunner(config)
    env = make_env(config)

    try:
        return [
            run_one_episode(
                episode_index=episode_index,
                worker_id=worker_id,
                env=env,
                runner=runner,
                config=config,
            )
            for episode_index in episode_indices
        ]
    finally:
        env.close()


def split_episode_indices(num_episodes: int, num_workers: int) -> list[list[int]]:
    buckets = [[] for _ in range(num_workers)]
    for offset, episode_index in enumerate(range(1, num_episodes + 1)):
        buckets[offset % num_workers].append(episode_index)
    return [bucket for bucket in buckets if bucket]


def run_serial(num_episodes: int, config: EvalConfig) -> list[EpisodeResult]:
    return run_worker(worker_id=0, episode_indices=list(range(1, num_episodes + 1)), config=config)

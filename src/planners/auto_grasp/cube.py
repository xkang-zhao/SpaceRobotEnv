from __future__ import annotations

import argparse
from dataclasses import dataclass, field


FPS = 20
ACTION_KEYS = (
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "delta_gripper",
)


@dataclass
class CubeAutoGraspConfig:
    env_id: str = "SpaceUR10e-Cube-v0"
    front_offset_world: list[float] = field(default_factory=lambda: [-0.12, 0.0, 0.0])
    grasp_offset_world: list[float] = field(default_factory=lambda: [-0.02, 0.0, 0.0])
    target_frame: str = "pinch"
    max_delta_xyz: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01])
    motion_scale: float = 0.3
    front_position_tolerance: float = 0.03
    grasp_position_tolerance: float = 0.015
    close_steps: int = 25
    success_hold_steps: int = 10
    episode_timeout_steps: int = 400
    fps: int = FPS
    render_mode: str = "human"
    debug: bool = False

    def __post_init__(self):
        vector_fields = (
            "front_offset_world",
            "grasp_offset_world",
            "max_delta_xyz",
        )
        for field_name in vector_fields:
            value = getattr(self, field_name)
            if len(value) != 3:
                raise ValueError(
                    f"{field_name} must contain exactly 3 values, got {value!r}"
                )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")


def _vec3(value) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2])]


def _add3(a, b) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _sub3(a, b) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _norm3(v) -> float:
    return sum(float(x) * float(x) for x in v) ** 0.5


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    import math

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _transpose_mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(float(matrix[row][col]) * float(vector[row]) for row in range(3))
        for col in range(3)
    ]


def parse_float_list(value: str) -> list[float]:
    import ast

    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise argparse.ArgumentTypeError(f"Expected a list, got {value!r}")
    return [float(item) for item in parsed]


def action_dict_to_vector(action: dict[str, float]) -> list[float]:
    return [float(action[key]) for key in ACTION_KEYS]


class CubeAutoGraspPlanner:
    def __init__(self, config: CubeAutoGraspConfig):
        self.config = config
        self.stage = "move_to_object_front"
        self.step_index = 0
        self.close_count = 0
        self.success_hold_count = 0
        self._done = False
        self._success = False
        self._base = None
        self._reference_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def bind_env_base(self, base) -> None:
        self._base = base

    def set_reference_pose(self, pose: list[float]) -> None:
        self._reference_pose = [float(value) for value in pose[:6]]

    def reset_episode(self) -> None:
        self.stage = "move_to_object_front"
        self.step_index = 0
        self.close_count = 0
        self.success_hold_count = 0
        self._done = False
        self._success = False

    def _target_position(self, obs: dict, offset_world: list[float]) -> list[float]:
        return _add3(_vec3(obs["target_pose"]), offset_world)

    def _reference_rotation(self) -> list[list[float]]:
        pose = self._reference_pose
        if self._base is not None and hasattr(self._base, "controller"):
            pose = self._base.controller.get_target_pose()
        return _euler_to_rotation_matrix(float(pose[3]), float(pose[4]), float(pose[5]))

    def _target_frame_position(self, obs: dict) -> list[float]:
        if self.config.target_frame == "pinch" and self._base is not None:
            return _vec3(self._base.data.site_xpos[self._base.pinch_site_id])
        return _vec3(obs["ee_pose"])

    def _action_to_position(self, frame_pos: list[float], target_pos: list[float], gripper: float) -> dict[str, float]:
        err_world = _sub3(target_pos, frame_pos)
        err_tool = _transpose_mat_vec(self._reference_rotation(), err_world)
        max_delta = self.config.max_delta_xyz
        dpos = [
            _clip(err_tool[i], float(max_delta[i])) * float(self.config.motion_scale)
            for i in range(3)
        ]
        return {
            "delta_x": dpos[0],
            "delta_y": dpos[1],
            "delta_z": dpos[2],
            "delta_roll": 0.0,
            "delta_pitch": 0.0,
            "delta_yaw": 0.0,
            "delta_gripper": float(gripper),
        }

    def _hold_action(self, gripper: float) -> dict[str, float]:
        return {
            "delta_x": 0.0,
            "delta_y": 0.0,
            "delta_z": 0.0,
            "delta_roll": 0.0,
            "delta_pitch": 0.0,
            "delta_yaw": 0.0,
            "delta_gripper": float(gripper),
        }

    def get_action(self, obs: dict, episode_success: bool = False) -> dict[str, float]:
        if self._done:
            return self._hold_action(0.0)

        frame_pos = self._target_frame_position(obs)
        front_pos = self._target_position(obs, self.config.front_offset_world)
        grasp_pos = self._target_position(obs, self.config.grasp_offset_world)

        if self.stage == "move_to_object_front":
            action = self._action_to_position(frame_pos, front_pos, 0.0)
            if _norm3(_sub3(front_pos, frame_pos)) <= self.config.front_position_tolerance:
                self.stage = "approach_object"
        elif self.stage == "approach_object":
            action = self._action_to_position(frame_pos, grasp_pos, 0.0)
            if _norm3(_sub3(grasp_pos, frame_pos)) <= self.config.grasp_position_tolerance:
                self.stage = "close_gripper"
        elif self.stage == "close_gripper":
            action = self._hold_action(1.0)
            self.close_count += 1
            if episode_success:
                self.success_hold_count += 1
            if self.close_count >= self.config.close_steps:
                self.stage = "hold_grasp"
        else:
            action = self._hold_action(1.0)
            if episode_success:
                self.success_hold_count += 1
            if self.success_hold_count >= self.config.success_hold_steps:
                self._success = True
                self._done = True

        self.step_index += 1
        if self.step_index >= self.config.episode_timeout_steps:
            self._done = True
        return action

    def should_rerecord_episode(self) -> bool:
        return self._done and not self._success

    def should_end_episode(self) -> bool:
        return self._success


def require_obs_keys(obs: dict, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in obs]
    if missing:
        available = ", ".join(sorted(obs.keys()))
        raise KeyError(f"Missing observation key(s): {missing}. Available keys: {available}")

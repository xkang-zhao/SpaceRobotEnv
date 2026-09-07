from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import gymnasium as gym
import envs  # noqa: F401  # Register Gymnasium environments.
from mujoco_robot.robot_controller import RobotController

from .geometry import (
    as_vector3,
    local_z_rotation as _local_z_rotation,
    normalize as _norm,
    parse_float_list,
    rotation_vector as _rotvec,
    smoothstep as _smoothstep,
)


DEFAULT_LEFT_PANEL_OFFSET = [-0.265751, 0.544939, 0.0]
DEFAULT_APPROACH_AXIS_BODY = [0.0, 1.0, 0.0]
DEFAULT_WORLD_LEFT_AXIS = [0.0, 1.0, 0.0]
DEFAULT_WORLD_UP_AXIS = [0.0, 0.0, 1.0]
DEFAULT_FORWARD_AXIS_LOCAL = [0.0, 0.0, 1.0]
DEFAULT_BASE_GRASP_RPY = [1.57, 0.0, 1.57]
DEFAULT_TOOL_Z_ROTATION = np.pi / 2.0


@dataclass
class LeftAntennaPanelGrabConfig:
    panel_offset: list[float] | None = None
    approach_axis_body: list[float] | None = None
    world_left_axis: list[float] | None = None
    world_up_axis: list[float] | None = None
    forward_axis_local: list[float] | None = None
    target_frame: str = "ee"

    env_id: str = "SpaceUR10e-Satellite-v0"
    grasp_distance: float = 0.0
    pregrasp_distance: float = 0.14
    escape_distance: float = 0.20
    corridor_distance: float = 0.42
    overpass_height: float = 0.28
    base_grasp_orientation_rpy: tuple[float, float, float] = tuple(DEFAULT_BASE_GRASP_RPY)
    tool_z_rotation: float = DEFAULT_TOOL_Z_ROTATION
    rotation_mode: str = "absolute"
    max_delta_xyz: tuple[float, float, float] = (0.015, 0.015, 0.015)
    max_delta_rpy: tuple[float, float, float] = (0.08, 0.08, 0.08)
    motion_scale: float = 0.6
    path_step_size: float = 0.01
    min_segment_steps: int = 20
    position_tolerance: float = 0.03
    grasp_position_tolerance: float = 0.04
    orientation_tolerance: float = 0.12
    close_steps: int = 35
    hold_steps: int = 20
    gripper_closed_target: float = 255.0
    gripper_closed_tolerance: float = 1e-3
    max_steps: int = 800
    stage_timeout: int = 220
    render_mode: str | None = "human"
    control_fps: int = 20
    debug: bool = False

    def __post_init__(self):
        vector_fields = (
            "panel_offset",
            "approach_axis_body",
            "world_left_axis",
            "world_up_axis",
            "forward_axis_local",
            "base_grasp_orientation_rpy",
            "max_delta_xyz",
            "max_delta_rpy",
        )
        for field_name in vector_fields:
            value = getattr(self, field_name)
            if value is not None and len(value) != 3:
                raise ValueError(
                    f"{field_name} must contain exactly 3 values, got {value!r}"
                )
        if self.rotation_mode not in {
            "absolute",
            "relative_local_z",
            "skip",
        }:
            raise ValueError(
                f"Unsupported rotation_mode: {self.rotation_mode!r}"
            )
        if self.control_fps <= 0:
            raise ValueError(
                f"control_fps must be positive, got {self.control_fps}"
            )

class LeftAntennaPanelGrabPlanner:
    def __init__(
        self,
        config: LeftAntennaPanelGrabConfig,
        env=None,
        initial_obs=None,
    ):
        self.cfg = config
        self._owns_env = env is None
        self._env = env or gym.make(config.env_id, render_mode=config.render_mode)
        self._base = self._env.unwrapped
        if initial_obs is None:
            self._obs, _ = self._env.reset()
        else:
            self._obs = initial_obs

        panel_offset = config.panel_offset if config.panel_offset is not None else DEFAULT_LEFT_PANEL_OFFSET
        approach_axis = config.approach_axis_body if config.approach_axis_body is not None else DEFAULT_APPROACH_AXIS_BODY
        world_left_axis = config.world_left_axis if config.world_left_axis is not None else DEFAULT_WORLD_LEFT_AXIS
        self._panel_offset = as_vector3(panel_offset, "panel_offset")
        self._approach_axis_body = _norm(
            as_vector3(approach_axis, "approach_axis_body"),
            np.array(DEFAULT_APPROACH_AXIS_BODY),
        )
        self._world_left_axis = _norm(
            as_vector3(world_left_axis, "world_left_axis"),
            np.array(DEFAULT_WORLD_LEFT_AXIS),
        )

        base_rot = RobotController.euler_to_rotation_matrix(*config.base_grasp_orientation_rpy)
        # The user-requested adjustment: rotate the end-effector about its own local Z axis.
        self._target_rot = base_rot @ _local_z_rotation(config.tool_z_rotation)
        self._stage = "move_left"
        self._stage_steps = 0
        self._close_cnt = 0
        self._hold_cnt = 0
        self._done = False
        self._success = False
        self._failure_reason = None
        self._step_idx = 0
        self._info = {}
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._stage_start = None
        self._stage_target = None
        self._stage_duration = 1
        self._stage_axis = None
        self._pending_step = None

        if config.debug:
            self._log("grasp_offset", self._panel_offset, 5)
            self._log("approach_axis_body", self._approach_axis_body, 5)
            self._log("world_left_axis", self._world_left_axis, 5)
            print(f"target_frame: {self.cfg.target_frame}")

    def _log(self, label, arr, prec=3):
        print(f"{label}: {np.array2string(np.asarray(arr), precision=prec, suppress_small=True)}")

    def _pinch_pos(self):
        return np.asarray(self._base.data.site_xpos[self._base.pinch_site_id], dtype=float)

    def _ee_pos(self):
        return np.asarray(self._base.data.site_xpos[self._base.ee_site_id], dtype=float)

    def _target_frame_pos(self):
        return self._pinch_pos() if self.cfg.target_frame == "pinch" else self._ee_pos()

    def _sat_pos(self):
        return np.asarray(self._base.data.xpos[self._base.target_body_id], dtype=float)

    def _sat_rot(self):
        return np.asarray(self._base.data.xmat[self._base.target_body_id], dtype=float).reshape(3, 3)

    def _ctrl_pose(self):
        p = np.asarray(self._base.controller.get_target_pose(), dtype=float)
        return p[:3], RobotController.euler_to_rotation_matrix(*p[3:6])

    def _compute_targets(self):
        sat_pos, sat_rot = self._sat_pos(), self._sat_rot()
        panel = sat_pos + sat_rot @ self._panel_offset
        axis = _norm(sat_rot @ self._approach_axis_body, np.array([0.0, 1.0, 0.0]))
        grasp = panel + axis * self.cfg.grasp_distance
        current = self._target_frame_pos()
        left_delta = float(np.dot(grasp - current, self._world_left_axis))
        left_pregrasp = current + self._world_left_axis * left_delta
        residual = grasp - left_pregrasp
        return (
            panel,
            left_pregrasp,
            grasp,
            residual,
        )

    def _stage_after_rotation(self):
        return "move_to_grasp"

    def _motion_stage_names(self):
        return ("move_left", "move_to_grasp")

    def _start_motion_segment(self, target, axis=None):
        self._stage_start = self._target_frame_pos().copy()
        self._stage_target = np.asarray(target, dtype=float).copy()
        self._stage_axis = None if axis is None else _norm(np.asarray(axis, dtype=float), np.array([1.0, 0.0, 0.0]))
        dist = float(np.linalg.norm(self._stage_target - self._stage_start))
        self._stage_duration = max(
            self.cfg.min_segment_steps,
            int(np.ceil(dist / max(self.cfg.path_step_size, 1e-6))),
        )
        self._stage_steps = 0

    def _smooth_desired(self):
        if self._stage_start is None or self._stage_target is None:
            raise RuntimeError("Motion segment was not initialized.")
        alpha = _smoothstep((self._stage_steps + 1) / self._stage_duration)
        return self._stage_start + alpha * (self._stage_target - self._stage_start)

    def _advance_stage(self, next_stage):
        if next_stage == "rotate_tool":
            rotation_mode = getattr(self.cfg, "rotation_mode", "absolute")
            if rotation_mode == "skip":
                next_stage = "move_to_grasp"
            elif rotation_mode == "relative_local_z":
                current_rot = self._ctrl_pose()[1]
                self._target_rot = current_rot @ _local_z_rotation(
                    self.cfg.tool_z_rotation
                )
        self._set_stage(next_stage)

    def _set_stage(self, next_stage):
        self._stage = next_stage
        self._stage_steps = 0
        self._stage_start = None
        self._stage_target = None
        self._stage_duration = 1
        self._stage_axis = None

    def _make_action(self, target_pos, gripper_cmd, rotate=True, world_translation_axis=None):
        _, ref_rot = self._ctrl_pose()
        err_world = target_pos - self._target_frame_pos()
        if world_translation_axis is not None:
            axis = _norm(np.asarray(world_translation_axis, dtype=float), np.array([1.0, 0.0, 0.0]))
            err_world = axis * float(np.dot(err_world, axis))
        err_tool = ref_rot.T @ err_world
        dpos = np.clip(
            err_tool,
            -np.array(self.cfg.max_delta_xyz),
            np.array(self.cfg.max_delta_xyz),
        ) * self.cfg.motion_scale
        if rotate:
            drot = np.clip(
                _rotvec(ref_rot.T @ self._target_rot),
                -np.array(self.cfg.max_delta_rpy),
                np.array(self.cfg.max_delta_rpy),
            ) * self.cfg.motion_scale
        else:
            drot = np.zeros(3, dtype=float)
        return np.array([dpos[0], dpos[1], dpos[2], drot[0], drot[1], drot[2], gripper_cmd], dtype=np.float32)

    def _rotate_action(self, gripper_cmd):
        ref_rot = self._ctrl_pose()[1]
        drot = np.clip(
            _rotvec(ref_rot.T @ self._target_rot),
            -np.array(self.cfg.max_delta_rpy),
            np.array(self.cfg.max_delta_rpy),
        ) * self.cfg.motion_scale
        return np.array([0.0, 0.0, 0.0, drot[0], drot[1], drot[2], gripper_cmd], dtype=np.float32)

    def _hold_action(self, gripper_cmd):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32)

    def _gripper_fully_closed(self):
        return (
            self._base.controller.target_gripper
            >= self.cfg.gripper_closed_target - self.cfg.gripper_closed_tolerance
        )

    def _close_stage_complete(self):
        return self._close_cnt >= self.cfg.close_steps and self._gripper_fully_closed()

    def _hold_stage_complete(self):
        return self._hold_cnt >= self.cfg.hold_steps and self._gripper_fully_closed()

    def _print_reward_status(self, reward, terminated, truncated):
        target_dof_adr = self._base.target_dof_adr
        target_vel = self._base.data.qvel[target_dof_adr:target_dof_adr + 6]
        target_vel_norm = float(np.linalg.norm(target_vel))
        raw_success = bool(self._base.is_success())
        print(
            f"[{self._step_idx:04d}] reward={reward:.4f}, "
            f"raw_success={raw_success}, confirmed_success={self._info.get('is_success', False)}, "
            f"success_counter={self._info.get('success_counter', 0)}, "
            f"left_contact={self._info.get('left_contact', False)}, "
            f"right_contact={self._info.get('right_contact', False)}, "
            f"r_contact={self._info.get('r_contact', 0.0):.4f}, "
            f"r_success={self._info.get('r_success', 0.0):.4f}, "
            f"distance={self._info.get('distance', float('nan')):.4f}, "
            f"gripper={self._base.controller.target_gripper:.1f}, "
            f"target_vel={target_vel_norm:.4f}, "
            f"terminated={terminated}, truncated={truncated}"
        )

    def compute_action(self) -> np.ndarray:
        """Compute an action without advancing the environment."""
        if self._done:
            return np.zeros(7, dtype=np.float32)
        if self._pending_step is not None:
            raise RuntimeError("update() must be called before computing another action.")

        panel, left_pregrasp, grasp, residual = self._compute_targets()
        if self.cfg.debug and self._step_idx == 0:
            self._log("panel", panel, 5)
            self._log("left_pregrasp", left_pregrasp, 5)
            self._log("move_to_grasp_target", grasp, 5)
            self._log("grasp", grasp, 5)
            self._log("remaining_after_move_left", residual, 5)

        motion_stages = {
            "move_left": (left_pregrasp, self.cfg.position_tolerance, "rotate_tool", False, self._world_left_axis),
            "move_to_grasp": (grasp, self.cfg.grasp_position_tolerance, "close_gripper", False, None),
        }

        if self._stage in motion_stages:
            raw_target, tol, next_stage, rotate, axis = motion_stages[self._stage]
            if self._stage_target is None:
                self._start_motion_segment(raw_target, axis=axis)
            target = self._stage_target
            axis = self._stage_axis
            desired = self._smooth_desired()
            action = self._make_action(desired, 0.0, rotate=rotate, world_translation_axis=axis)
        elif self._stage == "rotate_tool":
            target, tol, next_stage = left_pregrasp, self.cfg.position_tolerance, self._stage_after_rotation()
            desired = target
            action = self._rotate_action(0.0)
        elif self._stage == "close_gripper":
            target, tol, next_stage = grasp, self.cfg.grasp_position_tolerance, "hold_grasp"
            desired = target
            action = self._hold_action(1.0)
        else:
            target, tol, next_stage = grasp, self.cfg.grasp_position_tolerance, "hold_grasp"
            desired = target
            action = self._hold_action(1.0)

        self._pending_step = {
            "target": target,
            "desired": desired,
            "tolerance": tol,
            "next_stage": next_stage,
            "is_motion_stage": self._stage in motion_stages,
            "motion_axis": self._stage_axis,
        }
        return action

    def update(self, obs, reward, terminated, truncated, info) -> None:
        """Update planner state after the runner advances the environment."""
        if self._pending_step is None:
            raise RuntimeError("compute_action() must be called before update().")

        pending = self._pending_step
        self._pending_step = None
        target = pending["target"]
        desired = pending["desired"]
        tol = pending["tolerance"]
        next_stage = pending["next_stage"]
        is_motion_stage = pending["is_motion_stage"]
        motion_axis = pending["motion_axis"]

        self._obs = obs
        self._info = info
        self._last_reward = float(reward)
        self._last_terminated = bool(terminated)
        self._last_truncated = bool(truncated)
        self._stage_steps += 1
        self._step_idx += 1

        if terminated:
            self._success = bool(self._info.get("is_success", True))
            self._done = True
            if not self._success:
                self._failure_reason = "environment terminated without success"
            return
        if truncated:
            self._done = True
            self._failure_reason = "environment truncated"
            return

        pos_delta = target - self._target_frame_pos()
        pos_err = float(np.linalg.norm(pos_delta))
        axis_err = pos_err
        if is_motion_stage and motion_axis is not None:
            axis_err = abs(float(np.dot(pos_delta, motion_axis)))
        track_err = float(np.linalg.norm(desired - self._target_frame_pos()))
        ref_rot = self._ctrl_pose()[1]
        rot_err = float(np.linalg.norm(_rotvec(ref_rot.T @ self._target_rot)))

        if self._step_idx % 50 == 0:
            print(f"[{self._step_idx:04d}] stage={self._stage}, err={pos_err:.4f}, track={track_err:.4f} | "
                  f"reward={reward:.4f}, confirmed_success={self._info.get('is_success', False)} | "
                  f"sat={self._sat_pos()} | ee={self._ee_pos()} | "
                  f"pinch={self._pinch_pos()} | desired={desired}")

        if self._stage in {"close_gripper", "hold_grasp"} and (
            self._stage_steps == 1
            or self._stage_steps % 5 == 0
            or self._info.get("left_contact", False)
            or self._info.get("right_contact", False)
            or self._info.get("is_success", False)
        ):
            self._print_reward_status(reward, terminated, truncated)

        if is_motion_stage:
            path_complete = self._stage_steps >= self._stage_duration
            reached = path_complete and axis_err <= tol
            timed_out = self._stage_steps >= self.cfg.stage_timeout

            if reached:
                print(f"[{self._step_idx:04d}] {self._stage} -> {next_stage} (reached)  err={pos_err:.4f}, axis_err={axis_err:.4f}")
                self._advance_stage(next_stage)
            elif timed_out:
                print(f"[{self._step_idx:04d}] FAILED: {self._stage} timeout  err={pos_err:.4f}, axis_err={axis_err:.4f}")
                self._done = True
                self._failure_reason = f"{self._stage} timeout"
        elif self._stage == "rotate_tool":
            if rot_err <= self.cfg.orientation_tolerance:
                print(f"[{self._step_idx:04d}] rotate_tool -> {next_stage} (reached)  rot_err={rot_err:.4f}")
                self._advance_stage(next_stage)
            elif self._stage_steps >= self.cfg.stage_timeout:
                print(f"[{self._step_idx:04d}] FAILED: rotate_tool timeout  rot_err={rot_err:.4f}")
                self._done = True
                self._failure_reason = "rotate_tool timeout"
        elif self._stage == "close_gripper":
            self._close_cnt += 1
            if self._info.get("is_success", False) and self._gripper_fully_closed():
                self._success = True
                self._done = True
            if self._close_stage_complete():
                self._advance_stage(next_stage)
            elif self._stage_steps >= self.cfg.stage_timeout:
                print(
                    f"[{self._step_idx:04d}] FAILED: close_gripper timeout  "
                    f"gripper={self._base.controller.target_gripper:.1f}"
                )
                self._done = True
                self._failure_reason = "close_gripper timeout"
        else:
            self._hold_cnt += 1
            if self._info.get("is_success", False) and self._gripper_fully_closed():
                self._success = True
                self._done = True
            elif self._hold_stage_complete():
                self._done = True
                self._failure_reason = "grasp was not confirmed"

    def get_action(self) -> np.ndarray:
        """Compatibility API: compute and immediately execute one action."""
        action = self.compute_action()
        if self._done:
            return action
        transition = self._env.step(action)
        self.update(*transition)
        return action

    @property
    def is_done(self) -> bool:
        return self._done or (self._info.get("distance", 0) > 2.0)

    @property
    def is_success(self) -> bool:
        return self._success

    @property
    def failure_reason(self) -> str | None:
        if self._failure_reason:
            return self._failure_reason
        if self._info.get("distance", 0) > 2.0:
            return "target exceeded the distance limit"
        return None

    @property
    def step_index(self) -> int:
        return self._step_idx

    @property
    def max_steps(self) -> int:
        return self.cfg.max_steps

    @property
    def control_fps(self) -> int:
        return self.cfg.control_fps

    def close(self):
        if self._owns_env:
            self._env.close()

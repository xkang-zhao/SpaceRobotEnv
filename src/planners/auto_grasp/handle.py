from dataclasses import dataclass, field

import numpy as np

import gymnasium as gym
import envs  # noqa: F401  # Register Gymnasium environments.
from mujoco_robot.robot_controller import RobotController

from .geometry import (
    as_vector3,
    normalize as _norm,
    parse_float_list,
    rotation_vector as _rotvec,
)


# ---------------------------------------------------------------------------
# config (like PlannerTeleopConfig)
# ---------------------------------------------------------------------------

@dataclass
class AutograbConfig:
    handle_offset: list[float] = field(
        default_factory=lambda: [-0.600, 0.223, 0.004]
    )

    # fixed parameters
    env_id: str = "SpaceUR10e-Satellite-v0"
    grasp_distance: float = 0.02
    pregrasp_distance: float = 0.05
    grasp_orientation_rpy: tuple[float, float, float] = (1.57, 0.0, 1.57)
    max_delta_xyz: tuple[float, float, float] = (0.015, 0.015, 0.015)
    max_delta_rpy: tuple[float, float, float] = (0.08, 0.08, 0.08)
    motion_scale: float = 0.6
    position_tolerance: float = 0.025
    grasp_position_tolerance: float = 0.05
    orientation_tolerance: float = 0.12
    close_steps: int = 35
    hold_steps: int = 20
    max_steps: int = 500
    stage_timeout: int = 200
    render_mode: str | None = "human"
    control_fps: int = 20
    debug: bool = False

    def __post_init__(self):
        vector_fields = (
            "handle_offset",
            "grasp_orientation_rpy",
            "max_delta_xyz",
            "max_delta_rpy",
        )
        for field_name in vector_fields:
            value = getattr(self, field_name)
            if len(value) != 3:
                raise ValueError(
                    f"{field_name} must contain exactly 3 values, got {value!r}"
                )
        if self.control_fps <= 0:
            raise ValueError(
                f"control_fps must be positive, got {self.control_fps}"
            )


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# planner (like PlannerTeleop)
# ---------------------------------------------------------------------------

class AutograbPlanner:
    def __init__(self, config: AutograbConfig, env=None, initial_obs=None):
        self.cfg = config
        self._owns_env = env is None
        self._env = env or gym.make(config.env_id, render_mode=config.render_mode)
        self._base = self._env.unwrapped
        if initial_obs is None:
            self._obs, _ = self._env.reset()
        else:
            self._obs = initial_obs

        self._handle_offset = as_vector3(config.handle_offset, "handle_offset")
        self._target_rot = RobotController.euler_to_rotation_matrix(*config.grasp_orientation_rpy)
        self._pinch_offset = self._compute_pinch_offset()

        self._stage = "move_to_pregrasp"
        self._stage_steps = 0
        self._best_err = float("inf")
        self._stagnate = 0
        self._close_cnt = 0
        self._hold_cnt = 0
        self._done = False
        self._success = False
        self._failure_reason = None
        self._step_idx = 0
        self._info = {}
        self._prev_desired = None   # for smooth stage transition
        self._interp_left = 0       # remaining interpolation steps
        self._pending_step = None

        if config.debug:
            self._log("handle_offset", self._handle_offset, 5)
            self._log("pinch_offset", self._pinch_offset, 5)

    # ---- helpers ----
    def _log(self, label, arr, prec=3):
        print(f"{label}: {np.array2string(np.asarray(arr), precision=prec, suppress_small=True)}")

    def _pinch_pos(self):
        return np.asarray(self._base.data.site_xpos[self._base.pinch_site_id], dtype=float)

    def _ee_pos(self):
        return np.asarray(self._base.data.site_xpos[self._base.ee_site_id], dtype=float)

    def _sat_pos(self):
        return np.asarray(self._base.data.xpos[self._base.target_body_id], dtype=float)

    def _sat_rot(self):
        return np.asarray(self._base.data.xmat[self._base.target_body_id], dtype=float).reshape(3, 3)

    def _ctrl_pose(self):
        p = np.asarray(self._base.controller.get_target_pose(), dtype=float)
        return p[:3], RobotController.euler_to_rotation_matrix(*p[3:6])

    def _compute_pinch_offset(self):
        cp = self._ee_pos()
        cr = np.asarray(self._base.data.site_xmat[self._base.ee_site_id], dtype=float).reshape(3, 3)
        return cr.T @ (self._pinch_pos() - cp)

    def _compute_targets(self):
        sp, sr = self._sat_pos(), self._sat_rot()
        handle = sp + sr @ self._handle_offset
        # TODO：这里的轴向计算有点 hack，直接用卫星的 -X 轴作为抓取方向，可能不够鲁棒
        axis = _norm(sr @ np.array([-1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))
        return (handle,
                handle + axis * self.cfg.pregrasp_distance,
                handle + axis * self.cfg.grasp_distance)

    def _make_action(self, target_pos, gripper_cmd):
        ref_pos, ref_rot = self._ctrl_pose()
        err_w = target_pos - ref_pos
        err_t = ref_rot.T @ err_w
        dpos = np.clip(err_t, -np.array(self.cfg.max_delta_xyz),
                       np.array(self.cfg.max_delta_xyz)) * self.cfg.motion_scale
        drot = np.clip(_rotvec(ref_rot.T @ self._target_rot),
                       -np.array(self.cfg.max_delta_rpy),
                       np.array(self.cfg.max_delta_rpy)) * self.cfg.motion_scale
        return np.array([dpos[0], dpos[1], dpos[2], drot[0], drot[1], drot[2], gripper_cmd], dtype=np.float32)

    # ---- public API ----
    def compute_action(self) -> np.ndarray:
        """Compute an action without advancing the environment."""
        if self._done:
            return np.zeros(7, dtype=np.float32)
        if self._pending_step is not None:
            raise RuntimeError("update() must be called before computing another action.")

        handle, pregrasp, grasp = self._compute_targets()
        if self.cfg.debug and self._step_idx == 0:
            self._log("handle", handle, 5)
            self._log("pregrasp", pregrasp, 5)
            self._log("grasp", grasp, 5)

        # stage
        if self._stage == "move_to_pregrasp":
            desired = pregrasp
            gcmd = 0.0
            tol = self.cfg.position_tolerance
            nxt = "close_gripper" if abs(self.cfg.pregrasp_distance - self.cfg.grasp_distance) < 1e-6 else "approach_handle"
        elif self._stage == "approach_handle":
            desired = grasp
            gcmd = 0.0
            tol = self.cfg.grasp_position_tolerance
            nxt = "close_gripper"
        elif self._stage == "close_gripper":
            desired = grasp
            gcmd = 1.0
            tol = self.cfg.grasp_position_tolerance
            nxt = "hold_grasp"
        else:
            desired = grasp
            gcmd = 1.0
            tol = self.cfg.grasp_position_tolerance
            nxt = "hold_grasp"

        # smooth interpolation from previous target to new target
        if self._interp_left > 0 and self._prev_desired is not None:
            alpha = self._interp_left / 20.0
            desired = self._prev_desired + (desired - self._prev_desired) * (1 - alpha)
            self._interp_left -= 1

        ctrl_target = desired - self._target_rot @ self._pinch_offset
        action = self._make_action(ctrl_target, gcmd)

        self._pending_step = {
            "desired": desired,
            "tolerance": tol,
            "next_stage": nxt,
        }
        return action

    def update(self, obs, reward, terminated, truncated, info) -> None:
        """Update planner state after the runner advances the environment."""
        if self._pending_step is None:
            raise RuntimeError("compute_action() must be called before update().")

        pending = self._pending_step
        self._pending_step = None
        desired = pending["desired"]
        tol = pending["tolerance"]
        next_stage = pending["next_stage"]

        self._obs = obs
        self._info = info
        self._stage_steps += 1
        self._step_idx += 1

        if terminated:
            self._success = bool(info.get("is_success", True))
            self._done = True
            if not self._success:
                self._failure_reason = "environment terminated without success"
            return
        if truncated:
            self._done = True
            self._failure_reason = "environment truncated"
            return

        pos_err = float(np.linalg.norm(desired - self._pinch_pos()))
        ref_rot = self._ctrl_pose()[1]
        rot_err = float(np.linalg.norm(_rotvec(ref_rot.T @ self._target_rot)))

        if self._step_idx % 50 == 0:
            print(f"[{self._step_idx:04d}] stage={self._stage}, err={pos_err:.4f} | "
                  f"sat={self._sat_pos()} | ee={self._ee_pos()} | "
                  f"pinch={self._pinch_pos()} | desired={desired}")

        # transition
        if self._stage in {"move_to_pregrasp", "approach_handle"}:
            ok = rot_err <= self.cfg.orientation_tolerance
            reached = pos_err <= tol and ok
            timed_out = self._stage_steps >= self.cfg.stage_timeout
            if pos_err < self._best_err * 0.99:
                self._best_err = pos_err
                self._stagnate = 0
            else:
                self._stagnate += 1
            stagnated = self._stagnate >= 15 and self._stage_steps >= 30

            if reached or timed_out or stagnated:
                if reached:
                    reason = "reached"
                elif stagnated:
                    reason = "stagnated"
                else:
                    reason = "timeout"
                print(
                    f"[{self._step_idx:04d}] "
                    f"{self._stage} -> {next_stage} ({reason})  "
                    f"err={pos_err:.4f}"
                )
                self._prev_desired = desired.copy()   # save for smooth transition
                self._interp_left = 20                 # 20-step blend
                self._stage = next_stage
                self._stage_steps = 0
                self._best_err = float("inf")
                self._stagnate = 0
        elif self._stage == "close_gripper":
            self._close_cnt += 1
            if self._info.get("is_success", False):
                self._success = True
                self._done = True
            if self._close_cnt >= self.cfg.close_steps:
                self._stage = next_stage
                self._stage_steps = 0
        else:
            self._hold_cnt += 1
            if self._info.get("is_success", False):
                self._success = True
                self._done = True
            elif self._hold_cnt >= self.cfg.hold_steps:
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

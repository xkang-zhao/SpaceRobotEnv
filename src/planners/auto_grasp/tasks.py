"""Shared task registry and collection adapter for automatic grasp planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .profiles import LATERAL_GRASP_PROFILES


PlannerFamily = Literal["cube", "handle", "lateral"]


@dataclass(frozen=True)
class AutoGraspTaskSpec:
    """Stable task metadata shared by interactive and collection CLIs."""

    name: str
    env_id: str
    family: PlannerFamily
    instruction: str
    success_label: str


AUTO_GRASP_TASKS: dict[str, AutoGraspTaskSpec] = {
    "cube": AutoGraspTaskSpec(
        name="cube",
        env_id="SpaceUR10e-Cube-v0",
        family="cube",
        instruction="Grab the red cube",
        success_label="cube grasp confirmed",
    ),
    "satellite_handle": AutoGraspTaskSpec(
        name="satellite_handle",
        env_id="SpaceUR10e-Satellite-v0",
        family="handle",
        instruction="Grab the satellite handle",
        success_label="satellite handle grasp confirmed",
    ),
}
for _name, _profile in LATERAL_GRASP_PROFILES.items():
    AUTO_GRASP_TASKS[_name] = AutoGraspTaskSpec(
        name=_name,
        env_id=_profile.env_id,
        family="lateral",
        instruction=_profile.description.rstrip("."),
        success_label=_profile.success_label,
    )

TASK_NAMES = tuple(AUTO_GRASP_TASKS)


def get_task_spec(task_name: str) -> AutoGraspTaskSpec:
    try:
        return AUTO_GRASP_TASKS[task_name]
    except KeyError as exc:
        choices = ", ".join(TASK_NAMES)
        raise ValueError(
            f"Unknown automatic grasp task {task_name!r}. "
            f"Choose one of: {choices}"
        ) from exc


class AutoGraspSession:
    """Normalize the cube and stateful planner APIs for data collection."""

    def __init__(self, task_name: str, env, *, debug: bool = False):
        self.task = get_task_spec(task_name)
        self.env = env
        self.debug = bool(debug)
        self.planner = None
        self.obs = None
        self._transition_success = False
        self._transition_done = False
        self._failure_reason = None
        self._step_index = 0

    def reset(self, seed: int):
        obs, _ = self.env.reset(seed=seed)
        self.obs = obs
        self._transition_success = False
        self._transition_done = False
        self._failure_reason = None
        self._step_index = 0
        viewer = getattr(self.env.unwrapped, "viewer", None)

        if self.task.family == "cube":
            from .cube import CubeAutoGraspConfig, CubeAutoGraspPlanner

            config = CubeAutoGraspConfig(
                env_id=self.task.env_id,
                render_mode="human" if viewer else "none",
                debug=self.debug,
            )
            planner = CubeAutoGraspPlanner(config)
            planner.bind_env_base(self.env.unwrapped)
            planner.set_reference_pose(obs["ee_pose"])
            self.planner = planner
        elif self.task.family == "handle":
            from .handle import AutograbConfig, AutograbPlanner

            config = AutograbConfig(
                env_id=self.task.env_id,
                render_mode="human" if viewer else None,
                debug=self.debug,
            )
            self.planner = AutograbPlanner(
                config,
                env=self.env,
                initial_obs=obs,
            )
        else:
            from .lateral import (
                LeftAntennaPanelGrabConfig,
                LeftAntennaPanelGrabPlanner,
            )
            from .lateral_cli import config_from_profile

            profile = LATERAL_GRASP_PROFILES[self.task.name]
            config = config_from_profile(
                profile,
                LeftAntennaPanelGrabConfig,
            )
            config.render_mode = "human" if viewer else None
            config.debug = self.debug
            self.planner = LeftAntennaPanelGrabPlanner(
                config,
                env=self.env,
                initial_obs=obs,
            )
        return obs

    def compute_action(self, obs) -> np.ndarray:
        if self.planner is None:
            raise RuntimeError("reset() must be called before compute_action().")
        if self.task.family == "cube":
            from .cube import action_dict_to_vector

            action_dict = self.planner.get_action(
                obs,
                # Dataset collection accepts only the environment's
                # consecutive-step confirmation emitted after env.step().
                # Keeping the cube planner in its hold stage until that
                # transition prevents a transient contact from ending the
                # attempt early.
                episode_success=False,
            )
            return np.asarray(
                action_dict_to_vector(action_dict),
                dtype=np.float32,
            )
        return np.asarray(self.planner.compute_action(), dtype=np.float32)

    def update(self, obs, reward, terminated, truncated, info) -> None:
        if self.planner is None:
            raise RuntimeError("reset() must be called before update().")
        self.obs = obs
        self._step_index += 1
        confirmed = bool(terminated and info.get("is_success", False))
        if self.task.family == "cube":
            if confirmed:
                self._transition_success = True
                self._transition_done = True
            elif truncated:
                self._transition_done = True
                self._failure_reason = "environment truncated"
            elif self.planner.should_rerecord_episode():
                self._transition_done = True
                self._failure_reason = "planner timed out"
        else:
            self.planner.update(
                obs,
                reward,
                terminated,
                truncated,
                info,
            )

    @property
    def is_done(self) -> bool:
        if self.planner is None:
            return False
        if self.task.family == "cube":
            return (
                self._transition_done
                or self.planner.should_rerecord_episode()
            )
        return bool(self.planner.is_done)

    @property
    def is_success(self) -> bool:
        if self.planner is None:
            return False
        if self.task.family == "cube":
            return self._transition_success
        return bool(self.planner.is_success)

    @property
    def failure_reason(self) -> str | None:
        if self.task.family == "cube":
            return self._failure_reason
        if self.planner is None:
            return None
        return self.planner.failure_reason

    @property
    def step_index(self) -> int:
        if self.planner is None:
            return self._step_index
        if self.task.family == "cube":
            return int(self.planner.step_index)
        return int(self.planner.step_index)

    @property
    def max_steps(self) -> int:
        if self.planner is None:
            return 0
        if self.task.family == "cube":
            return int(self.planner.config.episode_timeout_steps)
        return int(self.planner.max_steps)

    @property
    def control_fps(self) -> int:
        if self.planner is None:
            return 20
        if self.task.family == "cube":
            return int(self.planner.config.fps)
        return int(self.planner.control_fps)

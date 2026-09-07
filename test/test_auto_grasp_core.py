import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from planners.auto_grasp.handle import AutograbConfig, AutograbPlanner
from planners.auto_grasp.lateral import (
    LeftAntennaPanelGrabConfig,
    LeftAntennaPanelGrabPlanner,
)
from planners.auto_grasp.profiles import LATERAL_GRASP_PROFILES


class AutoGraspCoreTests(unittest.TestCase):
    def test_original_satellite_offsets_are_preserved(self):
        self.assertEqual(
            AutograbConfig().handle_offset,
            [-0.600, 0.223, 0.004],
        )
        self.assertEqual(
            LATERAL_GRASP_PROFILES[
                "satellite_left_antenna_panel"
            ].grasp_offset_body,
            (-0.265751, 0.544939, 0.0),
        )

    def test_stateful_planners_accept_seeded_initial_observation(self):
        initial_obs = {"seeded": True}
        data = SimpleNamespace(
            site_xpos=np.zeros((2, 3)),
            site_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
        )
        base = SimpleNamespace(
            data=data,
            ee_site_id=0,
            pinch_site_id=1,
        )
        env = SimpleNamespace(
            unwrapped=base,
            reset=lambda: self.fail(
                "Planner must not reset when initial_obs is provided."
            ),
        )

        planners = [
            AutograbPlanner(
                AutograbConfig(),
                env=env,
                initial_obs=initial_obs,
            ),
            LeftAntennaPanelGrabPlanner(
                LeftAntennaPanelGrabConfig(),
                env=env,
                initial_obs=initial_obs,
            ),
        ]

        for planner in planners:
            with self.subTest(planner=type(planner).__name__):
                self.assertIs(planner._obs, initial_obs)

    def test_all_calibrated_lateral_profiles_are_registered(self):
        self.assertEqual(
            set(LATERAL_GRASP_PROFILES),
            {
                "satellite_left_antenna_panel",
                "satellite2_left_truss_connection",
                "satellite3_left_antenna_panel",
                "satellite3_upper_rod",
                "debris_antenna_panel",
                "debris_truss",
            },
        )
        self.assertEqual(
            LATERAL_GRASP_PROFILES["debris_truss"].grasp_offset_body,
            (-0.136453, 0.075573, 0.118878),
        )
        self.assertEqual(
            LATERAL_GRASP_PROFILES[
                "satellite2_left_truss_connection"
            ].rotation_mode,
            "skip",
        )

    def test_compute_action_does_not_advance_environment(self):
        planner = LeftAntennaPanelGrabPlanner.__new__(
            LeftAntennaPanelGrabPlanner
        )
        planner._done = False
        planner._pending_step = None
        planner._stage = "move_left"
        planner._stage_target = np.zeros(3)
        planner._stage_axis = np.array([0.0, 1.0, 0.0])
        planner._world_left_axis = planner._stage_axis
        planner._step_idx = 0
        planner.cfg = SimpleNamespace(
            debug=False,
            position_tolerance=0.03,
            grasp_position_tolerance=0.04,
        )
        planner._compute_targets = lambda: (
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
        )
        planner._smooth_desired = lambda: np.zeros(3)
        planner._make_action = lambda *_args, **_kwargs: np.ones(
            7,
            dtype=np.float32,
        )
        planner._env = SimpleNamespace(
            step=lambda _action: self.fail(
                "compute_action() must not call env.step()"
            )
        )

        action = planner.compute_action()

        np.testing.assert_allclose(action, np.ones(7))
        self.assertIsNotNone(planner._pending_step)

    def test_legacy_get_action_still_advances_one_step(self):
        planner = LeftAntennaPanelGrabPlanner.__new__(
            LeftAntennaPanelGrabPlanner
        )
        planner._done = False
        action = np.arange(7, dtype=np.float32)
        transition = ({}, 0.0, False, False, {})
        calls = []
        planner.compute_action = lambda: action
        planner._env = SimpleNamespace(
            step=lambda actual: calls.append(actual.copy()) or transition
        )
        planner.update = lambda *actual: calls.append(actual)

        returned = planner.get_action()

        np.testing.assert_allclose(returned, action)
        np.testing.assert_allclose(calls[0], action)
        self.assertEqual(calls[1], transition)

    def test_hold_completion_without_contact_is_failure(self):
        planner = LeftAntennaPanelGrabPlanner.__new__(
            LeftAntennaPanelGrabPlanner
        )
        planner._pending_step = {
            "target": np.zeros(3),
            "desired": np.zeros(3),
            "tolerance": 0.04,
            "next_stage": "hold_grasp",
            "is_motion_stage": False,
            "motion_axis": None,
        }
        planner._stage = "hold_grasp"
        planner._stage_steps = 1
        planner._step_idx = 0
        planner._hold_cnt = 0
        planner._done = False
        planner._success = False
        planner._failure_reason = None
        planner._target_rot = np.eye(3)
        planner._target_frame_pos = lambda: np.zeros(3)
        planner._ctrl_pose = lambda: (np.zeros(3), np.eye(3))
        planner.cfg = SimpleNamespace(
            hold_steps=1,
            gripper_closed_target=255.0,
            gripper_closed_tolerance=1e-3,
        )
        planner._base = SimpleNamespace(
            is_success=lambda: False,
            controller=SimpleNamespace(target_gripper=255.0),
        )

        planner.update({}, 0.0, False, False, {})

        self.assertTrue(planner.is_done)
        self.assertFalse(planner.is_success)
        self.assertEqual(planner.failure_reason, "grasp was not confirmed")

    def test_handle_motion_stagnation_advances_instead_of_failing(self):
        planner = AutograbPlanner.__new__(AutograbPlanner)
        planner._pending_step = {
            "desired": np.array([1.0, 0.0, 0.0]),
            "tolerance": 0.025,
            "next_stage": "approach_handle",
        }
        planner._stage = "move_to_pregrasp"
        planner._stage_steps = 29
        planner._step_idx = 29
        planner._done = False
        planner._success = False
        planner._failure_reason = None
        planner._best_err = 1.0
        planner._stagnate = 14
        planner._target_rot = np.eye(3)
        planner._pinch_pos = lambda: np.zeros(3)
        planner._ctrl_pose = lambda: (np.zeros(3), np.eye(3))
        planner.cfg = SimpleNamespace(
            orientation_tolerance=0.12,
            stage_timeout=200,
        )

        planner.update({}, 0.0, False, False, {})

        self.assertEqual(planner._stage, "approach_handle")
        self.assertFalse(planner.is_done)
        self.assertIsNone(planner.failure_reason)
        self.assertEqual(planner._stage_steps, 0)
        self.assertEqual(planner._stagnate, 0)

    def test_environment_truncation_stops_both_stateful_planners(self):
        planners = [
            (
                LeftAntennaPanelGrabPlanner.__new__(
                    LeftAntennaPanelGrabPlanner
                ),
                {
                    "target": np.zeros(3),
                    "desired": np.zeros(3),
                    "tolerance": 0.03,
                    "next_stage": "move_to_grasp",
                    "is_motion_stage": True,
                    "motion_axis": None,
                },
            ),
            (
                AutograbPlanner.__new__(AutograbPlanner),
                {
                    "desired": np.zeros(3),
                    "tolerance": 0.03,
                    "next_stage": "approach_handle",
                },
            ),
        ]
        for planner, pending in planners:
            with self.subTest(planner=type(planner).__name__):
                planner._pending_step = pending
                planner._stage_steps = 0
                planner._step_idx = 0
                planner._done = False
                planner._success = False
                planner._failure_reason = None

                planner.update({}, 0.0, False, True, {})

                self.assertTrue(planner.is_done)
                self.assertFalse(planner.is_success)
                self.assertEqual(
                    planner.failure_reason,
                    "environment truncated",
                )


if __name__ == "__main__":
    unittest.main()

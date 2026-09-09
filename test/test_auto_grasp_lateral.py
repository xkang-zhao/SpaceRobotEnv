import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from planners.auto_grasp.geometry import parse_float_list
from planners.auto_grasp.lateral import (
    LeftAntennaPanelGrabConfig,
    LeftAntennaPanelGrabPlanner,
)
from planners.auto_grasp.lateral_cli import (
    build_lateral_config,
    config_from_profile,
    make_lateral_parser,
)
from planners.auto_grasp.profiles import LATERAL_GRASP_PROFILES


EXPECTED_PROFILES = {
    "satellite_left_antenna_panel": {
        "env_id": "SpaceUR10e-Satellite-v0",
        "offset": (-0.265751, 0.544939, 0.0),
        "rotation": np.pi / 2.0,
        "rotation_mode": "absolute",
        "position_tolerance": 0.03,
        "grasp_position_tolerance": 0.04,
    },
    "satellite2_left_truss_connection": {
        "env_id": "SpaceUR10e-Satellite2-v0",
        "offset": (-1.26778, 0.476677, 0.426568),
        "rotation": 0.0,
        "rotation_mode": "skip",
        "position_tolerance": 0.03,
        "grasp_position_tolerance": 0.04,
    },
    "satellite3_left_antenna_panel": {
        "env_id": "SpaceUR10e-Satellite3-v0",
        "offset": (-0.776456, 0.576061, -0.007349),
        "rotation": np.pi / 2.0,
        "rotation_mode": "absolute",
        "position_tolerance": 0.03,
        "grasp_position_tolerance": 0.04,
    },
    "satellite3_upper_rod": {
        "env_id": "SpaceUR10e-Satellite3-v0",
        "offset": (-0.678392, 0.00316, 0.210),
        "rotation": 0.0,
        "rotation_mode": "skip",
        "position_tolerance": 0.03,
        "grasp_position_tolerance": 0.04,
    },
    "debris_antenna_panel": {
        "env_id": "SpaceUR10e-DebrisAntennaPanel-v0",
        "offset": (-0.46839, 0.127419, 0.132495),
        "rotation": -0.61,
        "rotation_mode": "relative_local_z",
        "position_tolerance": 0.03,
        "grasp_position_tolerance": 0.04,
    },
    "debris_truss": {
        "env_id": "SpaceUR10e-DebrisTruss-v0",
        "offset": (-0.136453, 0.075573, 0.118878),
        "rotation": -0.8,
        "rotation_mode": "relative_local_z",
        "position_tolerance": 0.015,
        "grasp_position_tolerance": 0.012,
    },
}


class LateralProfileTests(unittest.TestCase):
    def test_profiles_preserve_all_task_calibrations(self):
        self.assertEqual(
            set(LATERAL_GRASP_PROFILES),
            set(EXPECTED_PROFILES),
        )

        for name, expected in EXPECTED_PROFILES.items():
            with self.subTest(task=name):
                profile = LATERAL_GRASP_PROFILES[name]
                config = config_from_profile(
                    profile,
                    LeftAntennaPanelGrabConfig,
                )

                self.assertEqual(config.env_id, expected["env_id"])
                np.testing.assert_allclose(
                    config.panel_offset,
                    expected["offset"],
                )
                self.assertAlmostEqual(
                    config.tool_z_rotation,
                    expected["rotation"],
                )
                self.assertEqual(
                    config.rotation_mode,
                    expected["rotation_mode"],
                )
                self.assertEqual(
                    config.position_tolerance,
                    expected["position_tolerance"],
                )
                self.assertEqual(
                    config.grasp_position_tolerance,
                    expected["grasp_position_tolerance"],
                )

    def test_cli_overrides_environment_and_grasp_offset(self):
        profile = LATERAL_GRASP_PROFILES["debris_truss"]
        parser = make_lateral_parser(
            profile,
            LeftAntennaPanelGrabConfig,
            parse_float_list,
        )
        args = parser.parse_args(
            [
                "--env",
                "SpaceUR10e-DebrisAntennaPanel-v0",
                "--grasp_offset",
                "[1.0, 2.0, 3.0]",
                "--tool_z_rotation",
                "0.4",
            ]
        )

        config = build_lateral_config(
            args,
            profile,
            LeftAntennaPanelGrabConfig,
        )

        self.assertEqual(
            config.env_id,
            "SpaceUR10e-DebrisAntennaPanel-v0",
        )
        self.assertEqual(config.panel_offset, [1.0, 2.0, 3.0])
        self.assertEqual(config.tool_z_rotation, 0.4)
        self.assertEqual(config.rotation_mode, "relative_local_z")
        self.assertEqual(config.position_tolerance, 0.015)
        self.assertEqual(config.grasp_position_tolerance, 0.012)

    def test_task_without_rotation_does_not_expose_rotation_option(self):
        profile = LATERAL_GRASP_PROFILES[
            "satellite2_left_truss_connection"
        ]
        parser = make_lateral_parser(
            profile,
            LeftAntennaPanelGrabConfig,
            parse_float_list,
        )

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--tool_z_rotation", "0.4"])


class LateralPlannerBehaviorTests(unittest.TestCase):
    def test_skip_rotation_mode_moves_directly_to_grasp(self):
        planner = LeftAntennaPanelGrabPlanner.__new__(
            LeftAntennaPanelGrabPlanner
        )
        planner.cfg = SimpleNamespace(
            rotation_mode="skip",
            tool_z_rotation=0.0,
        )

        planner._advance_stage("rotate_tool")

        self.assertEqual(planner._stage, "move_to_grasp")

    def test_absolute_rotation_mode_keeps_rotation_stage(self):
        planner = LeftAntennaPanelGrabPlanner.__new__(
            LeftAntennaPanelGrabPlanner
        )
        planner.cfg = SimpleNamespace(
            rotation_mode="absolute",
            tool_z_rotation=np.pi / 2.0,
        )

        planner._advance_stage("rotate_tool")

        self.assertEqual(planner._stage, "rotate_tool")

    def test_relative_rotation_uses_current_end_effector_pose(self):
        planner = LeftAntennaPanelGrabPlanner.__new__(
            LeftAntennaPanelGrabPlanner
        )
        current_rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        local_rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        planner.cfg = SimpleNamespace(
            rotation_mode="relative_local_z",
            tool_z_rotation=-0.8,
        )
        planner._ctrl_pose = lambda: (None, current_rotation)

        with patch(
            "planners.auto_grasp.lateral._local_z_rotation",
            return_value=local_rotation,
        ) as local_z_rotation:
            planner._advance_stage("rotate_tool")

        local_z_rotation.assert_called_once_with(-0.8)
        np.testing.assert_allclose(
            planner._target_rot,
            current_rotation @ local_rotation,
        )
        self.assertEqual(planner._stage, "rotate_tool")

    def test_gripper_completion_requires_fully_closed_target(self):
        planner = LeftAntennaPanelGrabPlanner.__new__(
            LeftAntennaPanelGrabPlanner
        )
        planner.cfg = LeftAntennaPanelGrabConfig(
            close_steps=1,
            hold_steps=3,
        )
        planner._close_cnt = 10
        planner._hold_cnt = 3
        planner._base = SimpleNamespace(
            controller=SimpleNamespace(
                target_gripper=(
                    planner.cfg.gripper_closed_target - 1.0
                )
            )
        )

        self.assertFalse(planner._close_stage_complete())
        self.assertFalse(planner._hold_stage_complete())

        planner._base.controller.target_gripper = (
            planner.cfg.gripper_closed_target
        )
        self.assertTrue(planner._close_stage_complete())
        self.assertTrue(planner._hold_stage_complete())

    def test_motion_plan_moves_from_rotation_to_grasp(self):
        planner = LeftAntennaPanelGrabPlanner.__new__(
            LeftAntennaPanelGrabPlanner
        )

        self.assertEqual(
            planner._stage_after_rotation(),
            "move_to_grasp",
        )
        self.assertNotIn(
            "move_vertical",
            planner._motion_stage_names(),
        )
        self.assertIn(
            "move_to_grasp",
            planner._motion_stage_names(),
        )


if __name__ == "__main__":
    unittest.main()

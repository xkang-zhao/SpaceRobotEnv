import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from planners.auto_grasp.cube import (
    CubeAutoGraspConfig,
    CubeAutoGraspPlanner,
    action_dict_to_vector,
)


class CubeAutoGraspTests(unittest.TestCase):
    def test_default_config_targets_cube_environment(self):
        cfg = CubeAutoGraspConfig()

        self.assertEqual(cfg.env_id, "SpaceUR10e-Cube-v0")
        self.assertEqual(cfg.front_offset_world, [-0.12, 0.0, 0.0])
        self.assertEqual(cfg.grasp_offset_world, [-0.02, 0.0, 0.0])
        self.assertEqual(cfg.target_frame, "pinch")
        self.assertEqual(cfg.fps, 20)

    def test_action_dict_to_vector_uses_env_action_order(self):
        action = action_dict_to_vector(
            {
                "delta_x": 1,
                "delta_y": 2,
                "delta_z": 3,
                "delta_roll": 4,
                "delta_pitch": 5,
                "delta_yaw": 6,
                "delta_gripper": 7,
            }
        )

        self.assertEqual(
            action,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        )

    def test_planner_moves_front_then_grasp_then_closes(self):
        planner = CubeAutoGraspPlanner(
            CubeAutoGraspConfig(
                front_position_tolerance=0.02,
                grasp_position_tolerance=0.02,
                max_delta_xyz=[1.0, 1.0, 1.0],
                motion_scale=1.0,
            )
        )
        obs = {
            "ee_pose": [1.0, 0.0, 2.7, 1.0, 0.0, 0.0, 0.0],
            "target_pose": [1.2, 0.0, 2.7, 1.0, 0.0, 0.0, 0.0],
        }

        action = planner.get_action(obs)
        self.assertEqual(planner.stage, "move_to_object_front")
        self.assertAlmostEqual(action["delta_x"], 0.08)

        obs["ee_pose"][:3] = [1.08, 0.0, 2.7]
        planner.get_action(obs)
        self.assertEqual(planner.stage, "approach_object")

        obs["ee_pose"][:3] = [1.18, 0.0, 2.7]
        planner.get_action(obs)
        self.assertEqual(planner.stage, "close_gripper")
        action = planner.get_action(obs)
        self.assertEqual(action["delta_gripper"], 1.0)

    def test_world_error_is_converted_to_tool_frame_delta(self):
        planner = CubeAutoGraspPlanner(
            CubeAutoGraspConfig(
                front_offset_world=[1.0, 0.0, 0.0],
                max_delta_xyz=[1.0, 1.0, 1.0],
                motion_scale=1.0,
                target_frame="ee",
            )
        )
        obs = {
            "ee_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "target_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        planner.set_reference_pose(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.5707963267948966]
        )

        action = planner.get_action(obs)

        self.assertAlmostEqual(action["delta_x"], 0.0, places=6)
        self.assertAlmostEqual(action["delta_y"], -1.0, places=6)
        self.assertAlmostEqual(action["delta_z"], 0.0, places=6)

    def test_default_target_frame_uses_bound_pinch_site(self):
        class FakeData:
            site_xpos = [[0.0, 0.0, 0.0]]

        class FakeBase:
            pinch_site_id = 0
            data = FakeData()

        planner = CubeAutoGraspPlanner(
            CubeAutoGraspConfig(
                front_offset_world=[1.0, 0.0, 0.0],
                max_delta_xyz=[1.0, 1.0, 1.0],
                motion_scale=1.0,
            )
        )
        planner.bind_env_base(FakeBase())
        obs = {
            "ee_pose": [10.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "target_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }

        action = planner.get_action(obs)

        self.assertAlmostEqual(action["delta_x"], 1.0)


if __name__ == "__main__":
    unittest.main()

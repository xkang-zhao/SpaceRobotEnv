import numpy as np

from lerobot_planner_teleop import PlannerTeleop, PlannerTeleopConfig
from mujoco_robot.robot_controller import RobotController


class FakeEnv:
    def __init__(self):
        self.success = False

    def is_success(self):
        return self.success


class FakeRobot:
    def __init__(self):
        self.env = FakeEnv()
        self.obs = {
            "ee_pose": np.array([1.2, 0.1, 2.82, 0.5, 0.5, 0.5, 0.5], dtype=float),
            "target_pose": np.array([1.2, 0.1, 2.7, 1.0, 0.0, 0.0, 0.0], dtype=float),
        }

    def get_planner_observation(self):
        return {
            "ee_pose": self.obs["ee_pose"].copy(),
            "target_pose": self.obs["target_pose"].copy(),
        }

    def is_episode_success(self):
        return self.env.is_success()


def test_planner_teleop_outputs_complete_action_dict():
    teleop = PlannerTeleop(PlannerTeleopConfig())
    teleop.bind_robot(FakeRobot())
    teleop.connect()

    action = teleop.get_action()

    assert set(action.keys()) == {
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw",
        "delta_gripper",
    }
    assert all(np.isfinite(value) for value in action.values())


def test_planner_teleop_gripper_closes_monotonically():
    teleop = PlannerTeleop(PlannerTeleopConfig(gripper_close_step=0.2, gripper_close_max=0.6))
    teleop.bind_robot(FakeRobot())
    teleop.connect()
    teleop.reset_episode()
    teleop._stage = "close_gripper"

    values = [teleop.get_action()["delta_gripper"] for _ in range(5)]

    assert values == sorted(values)
    assert values[-1] == 0.6
    assert max(values) <= teleop.config.gripper_close_max


def test_planner_teleop_uses_three_stage_front_approach():
    teleop = PlannerTeleop(
        PlannerTeleopConfig(
            front_offset_world=[-0.10, 0.0, 0.0],
            grasp_offset_world=[-0.02, 0.0, 0.0],
            front_position_tolerance=0.02,
            grasp_position_tolerance=0.02,
            orientation_tolerance=10.0,
        )
    )
    robot = FakeRobot()
    teleop.bind_robot(robot)
    teleop.connect()

    teleop.get_action()
    assert teleop._stage == "move_to_object_front"

    robot.obs["ee_pose"][:3] = np.array([1.1, 0.1, 2.7])
    teleop.get_action()
    assert teleop._stage == "approach_object"

    robot.obs["ee_pose"][:3] = np.array([1.18, 0.1, 2.7])
    teleop.get_action()
    assert teleop._stage == "close_gripper"


def test_planner_teleop_front_point_uses_world_offset_not_object_rotation():
    teleop = PlannerTeleop(PlannerTeleopConfig(front_offset_world=[-0.10, 0.0, 0.0]))
    target_pose = np.array([1.2, 0.1, 2.7, 0.70710678, 0.0, 0.0, 0.70710678], dtype=float)

    desired = teleop._target_position_from_world_offset(target_pose, np.array(teleop.config.front_offset_world))

    assert np.allclose(desired, np.array([1.1, 0.1, 2.7]))


def test_planner_teleop_motion_scale_reduces_cartesian_action_speed():
    teleop = PlannerTeleop(
        PlannerTeleopConfig(
            max_delta_xyz=[1.0, 1.0, 1.0],
            max_delta_rpy=[1.0, 1.0, 1.0],
            motion_scale=0.25,
        )
    )

    action = teleop._compute_action_to_target_step(
        reference_position=np.zeros(3, dtype=float),
        reference_rotation=np.eye(3, dtype=float),
        target_step={
            "position_world": np.array([2.0, -2.0, 0.5], dtype=float),
            "rotation_world": teleop._euler_to_rotation_matrix(np.array([1.0, 0.0, 0.0], dtype=float)),
            "gripper_command": 0.0,
        },
    )

    assert np.allclose(
        [action["delta_x"], action["delta_y"], action["delta_z"]],
        [0.25, -0.25, 0.125],
    )
    assert np.isclose(action["delta_roll"], 0.25)
    assert np.isclose(action["delta_pitch"], 0.0)
    assert np.isclose(action["delta_yaw"], 0.0)


def test_robot_controller_maps_normalized_gripper_commands_continuously():
    controller = RobotController(model=None, data=None)

    controller.update_target_gripper([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    halfway_target = controller.target_gripper
    for _ in range(20):
        controller.update_target_gripper(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )

    assert halfway_target == 12.5
    assert controller.target_gripper == 255.0

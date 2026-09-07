import numpy as np


def test_env_obs_preprocess_maps_state_and_depth_image():
    from scripts.evals.lerobot.lerobot_eval_common import CAMERA_NAME_MAP, env_obs_preprocess

    obs = {
        "base_pose": np.array([1.0, 2.0, 3.0, 0.7, 0.1, 0.2, 0.3]),
        "joint_pos": np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "third_left_camera": np.zeros((2, 3, 3), dtype=np.uint8),
        "left_wrist_camera": np.ones((2, 3, 3), dtype=np.uint8),
        "third_right_camera": np.full((2, 3, 3), 2, dtype=np.uint8),
        "left_wrist_camera_depth": np.full((2, 3), 0.5, dtype=np.float32),
    }

    policy_obs = env_obs_preprocess(obs.copy(), CAMERA_NAME_MAP)

    assert policy_obs["base_pos_x"] == 1.0
    assert policy_obs["base_quat_z"] == 0.3
    assert policy_obs["wrist_3_joint.pos"] == 0.5
    assert policy_obs["gripper_joint.pos"] == 0.0
    assert policy_obs["third_left_0_rgb"].shape == (2, 3, 3)
    assert policy_obs["left_wrist_0_depth"].shape == (2, 3, 3)
    np.testing.assert_allclose(policy_obs["left_wrist_0_depth"][0, 0], [0.5, 0.5, 0.5])


def test_action_dict_to_env_actions_accepts_scalar_and_chunk_values():
    from scripts.evals.lerobot.lerobot_eval_common import action_dict_to_env_actions

    scalar_action = {
        "delta_x": 1,
        "delta_y": 2,
        "delta_z": 3,
        "delta_roll": 4,
        "delta_pitch": 5,
        "delta_yaw": 6,
        "delta_gripper": 7,
    }
    scalar_result = action_dict_to_env_actions(scalar_action)
    assert scalar_result.shape == (1, 7)
    np.testing.assert_allclose(scalar_result[0], [1, 2, 3, 4, 5, 6, 7])

    chunk_action = {
        "delta_x": np.array([1, 10]),
        "delta_y": np.array([2, 20]),
        "delta_z": np.array([3, 30]),
        "delta_roll": np.array([4, 40]),
        "delta_pitch": np.array([5, 50]),
        "delta_yaw": np.array([6, 60]),
        "delta_gripper": np.array([7, 70]),
    }
    chunk_result = action_dict_to_env_actions(chunk_action)
    assert chunk_result.shape == (2, 7)
    np.testing.assert_allclose(chunk_result[1], [10, 20, 30, 40, 50, 60, 70])


def test_policy_output_to_env_actions_uses_make_robot_action_callback():
    from scripts.evals.lerobot.lerobot_eval_common import policy_output_to_env_actions

    def fake_make_robot_action(action, _dataset_features):
        assert action == "raw-policy-output"
        return {
            "delta_x": np.array([0.1, 0.2]),
            "delta_y": np.array([0.0, 0.0]),
            "delta_z": np.array([0.3, 0.4]),
            "delta_roll": np.array([0.0, 0.0]),
            "delta_pitch": np.array([0.0, 0.0]),
            "delta_yaw": np.array([0.0, 0.0]),
            "delta_gripper": np.array([1.0, -1.0]),
        }

    actions = policy_output_to_env_actions(
        "raw-policy-output",
        dataset_features={"action.delta_x": float},
        make_robot_action_func=fake_make_robot_action,
    )

    assert actions.dtype == np.float32
    assert actions.shape == (2, 7)
    np.testing.assert_allclose(actions[:, [0, 2, 6]], [[0.1, 0.3, 1.0], [0.2, 0.4, -1.0]])


def test_policy_output_to_env_actions_accepts_direct_action_chunk_array():
    from scripts.evals.lerobot.lerobot_eval_common import policy_output_to_env_actions

    raw_actions = np.array([[[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]]], dtype=np.float32)

    actions = policy_output_to_env_actions(
        raw_actions,
        dataset_features={},
        make_robot_action_func=lambda _action, _features: (_ for _ in ()).throw(AssertionError("unused")),
    )

    assert actions.shape == (2, 7)
    np.testing.assert_allclose(actions[0], [1, 2, 3, 4, 5, 6, 7])

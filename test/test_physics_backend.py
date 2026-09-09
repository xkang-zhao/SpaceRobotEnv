import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mujoco_robot.physics_backend import MujocoBackend, WarpBackend


class PhysicsBackendTests(unittest.TestCase):
    def test_contact_detection_uses_current_geom_pairs(self):
        from envs.space_ur10e_env import SpaceUR10eEnv
        base = SpaceUR10eEnv.__new__(SpaceUR10eEnv)
        base.target_geom_ids = {10}
        base.finger_A_pad_ids = {20}
        base.finger_B_pad_ids = {30}
        base.data = SimpleNamespace(ncon=2, contact=[
            SimpleNamespace(geom=np.array([10, 20]), geom1=0, geom2=0),
            SimpleNamespace(geom=np.array([30, 10]), geom1=0, geom2=0),
        ])
        self.assertEqual(base._check_gripper_target_contacts(), (True, True))
        base.data.contact[1].geom[:] = [30, 99]
        self.assertEqual(base._check_gripper_target_contacts(), (True, False))

    def test_cpu_backend_matches_native_steps(self):
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
            '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        actual, expected = mujoco.MjData(model), mujoco.MjData(model)
        backend = MujocoBackend(model, actual)
        backend.reset()
        backend.step(100)
        for _ in range(100):
            mujoco.mj_step(model, expected)
        np.testing.assert_array_equal(actual.qpos, expected.qpos)
        np.testing.assert_array_equal(actual.qvel, expected.qvel)
        self.assertEqual(actual.time, expected.time)

    def test_missing_warp_dependency_has_actionable_error(self):
        with patch.dict(sys.modules, {"mujoco_warp": None}):
            with self.assertRaisesRegex(ImportError, r"cube.xml.*optional dependency"):
                WarpBackend(None, None, device="cuda:0", scene_path="cube.xml")

    def test_invalid_backend_rejected_before_loading_scene(self):
        from envs.space_ur10e_env import SpaceUR10eEnv
        with self.assertRaisesRegex(ValueError, "physics_backend"):
            SpaceUR10eEnv(physics_backend="invalid", scene_path="missing.xml")

    def test_cpu_reset_clears_gripper_control(self):
        import gymnasium as gym
        import envs  # noqa: F401
        env = gym.make("SpaceUR10e-Cube-v0", observation_mode="state")
        try:
            env.reset(seed=7)
            env.step(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
            self.assertGreater(env.unwrapped.data.ctrl[6], 0)
            env.reset(seed=7)
            self.assertEqual(env.unwrapped.controller.target_gripper, 0)
            self.assertEqual(env.unwrapped.data.ctrl[6], 0)
        finally:
            env.close()


@unittest.skipUnless(os.environ.get("RUN_WARP_TESTS") == "1",
                     "Set RUN_WARP_TESTS=1 with optional Warp dependencies and CUDA")
class WarpIntegrationTests(unittest.TestCase):
    def test_cube_grasp_completes_with_physical_contacts(self):
        from mujoco_robot.physics_benchmark import benchmark_cube
        result = benchmark_cube("warp", seed=7, max_steps=150)
        self.assertTrue(result["success"], result)
        self.assertTrue(result["left_contact"])
        self.assertTrue(result["right_contact"])
        self.assertEqual(result["success_counter"], 10)

    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import envs  # noqa: F401
        cls.env = gym.make("SpaceUR10e-Cube-v0", observation_mode="state",
                           physics_backend="warp")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reset_replays_actions_without_stale_graph_or_gripper(self):
        base = self.env.unwrapped
        trajectories = []
        targets = []
        for _ in range(2):
            obs, _ = self.env.reset(seed=7)
            targets.append(base._get_target_pose().copy())
            self.assertEqual(base.data.time, 0)
            self.assertEqual(base.data.ctrl[6], 0)
            self.assertIsNone(base.sensor)
            states = []
            for grip in (1., 1., -1.):
                obs, reward, terminated, truncated, info = self.env.step(
                    np.array([0, 0, 0, 0, 0, 0, grip], dtype=np.float32))
                self.assertTrue(self.env.observation_space.contains(obs))
                self.assertFalse(terminated)
                self.assertFalse(truncated)
                self.assertEqual(reward, 0)
                states.append(base.data.qpos.copy())
            self.assertAlmostEqual(base.data.time, 0.15, places=5)
            self.assertGreater(obs["gripper_pos"][0], 0)
            trajectories.append(states)
        np.testing.assert_allclose(trajectories[0], trajectories[1], atol=1e-5, rtol=0)
        np.testing.assert_array_equal(targets[0], targets[1])
        self.env.reset(seed=8)
        self.assertFalse(np.allclose(targets[0], base._get_target_pose()))

    def test_cpu_gpu_short_trajectory_agree(self):
        import gymnasium as gym
        cpu = gym.make("SpaceUR10e-Cube-v0", observation_mode="state")
        try:
            cpu.reset(seed=7)
            self.env.reset(seed=7)
            for grip in (0., 1., 1.):
                action = np.array([0, 0, 0, 0, 0, 0, grip], dtype=np.float32)
                cpu.step(action)
                self.env.step(action)
                np.testing.assert_allclose(cpu.unwrapped.data.qpos,
                                           self.env.unwrapped.data.qpos,
                                           atol=2e-3, rtol=0)
        finally:
            cpu.close()

    def test_success_requires_ten_consecutive_control_steps(self):
        base = self.env.unwrapped
        self.env.reset(seed=7)
        original = base.compute_reward

        outcomes = iter([True] * 9 + [False] + [True] * 10)

        def success_reward(obs, action):
            reward, info = original(obs, action)
            info["is_success"] = next(outcomes)
            return reward, info

        # Isolate the confirmation contract while still running real GPU steps.
        with patch.object(base, "compute_reward", side_effect=success_reward):
            for count in list(range(1, 10)) + [0] + list(range(1, 11)):
                _, reward, terminated, _, info = self.env.step(np.zeros(7))
                self.assertEqual(info["success_counter"], count)
                self.assertEqual(terminated, count == 10)
                self.assertEqual(info["is_success"], count == 10)
                self.assertEqual(reward, float(count == 10))
        self.env.reset(seed=7)
        self.assertEqual(base._success_counter, 0)

    def test_overflow_is_an_error_and_reset_clears_it(self):
        base = self.env.unwrapped
        self.env.reset(seed=7)
        base._physics.warp_data.overflow.assign(np.array([1], dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, "capacity overflow"):
            self.env.step(np.zeros(7))
        self.env.reset(seed=7)
        self.env.step(np.zeros(7))


if __name__ == "__main__":
    unittest.main()

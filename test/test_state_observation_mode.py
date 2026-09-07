import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class StateObservationModeTests(unittest.TestCase):
    def test_state_mode_skips_renderer_and_reports_gripper_position(self):
        import gymnasium as gym
        import envs  # noqa: F401

        env = gym.make(
            "SpaceUR10e-Cube-v0",
            observation_mode="state",
            render_mode=None,
        )
        try:
            obs, _ = env.reset(seed=7)

            self.assertIsNone(env.unwrapped.sensor)
            self.assertEqual(
                set(obs),
                {"joint_pos", "base_pose", "ee_pose", "gripper_pos"},
            )
            self.assertEqual(obs["gripper_pos"].shape, (1,))
            self.assertEqual(obs["gripper_pos"].dtype, np.float32)
            np.testing.assert_allclose(obs["gripper_pos"], [0.0])
            self.assertTrue(env.observation_space.contains(obs))

            close_action = np.zeros(7, dtype=np.float32)
            close_action[-1] = 1.0
            next_obs, _, _, _, _ = env.step(close_action)
            self.assertEqual(set(next_obs), set(obs))
            self.assertTrue(env.observation_space.contains(next_obs))
            self.assertGreater(next_obs["gripper_pos"][0], 0.0)
            self.assertLess(next_obs["gripper_pos"][0], 1.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

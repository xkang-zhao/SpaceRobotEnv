"""CPU regression for upper-rod grasp calibration (MuJoCo + Pinocchio)."""
import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import gymnasium as gym
import envs  # noqa: F401
from planners.auto_grasp.tasks import AutoGraspSession, get_task_spec


class UpperRodSmokeTests(unittest.TestCase):
    def test_calibration_confirms_grasp_on_regression_seeds(self):
        task = get_task_spec('satellite3_upper_rod')
        env = gym.make(task.env_id, observation_mode='state', render_mode=None,
                       physics_backend='mujoco')
        try:
            for seed in (4, 5, 7):
                with self.subTest(seed=seed):
                    session = AutoGraspSession(task.name, env)
                    obs = session.reset(seed)
                    consecutive = 0
                    confirmed = False
                    with contextlib.redirect_stdout(io.StringIO()):
                        for _ in range(session.max_steps):
                            action = session.compute_action(obs)
                            obs, reward, terminated, truncated, info = env.step(action)
                            raw_success = env.unwrapped.is_success()
                            consecutive = consecutive + 1 if raw_success else 0
                            session.update(obs, reward, terminated, truncated, info)
                            confirmed = bool(terminated and info.get('is_success'))
                            if session.is_done:
                                break
                    self.assertTrue(confirmed, f'{task.name} seed={seed}: {session.failure_reason}')
                    self.assertFalse(truncated)
                    self.assertGreaterEqual(consecutive, 10)
                    self.assertEqual(info['success_counter'], 10)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@unittest.skipUnless(os.environ.get('RUN_WARP_TESTS') == '1', 'Requires optional Warp and CUDA')
class WarpVectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from envs.space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv
        cls.env = SpaceUR10eWarpVectorEnv(4)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_seed_expansion_matches_cpu_and_actions_are_independent(self):
        import gymnasium as gym
        env = self.env
        obs, _ = env.reset(seed=7)
        teacher = env.get_planner_state()
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(set(obs), {'joint_pos', 'base_pose', 'ee_pose', 'gripper_pos'})
        actions = np.zeros((4, 7), dtype=np.float32)
        actions[0, 6] = 1
        actions[1, 0] = 0.001
        next_obs, rewards, term, trunc, info = env.step(actions)
        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(term.dtype, np.bool_)
        self.assertFalse(term.any() or trunc.any())
        self.assertGreater(next_obs['gripper_pos'][0, 0], next_obs['gripper_pos'][1, 0])
        for w in (0, 1):
            cpu = gym.make('SpaceUR10e-Cube-v0', observation_mode='state')
            try:
                cpu_obs, _ = cpu.reset(seed=7 + w)
                np.testing.assert_allclose(teacher['target_pose'][w], cpu.unwrapped._get_target_pose(), atol=3e-7)
                np.testing.assert_allclose(obs['ee_pose'][w], cpu_obs['ee_pose'], atol=3e-6)
                cpu_obs, r, t, tr, cpu_info = cpu.step(actions[w])
                np.testing.assert_allclose(next_obs['joint_pos'][w], cpu_obs['joint_pos'], atol=2e-3)
                self.assertAlmostEqual(info['distance'][w], cpu_info['distance'], places=4)
                self.assertAlmostEqual(info['shaped_reward'][w], cpu_info['shaped_reward'], places=3)
            finally:
                cpu.close()

    def test_partial_reset_preserves_other_worlds_and_graph(self):
        env = self.env
        env.reset(seed=7)
        actions = np.zeros((4, 7), dtype=np.float32)
        actions[:, 6] = 1
        obs, *_ = env.step(actions)
        before = {name: getattr(env.d, name).numpy() for name in ('qpos', 'qvel', 'time', 'ctrl', 'qacc_warmstart')}
        graph = env._graph
        env._counts.assign(np.array([4, 3, 2, 1], dtype=np.int32))
        mask = np.array([False, True, False, False])
        reset_obs, _ = env.reset(seed=99, options={'reset_mask': mask})
        self.assertIs(env._graph, graph)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(env.d, name).numpy()[~mask], value[~mask])
        for key in obs:
            np.testing.assert_array_equal(reset_obs[key][~mask], obs[key][~mask])
        np.testing.assert_array_equal(env._counts.numpy(), [4, 0, 2, 1])
        self.assertEqual(env.d.time.numpy()[1], 0)
        self.assertEqual(env.d.ctrl.numpy()[1, 6], 0)
        env.step(np.zeros((4, 7)))
        np.testing.assert_allclose(env.d.time.numpy(), [0.1, 0.05, 0.1, 0.1], atol=2e-6)

    def test_reset_streams_are_independent_and_reproducible(self):
        env = self.env
        selected = np.array([False, True, False, False])
        other = np.array([False, False, True, False])
        env.reset(seed=7)
        initial = env.get_planner_state()['target_pose'][1]
        env.reset(options={'reset_mask': selected})
        expected = env.get_planner_state()['target_pose'][1]
        self.assertFalse(np.array_equal(initial, expected))
        env.reset(seed=7)
        env.reset(options={'reset_mask': other})
        env.reset(options={'reset_mask': other})
        env.reset(options={'reset_mask': selected})
        np.testing.assert_array_equal(env.get_planner_state()['target_pose'][1], expected)

    def test_gpu_contact_confirmation_and_failure_are_per_world(self):
        env = self.env
        env.reset(seed=7)
        d, wp = env.d, env.wp
        base = env._template
        target = next(iter(base.target_geom_ids))
        a, b = next(iter(base.finger_A_pad_ids)), next(iter(base.finger_B_pad_ids))
        pairs = np.zeros((d.naconmax, 2), dtype=np.int32)
        worlds = np.zeros(d.naconmax, dtype=np.int32)
        pairs[:5] = [[target, a], [b, target], [a, target], [target, b], [target, a]]
        worlds[:5] = [0, 0, 1, 1, 2]
        with wp.ScopedDevice(env.device):
            d.contact.geom.assign(pairs)
            d.contact.worldid.assign(worlds)
            d.nacon.assign(np.array([5], dtype=np.int32))
            ctrl = d.ctrl.numpy()
            ctrl[:, 6] = 255
            d.ctrl.assign(ctrl)
            d.qvel.zero_()
            for _ in range(9):
                env._observe(update=True)
            np.testing.assert_array_equal(env._counts.numpy(), [9, 9, 0, 0])
            self.assertFalse(env._terminated.numpy().any())
            # A transient loss of stability interrupts world 1's streak only.
            velocity = d.qvel.numpy()
            velocity[1, base.target_dof_adr] = 1
            d.qvel.assign(velocity)
            env._observe(update=True)
            np.testing.assert_array_equal(env._counts.numpy(), [10, 0, 0, 0])
            np.testing.assert_array_equal(env._terminated.numpy(), [1, 0, 0, 0])
            poses = d.xpos.numpy()
            poses[3, base.target_body_id, 0] += 5
            d.xpos.assign(poses)
            env._observe(update=False)
            np.testing.assert_array_equal(env._truncated.numpy(), [0, 0, 0, 1])

    def test_invalid_inputs_and_done_world_require_reset(self):
        env = self.env
        env.reset(seed=7)
        for actions in (np.zeros(7), np.full((4, 7), np.nan), np.full((4, 7), 2)):
            with self.assertRaises(ValueError):
                env.step(actions)
        for mask in (np.array([True]), np.zeros(4, dtype=bool), np.ones(4, dtype=int)):
            with self.assertRaises(ValueError):
                env.reset(options={'reset_mask': mask})
        with self.assertRaises(ValueError):
            env.reset(seed=[1, 2])
        env._needs_reset[2] = True
        with self.assertRaisesRegex(RuntimeError, 'Reset required'):
            env.step(np.zeros((4, 7)))
        env.reset(seed=10, options={'reset_mask': np.array([False, False, True, False])})
        env.step(np.zeros((4, 7)))

    def test_overflow_is_not_accepted(self):
        env = self.env
        env.reset(seed=7)
        env.d.overflow.assign(np.array([0, 1, 0, 0], dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, 'worlds.*1'):
            env._check_state()
        env.reset(options={'reset_mask': np.array([False, True, False, False])})
        env.step(np.zeros((4, 7)))

    def test_ik_failure_requires_reset_of_partially_updated_controllers(self):
        env = self.env
        env.reset(seed=7)
        original = env._template._compute_ik_control
        calls = 0

        def fail_second_world(target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError('injected IK failure')
            return original(target)

        with patch.object(env._template, '_compute_ik_control', side_effect=fail_second_world):
            with self.assertRaisesRegex(RuntimeError, 'world 1'):
                env.step(np.zeros((4, 7)))
        self.assertTrue(env._needs_reset.all())
        env.reset(seed=7)
        env.step(np.zeros((4, 7)))


if __name__ == '__main__':
    unittest.main()

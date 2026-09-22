import os
import subprocess
import sys
import textwrap
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
                cpu_obs, _, _, _, cpu_info = cpu.step(actions[w])
                np.testing.assert_allclose(next_obs['joint_pos'][w], cpu_obs['joint_pos'], atol=2e-3)
                self.assertAlmostEqual(info['distance'][w], cpu_info['distance'], places=4)
                self.assertAlmostEqual(info['shaped_reward'][w], cpu_info['shaped_reward'], places=3)
            finally:
                cpu.close()

    def test_small_graph_matches_full_graph_and_control_period(self):
        from envs.space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv
        small = SpaceUR10eWarpVectorEnv(4, physics_substeps_per_graph=10)
        try:
            self.env.reset(seed=7)
            small.reset(seed=7)
            action = np.zeros((4, 7), dtype=np.float32)
            action[:, 6] = 1
            for _ in range(4):
                self.env.step(action)
                small.step(action)
                for name in ('qpos', 'qvel', 'ctrl', 'time'):
                    np.testing.assert_allclose(
                        getattr(small.d, name).numpy(), getattr(self.env.d, name).numpy(),
                        atol=2e-6, rtol=1e-5,
                    )
                np.testing.assert_array_equal(small._counts.numpy(), self.env._counts.numpy())
            np.testing.assert_allclose(small.d.time.numpy(), 0.2, atol=2e-6)
            small.reset(seed=8, options={'reset_mask': np.array([True, False, False, False])})
            small.step(action, active_mask=np.array([True, False, True, True]))
            np.testing.assert_allclose(small.d.time.numpy(), [0.05, 0.2, 0.25, 0.25], atol=2e-6)
        finally:
            small.close()

    def test_invalid_graph_sizes_rejected(self):
        from envs.space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv
        for size in (0, -1, 3, 101, True, 10.5):
            with self.assertRaisesRegex(ValueError, 'positive divisor of 100'):
                SpaceUR10eWarpVectorEnv(physics_substeps_per_graph=size)

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

    def test_paused_world_preserves_integrator_and_observation(self):
        env = self.env
        env.reset(seed=7)
        actions = np.zeros((4, 7), dtype=np.float32)
        obs, *_ = env.step(actions)
        names = ('qpos', 'qvel', 'act', 'time', 'ctrl', 'qacc_warmstart')
        before = {name: getattr(env.d, name).numpy().copy() for name in names}
        env._needs_reset[1] = True
        env._counts.assign(np.array([0, 10, 0, 0], dtype=np.int32))
        active = np.array([True, False, True, True])
        actions[1] = 0.5  # Ignored for a paused world, including its controller.
        result, reward, *_ = env.step(actions, active_mask=active)
        for name in names:
            np.testing.assert_array_equal(getattr(env.d, name).numpy()[1], before[name][1])
        for key in obs:
            np.testing.assert_array_equal(result[key][1], obs[key][1])
        self.assertEqual(reward[1], 0)
        self.assertEqual(env._counts.numpy()[1], 10)
        self.assertTrue(env._needs_reset[1])
        self.assertGreater(env.d.time.numpy()[0], before['time'][0])
        env.reset(options={'reset_mask': ~active})
        env.step(np.zeros((4, 7)))

    def test_active_observation_is_independent_of_neighbor_pause(self):
        env = self.env
        actions = np.zeros((4, 7), dtype=np.float32)
        actions[:, 0] = 0.01
        env.reset(seed=7)
        reference = [env.step(actions)[0] for _ in range(3)]
        env.reset(seed=7)
        for expected in reference:
            actual = env.step(actions, active_mask=np.array([True, False, True, True]))[0]
            for key in actual:
                np.testing.assert_array_equal(actual[key][0], expected[key][0],
                                              err_msg=key)

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

        with patch.object(
            env._template, '_compute_ik_control', side_effect=fail_second_world
        ), self.assertRaisesRegex(RuntimeError, 'world 1'):
            env.step(np.zeros((4, 7)))
        self.assertTrue(env._needs_reset.all())
        env.reset(seed=7)
        env.step(np.zeros((4, 7)))


@unittest.skipUnless(
    os.environ.get('RUN_WARP_TESTS') == '1', 'Requires optional Warp and CUDA'
)
class WarpDeterministicPhysicsTests(unittest.TestCase):
    def test_fresh_rollouts_are_bit_exact_in_isolated_process(self):
        """Strict mode is process-global, so isolate it from regular-mode tests."""
        code = textwrap.dedent(
            """
            import numpy as np
            from envs.space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv

            def rollout():
                env = SpaceUR10eWarpVectorEnv(
                    2, deterministic_physics=True, physics_substeps_per_graph=10
                )
                try:
                    env.reset(seed=71)
                    action = np.zeros((2, 7), dtype=np.float32)
                    action[0, 0] = 0.01
                    action[1, 1] = -0.01
                    states = []
                    for _ in range(3):
                        env.step(action)
                        states.append(env.d.qpos.numpy().copy())
                    return np.stack(states)
                finally:
                    env.close()

            np.testing.assert_array_equal(rollout(), rollout())
            """
        )
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root / 'src')
        subprocess.run(
            [sys.executable, '-c', code],
            check=True,
            env=env,
            text=True,
            timeout=600,
        )


if __name__ == '__main__':
    unittest.main()

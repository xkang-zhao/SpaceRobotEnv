import os
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mujoco_robot.batch_physics_benchmark import ControlReplay, compare_states, replay_cpu, replay_warp_batch


def sample_replay():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><option timestep="0.0005" gravity="0 0 0"/>
      <worldbody><body><joint name="hinge" damping="0.1"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0 0"/>
      </body></worldbody><actuator><motor joint="hinge"/></actuator>
    </mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[0] = 0.2
    mujoco.mj_forward(model, data)
    return ControlReplay(model, data, np.array([[1.], [-1.], [0.5]]), substeps=20)


class BatchBenchmarkTests(unittest.TestCase):
    def test_replay_reset_does_not_include_warmup(self):
        trace = sample_replay()
        _, qpos, qvel = replay_cpu(trace)
        data = mujoco.MjData(trace.model)
        mujoco.mj_copyData(data, trace.model, trace.initial)
        for index, ctrl in enumerate(trace.controls):
            data.ctrl[:] = ctrl
            mujoco.mj_step(trace.model, data, nstep=trace.substeps)
            np.testing.assert_array_equal(qpos[index], data.qpos)
            np.testing.assert_array_equal(qvel[index], data.qvel)
        self.assertEqual(trace.initial.time, 0)
        self.assertEqual(trace.initial.qpos[0], 0.2)

    def test_invalid_trace_rejected(self):
        trace = sample_replay()
        for controls in (np.empty((0, 1)), np.zeros((2, 2)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                ControlReplay(trace.model, trace.initial, controls)

    def test_comparison_checks_every_world(self):
        result = compare_states(np.array([[1.], [1.], [1.5]]), np.zeros((3, 1)), [1.], [0.])
        self.assertEqual(result['max_qpos_abs_error'], 0.5)
        self.assertEqual(result['max_world_qpos_spread'], 0.5)
        self.assertTrue(result['finite'])
        result = compare_states(np.array([[np.nan]]), np.zeros((1, 1)), [1.], [0.])
        self.assertFalse(result['finite'])

    @unittest.skipUnless(os.environ.get('RUN_WARP_TESTS') == '1', 'Requires optional Warp and CUDA')
    def test_batched_control_replay_matches_cpu_after_warmup(self):
        trace = sample_replay()
        _, qpos, qvel = replay_cpu(trace)
        result = replay_warp_batch(trace, 4, qpos, qvel, nconmax=16, njmax=32,
                                   qpos_atol=1e-5, qvel_atol=1e-4)
        self.assertTrue(result['state_within_tolerance'], result)
        self.assertFalse(result['overflow'])
        self.assertLess(result['state_errors']['max_world_qpos_spread'], 1e-6)
        self.assertGreater(result['physics_steps_per_second'], 0)


if __name__ == '__main__':
    unittest.main()

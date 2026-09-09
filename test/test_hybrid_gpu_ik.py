"""Actual CUDA checks for optional vector-environment GPU IK integration."""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

@unittest.skipUnless(os.environ.get('RUN_WARP_TESTS')=='1','Requires CUDA')
class HybridIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from envs.space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv
        cls.cpu=SpaceUR10eWarpVectorEnv(4)
        cls.gpu=SpaceUR10eWarpVectorEnv(4,ik_backend='gpu')

    @classmethod
    def tearDownClass(cls):
        cls.cpu.close();cls.gpu.close()

    def reset(self):
        self.cpu.reset(seed=7);self.gpu.reset(seed=7)
        self.gpu.ik_backend='gpu'

    def test_rejected_candidate_uses_original_cpu_state(self):
        self.reset()
        action=np.zeros((4,7),dtype=np.float32);action[:,0]=.001
        self.cpu.step(action)
        original=self.gpu._gpu_ik.solve
        def reject(q,targets):
            candidate,accepted,residual=original(q,targets)
            candidate[1]=np.nan;accepted[1]=False;residual[1]=np.inf
            return candidate,accepted,residual
        with patch.object(self.gpu._gpu_ik,'solve',side_effect=reject):
            obs,_,_,_,info=self.gpu.step(action)
        self.assertTrue(info['ik_cpu_fallback'][1])
        self.assertTrue(info['ik_gpu_accepted'][0])
        np.testing.assert_array_equal(self.gpu._control[1],self.cpu._control[1])
        self.assertTrue(np.isfinite(self.gpu._control).all())
        self.assertTrue(self.gpu.observation_space.contains(obs))

    def test_shadow_preserves_cpu_controls_and_physics(self):
        self.reset();self.gpu.ik_backend='shadow'
        action=np.zeros((4,7),dtype=np.float32);action[:,1]=.001
        cpu=self.cpu.step(action);shadow=self.gpu.step(action)
        np.testing.assert_array_equal(self.gpu._control,self.cpu._control)
        for key in cpu[0]:np.testing.assert_allclose(cpu[0][key],shadow[0][key],atol=1e-7)
        self.assertFalse(shadow[4]['ik_cpu_fallback'].any())
        self.assertLess(shadow[4]['ik_shadow_joint_max_abs'].max(),.001)

    def test_partial_reset_retains_graph_and_other_worlds(self):
        self.reset();self.gpu.step(np.zeros((4,7),dtype=np.float32))
        before=self.gpu._qpos.copy();graph=self.gpu._gpu_ik._graph
        self.gpu.reset(seed=17,options={'reset_mask':np.array([True,False,False,False])})
        np.testing.assert_array_equal(self.gpu._qpos[1:],before[1:])
        self.gpu.step(np.zeros((4,7),dtype=np.float32))
        self.assertIs(self.gpu._gpu_ik._graph,graph)
        self.assertTrue(np.isfinite(self.gpu._control).all())

    def test_solver_exception_requires_reset(self):
        self.reset()
        with patch.object(self.gpu._gpu_ik,'solve',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'batch IK failed'):
                self.gpu.step(np.zeros((4,7),dtype=np.float32))
        self.assertTrue(self.gpu._needs_reset.all())
        with self.assertRaisesRegex(RuntimeError,'Reset required'):
            self.gpu.step(np.zeros((4,7),dtype=np.float32))

if __name__=='__main__':unittest.main()

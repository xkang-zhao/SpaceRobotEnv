"""Optional real-CUDA regressions for the isolated floating-base IK prototype."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(os.environ.get('RUN_WARP_TESTS')=='1', 'requires CUDA; set RUN_WARP_TESTS=1')
class GPUPrototypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mujoco_robot.robot_ik import Kinematics
        cls.kin=Kinematics('mjcf/arm.xml','left_attachment')

    def test_matrices_jacobians_integration_and_momentum_against_pinocchio(self):
        from reports.analysis.validate_gpu_ik import validate
        result=validate(self.kin)
        self.assertEqual(result['samples'],96)
        self.assertEqual(result['groups']['unreachable']['accepted'],0)
        self.assertEqual(result['groups']['aligned']['accepted'],8)

    def test_independent_rows_quaternion_sign_and_replay(self):
        import numpy as np
        import pinocchio as pin
        import warp as wp
        from mujoco_robot.gpu_ik_prototype import GPUChainIK
        q=pin.neutral(self.kin.model)
        q[7:]=[.1,-.3,.4,.2,.1,-.2]
        frame=self.kin.model.getFrameId(self.kin.frame_name)
        pin.framesForwardKinematics(self.kin.model,self.kin.data,q)
        pose=pin.SE3(self.kin.data.oMf[frame])
        target=pin.SE3(pose.rotation,pose.translation+[0,.002,0])
        inputs=np.repeat(q[None],3,axis=0);inputs[1,3:7]*=-1
        gpu=GPUChainIK(self.kin,3)
        targets=[target,target,pose]
        gpu.upload(inputs,targets);graph=gpu.prepare()
        gpu.upload(inputs,targets);wp.capture_launch(graph)
        result,residual,accepted=gpu.read()
        self.assertTrue(accepted.all())
        np.testing.assert_array_equal(result[2],inputs[2])
        np.testing.assert_allclose(result[0,7:],result[1,7:],atol=1e-12)
        np.testing.assert_allclose(result[0,3:7],-result[1,3:7],atol=1e-12)
        gpu.upload(inputs,targets);wp.capture_launch(graph)
        again,_,_=gpu.read()
        np.testing.assert_array_equal(again,result)

    def test_reject_invalid_inputs_and_unsupported_chain(self):
        import numpy as np
        import pinocchio as pin
        from mujoco_robot.gpu_ik_prototype import GPUChainIK
        with self.assertRaises(ValueError):GPUChainIK(self.kin,0)
        gpu=GPUChainIK(self.kin,1)
        q=pin.neutral(self.kin.model)[None]
        with self.assertRaises(ValueError):gpu.upload(q[:,:12],[pin.SE3.Identity()])
        bad=q.copy();bad[0,6]=2.
        with self.assertRaisesRegex(ValueError,'quaternion'):gpu.upload(bad,[pin.SE3.Identity()])
        with self.assertRaises(ValueError):gpu.prepare(iterations=0)
        with self.assertRaises(ValueError):gpu.prepare(damping=0)


if __name__=='__main__':unittest.main()

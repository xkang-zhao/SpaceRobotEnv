"""Numerical regressions for the floating-base IK hot path."""
import unittest
from unittest.mock import patch

import numpy as np
import pinocchio as pin

from mujoco_robot.robot_ik import Kinematics


class ReferenceKinematics(Kinematics):
    """Original inverse-based update, retained as a regression oracle."""

    def step_ik(self, q, target_pose, frame_id, damping=0.1, dt=0.05):
        pin.framesForwardKinematics(self.model, self.data, q)
        pose = self.data.oMf[frame_id]
        error = np.r_[target_pose.translation - pose.translation,
                      pose.rotation @ pin.log3(pose.rotation.T @ target_pose.rotation)]
        jac, inverse, mass = self.compute_matrices(q, frame_id)
        arm = np.linalg.inv(jac.T @ jac + damping**2 * np.eye(6)) @ (jac.T @ error)
        velocity = np.r_[-inverse @ mass @ arm, arm]
        length = np.linalg.norm(velocity)
        if length > 0.2:
            velocity *= 0.2 / length
        return pin.integrate(self.model, q, velocity * dt), np.linalg.norm(error)


class TestIKOptimization(unittest.TestCase):
    def setUp(self):
        self.kin = Kinematics('mjcf/arm.xml', 'left_attachment')
        self.reference = ReferenceKinematics('mjcf/arm.xml', 'left_attachment')
        self.frame = self.kin.model.getFrameId(self.kin.frame_name)

    def cases(self):
        rng = np.random.default_rng(42)
        for i in range(12):
            q = pin.integrate(self.kin.model, pin.neutral(self.kin.model),
                              rng.normal(0, 0.5, self.kin.model.nv))
            if i == 0:
                q = pin.neutral(self.kin.model)
            pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
            pose = self.kin.data.oMf[self.frame]
            yield q, pin.SE3(pose.rotation @ pin.exp3(rng.normal(0, 0.1, 3)),
                            pose.translation + rng.normal(0, 0.03, 3))

    def test_step_matches_original_and_preserves_momentum(self):
        for q, target in self.cases():
            with self.subTest(q=q.tolist()):
                expected, err = self.reference.step_ik(q, target, self.frame)
                actual, actual_err = self.kin.step_ik(q, target, self.frame)
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
                self.assertAlmostEqual(actual_err, err, places=12)
                velocity = pin.difference(self.kin.model, q, actual)
                pin.crba(self.kin.model, self.kin.data, q)
                np.testing.assert_allclose(self.kin.data.M[:6] @ velocity, 0, atol=1e-9)

    def test_full_trajectory_matches_original(self):
        for q, target in self.cases():
            expected, err = self.reference.ik(q, target, steps=20, early_stop=False)
            actual, actual_err = self.kin.ik(q, target, steps=20, early_stop=False)
            np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
            self.assertAlmostEqual(actual_err, err, places=10)

    def test_aligned_target_requires_no_iterations(self):
        q = pin.neutral(self.kin.model)
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        target = pin.SE3(self.kin.data.oMf[self.frame])
        with patch.object(self.kin, 'step_ik', wraps=self.kin.step_ik) as step:
            result, error = self.kin.ik(q, target, steps=20)
        self.assertEqual(step.call_count, 0)
        np.testing.assert_array_equal(result, q)
        self.assertLess(error, 1e-12)
        self.assertIsNot(result, q)

    def test_early_stop_checks_final_target_and_returned_configuration(self):
        q = pin.neutral(self.kin.model)
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        pose = pin.SE3(self.kin.data.oMf[self.frame])
        target = pin.SE3(pose.rotation, pose.translation + [0, 0.002, 0])
        with patch.object(self.kin, 'step_ik', wraps=self.kin.step_ik) as step:
            result, error = self.kin.ik(q, target, steps=20)
        self.assertGreater(step.call_count, 0)
        self.assertLess(step.call_count, 200)
        pin.framesForwardKinematics(self.kin.model, self.kin.data, result)
        actual = self.kin.data.oMf[self.frame]
        residual = np.linalg.norm(np.r_[target.translation - actual.translation,
                                       pin.log3(actual.rotation.T @ target.rotation)])
        self.assertLessEqual(residual, 1e-4)
        self.assertAlmostEqual(error, residual, places=12)

    def test_unreachable_target_keeps_original_budget_and_trajectory(self):
        q = pin.neutral(self.kin.model)
        target = pin.SE3(np.eye(3), np.array([100., 100., 100.]))
        expected, err = self.kin.ik(q, target, steps=20, early_stop=False)
        with patch.object(self.kin, 'step_ik', wraps=self.kin.step_ik) as step:
            result, error = self.kin.ik(q, target, steps=20)
        self.assertEqual(step.call_count, 250)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        self.assertAlmostEqual(error, err, places=12)

    def test_residual_check_does_not_mutate_interpolation_pose(self):
        q = pin.neutral(self.kin.model)
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        pose = self.kin.data.oMf[self.frame]
        before = pin.SE3(pose)
        changed = q.copy()
        changed[7] += 0.5
        self.kin._target_error(changed, before, self.frame)
        np.testing.assert_array_equal(pose.homogeneous, before.homogeneous)

    def test_invalid_interpolation_budget(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.kin.ik(pin.neutral(self.kin.model), pin.SE3.Identity(), steps=0)

    def test_direct_path_accuracy_and_iteration_budget(self):
        rng = np.random.default_rng(15)
        accepted = 0
        for _ in range(12):
            q = pin.integrate(self.kin.model, pin.neutral(self.kin.model),
                              rng.normal(0, 0.3, self.kin.model.nv))
            pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
            pose = pin.SE3(self.kin.data.oMf[self.frame])
            target = pin.SE3(pose.rotation @ pin.exp3(rng.normal(0, 0.001, 3)),
                            pose.translation + rng.normal(0, 0.001, 3))
            before = q.copy()
            with patch.object(self.kin, 'step_ik', wraps=self.kin.step_ik) as step:
                result = self.kin._try_direct_ik(q, target, self.frame)
            self.assertLessEqual(step.call_count, 40)
            np.testing.assert_array_equal(q, before)
            if result is not None:
                accepted += 1
                actual, error = result
                self.assertLessEqual(error, 1e-4)
                self.assertAlmostEqual(error, self.kin._target_error(actual, target, self.frame), places=12)
        self.assertGreater(accepted, 0)

    def test_direct_path_stall_and_nonfinite_restart_from_input(self):
        q = pin.neutral(self.kin.model)
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        pose = pin.SE3(self.kin.data.oMf[self.frame])
        target = pin.SE3(pose.rotation, pose.translation + [0, 0.002, 0])
        original_step = self.kin.step_ik
        expected, error = self.kin.ik(q, target, fast_path=False)
        for bad in (q.copy(), np.full_like(q, np.nan)):
            def stalled(config, goal, frame, damping=0.1, dt=0.05):
                if dt == 0.2:
                    return bad.copy(), 1.0
                return original_step(config, goal, frame, damping=damping, dt=dt)
            with patch.object(self.kin, 'step_ik', side_effect=stalled):
                actual, actual_error = self.kin.ik(q, target)
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            self.assertAlmostEqual(actual_error, error, places=12)

    def test_direct_path_failure_preserves_full_fallback_budget(self):
        q = pin.neutral(self.kin.model)
        pin.framesForwardKinematics(self.kin.model, self.kin.data, q)
        pose = pin.SE3(self.kin.data.oMf[self.frame])
        target = pin.SE3(pose.rotation, pose.translation + [0, 0.002, 0])
        with patch.object(self.kin, '_target_error', side_effect=[.002, .0019, .0018, .0017, .0016]):
            with patch.object(self.kin, 'step_ik', return_value=(q, .002)) as step:
                self.assertIsNone(self.kin._try_direct_ik(q, target, self.frame))
        self.assertEqual(step.call_count, 40)

    def test_singular_base_mass_falls_back_to_pseudoinverse(self):
        q = pin.neutral(self.kin.model)
        mass = np.diag([1., 2., 0., 4., 5., 6.])
        coupling = np.arange(36, dtype=float).reshape(6, 6)
        def singular_crba(model, data, config):
            data.M[:6, :6] = mass
            data.M[:6, 6:] = coupling
        with patch('mujoco_robot.robot_ik.pin.crba', side_effect=singular_crba):
            jacobian, result = self.kin._compute_coupled_jacobian(q, self.frame)
        np.testing.assert_allclose(result, np.linalg.pinv(mass) @ coupling)
        self.assertTrue(np.isfinite(jacobian).all())


if __name__ == '__main__':
    unittest.main()

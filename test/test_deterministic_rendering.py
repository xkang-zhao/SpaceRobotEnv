"""Headless rendering regression; opt in with RUN_RENDER_TESTS=1 and MUJOCO_GL=egl."""

import os
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(os.environ.get("RUN_RENDER_TESTS") == "1", "requires EGL rendering")
class DeterministicRenderingTests(unittest.TestCase):
    def test_all_scenes_repeat_and_recreate(self):
        import gymnasium as gym
        import mujoco

        import envs  # noqa: F401
        from mujoco_robot.robot_sensor import RobotSensor

        root = Path(__file__).resolve().parents[1]
        for name in ("Cube", "Satellite", "Satellite2", "Satellite3",
                     "DebrisAntennaPanel", "DebrisTruss"):
            with self.subTest(scene=name):
                spec = gym.spec(f"SpaceUR10e-{name}-v0")
                model = mujoco.MjModel.from_xml_path(str(root / spec.kwargs["scene_path"]))
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                samples = model.vis.quality.offsamples
                reference = None
                for _ in range(2):
                    sensor = RobotSensor(model, data, deterministic_rendering=True)
                    try:
                        self.assertEqual(model.vis.quality.offsamples, samples)
                        self.assertEqual(sensor.renderer._mjr_context.offSamples, 0)
                        for _ in range(3):
                            images = sensor.update_camera_view()
                            if reference is None:
                                reference = images
                            for key in reference:
                                np.testing.assert_array_equal(reference[key], images[key])
                            sensor.cameras_id.reverse()
                    finally:
                        sensor.close()

    def test_default_renderer_is_unchanged(self):
        import mujoco

        from mujoco_robot.robot_sensor import RobotSensor

        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><camera name="cam" pos="0 0 1"/><geom type="sphere" size=".1"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        sensor = RobotSensor(model, data)
        try:
            self.assertIs(type(sensor.renderer), mujoco.Renderer)
        finally:
            sensor.close()


if __name__ == "__main__":
    unittest.main()

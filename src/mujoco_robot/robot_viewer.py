import mujoco
import mujoco.viewer


class RobotViewer:
    def __init__(
        self,
        model,
        data,
        distance=5.0,
        azimuth=135,
        elevation=-30,
        lookat=(1.6, 0.0, 2.65),
    ):
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.lookat = lookat
        self._camera_apply_count = 0
        self.viewer = mujoco.viewer.launch_passive(model, data)

        self._apply_camera()

    def _apply_camera(self):
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.fixedcamid = -1
            self.viewer.cam.trackbodyid = -1
            self.viewer.cam.distance = self.distance
            self.viewer.cam.azimuth = self.azimuth
            self.viewer.cam.elevation = self.elevation
            self.viewer.cam.lookat[:] = self.lookat

    def is_running(self):
        return self.viewer.is_running()

    def sync(self):
        if self._camera_apply_count < 5:
            self._apply_camera()
            self._camera_apply_count += 1
        self.viewer.sync()

    @property
    def cam(self):
        return self.viewer.cam

    @property
    def viewport(self):
        return self.viewer.viewport

    def close(self):
        self.viewer.close()

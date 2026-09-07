
import mujoco
import cv2
import numpy as np

class RobotSensor:
    def __init__(self, model, data, depth_rendering=False):
        self.model = model
        self.data = data

        self.renderer = mujoco.Renderer(self.model, width=640, height=480)

        if depth_rendering == True:
            self.depth_renderer = mujoco.Renderer(self.model, width=640, height=480)
            self.depth_renderer.enable_depth_rendering()
        
        self.cameras_id: list[tuple[int, str]] = self.list_cameras()
        # for cam_id, cam_name in self.cameras_id:
            # cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)

        # print(f"Available cameras: {self.cameras_id}")
        # quit()

    def list_cameras(self):
        cams = []
        for cam_id in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            cams.append((cam_id, name))
        return cams

    def update_camera_view(self, viewer_mode=None):
        images = {}

        for cam_id, cam_name in self.cameras_id:
            self.renderer.update_scene(self.data, camera=cam_id)
            # MuJoCo 返回 RGB。LeRobot 和 Rerun 期望 RGB，因此不需要转换为 BGR
            img = self.renderer.render()
            # print(f"img shape: {img.shape}")
            
            images[cam_name] = img

            # if viewer_mode == "human":
            #     cv2.imshow(cam_name, img)

            # 深度图像
            if hasattr(self, 'depth_renderer') and "left_wrist" in cam_name:
                self.depth_renderer.update_scene(self.data, camera=cam_id)
                depth_img = self.depth_renderer.render()
                # print(f"depth_img shape: {depth_img.shape}")
                # cv2.imshow(f"{cam_name}_depth", depth_img)
                images[f"{cam_name}_depth"] = depth_img
        return images


    def get_force_sensor_data(self, sensor_name):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in the model.")
        sensor_data = self.data.sensordata[sensor_id * 6:(sensor_id + 1) * 6]
        force = sensor_data[0:3]
        torque = sensor_data[3:6]
        return force, torque
    
    def close(self):
        
        self.renderer.close()
        if hasattr(self, 'depth_renderer'):
            self.depth_renderer.close()

        # for cam_id, cam_name in self.cameras_id:
            # cv2.destroyWindow(cam_name)
    

    def get_wrist_force_torque(self):
        force_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_force")
        torque_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_torque")

        # 读取数据的起始地址
        force_adr = self.model.sensor_adr[force_id]
        torque_adr = self.model.sensor_adr[torque_id]

        force_dim = self.model.sensor_dim[force_id]
        torque_dim = self.model.sensor_dim[torque_id]

        force = self.data.sensordata[force_adr : force_adr + force_dim]
        torque = self.data.sensordata[torque_adr : torque_adr + torque_dim]

        return np.concatenate([force, torque])

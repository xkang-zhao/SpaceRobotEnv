# from curses.ascii import FS
from hmac import new
from tkinter import NW
import mujoco

import numpy as np
# import kinematics
import time

import os

# 设置环境变量以确保正确访问游戏杆设备
# os.environ["SDL_JOYSTICK_DEVICE"] = "/dev/input/js0"

# 使用他写的运动学的逆解只有3个自由度，而且也不知道是相对于哪个的

from robot_viewer import RobotViewer
from utils.xbox_controller import XboxController
from robot_ik import Kinematics
from robot_controller import RobotController
from robot_sensor import RobotSensor

class Robot:
    def __init__(self, scene_path, arm_path, frame_name):
        self.scene_path = scene_path
        self.arm_path = arm_path
    
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        
        self.viewer = RobotViewer(self.model, self.data, distance=1.5, azimuth=135, elevation=-30)

        # 这个后面再改
        self.frame_name = frame_name
        self.Kinematics = Kinematics(arm_path, frame_name)
        # self.Kinematics.buildFromMJCF(arm_path)

        # TODO: 写一个sensor类吧
        self.sensor = RobotSensor(self.model, self.data)

        self.controller = RobotController(self.model, self.data)
        self.xbox = XboxController()

        # self.init_pose(np.array([1.0, 0.2, 0.3, 0.0, 3.14, -1.57]))
        self.init_pose(np.array([1.0, 1.0, 3, 1.57, 0.0, 1.57]))

    def init_pose(self, init_pose: np.ndarray):
           
        def compute_ik_from_pose_with_fixed_base(init_pose):
            # 前7个为基座的自由度，再往后是机械臂的关节角
            sim_q = self.data.qpos.copy()
            current_q_pin = np.zeros(13)
            current_q_pin[2] = sim_q[2]
            current_q_pin[3] = sim_q[4] # x
            current_q_pin[4] = sim_q[5] # y
            current_q_pin[5] = sim_q[6] # z
            current_q_pin[6] = sim_q[3] # w
            current_q_pin[8] = -1.24  # 初值保证不奇异

            rotation_matrix = self.Kinematics.euler_to_rotation_matrix(init_pose[3], init_pose[4], init_pose[5])
            target_se3 = self.Kinematics.create_target_pose(rotation_matrix, np.array([init_pose[0], init_pose[1], init_pose[2]]))

            init_q, _ = self.Kinematics.ik_with_fixed_base(current_q_pin[7:], current_q_pin[:7], target_se3, steps=100)
            
            return init_q

        init_q = compute_ik_from_pose_with_fixed_base(init_pose)
        
        # 讲机器人控制器初始化到该位置
        self.controller.target_pose = init_pose
        self.controller.update_qpos(init_q[7:])
        self.controller.update_control(init_q[7:])
        mujoco.mj_forward(self.model, self.data) # 正向动力学
        self.viewer.sync()
        

    def run_loop(self):
        self._running = True
        while self._running:
            # 从xbox获取actions，相对于工件坐标系的增量
            actions = self.xbox.handle_input() # 一个7维的数组，表示变化量
            # print(np.round(actions, 2)) # 格式化为小数点后两位
            
            # 使用controller更新目标位置，得到世界坐标系下的位姿
            self.controller.update_target_pose(actions)
            self.controller.update_target_gripper(actions)
            target_pose = self.controller.get_target_pose() # 六维向量，可以考虑后面改为四元数
            # print(np.round(target_pose, 6))
            target_gripper = self.controller.target_gripper

            def compute_ik_from_pose(target_pose):
                # 前7个为基座的自由度，再往后是机械臂的关节角

                sim_q = self.data.qpos.copy()
                current_q_pin = np.zeros(13)
                # 复制并重排四元数
                current_q_pin[:3] = sim_q[:3]
                current_q_pin[3] = sim_q[4] # x
                current_q_pin[4] = sim_q[5] # y
                current_q_pin[5] = sim_q[6] # z
                current_q_pin[6] = sim_q[3] # w
                current_q_pin[7:] = sim_q[7:13]

                # 求解逆运动学，将目标位姿转换为关节角度
                rotation_matrix = self.Kinematics.euler_to_rotation_matrix(target_pose[3], target_pose[4], target_pose[5])
                target_pose_se3 = self.Kinematics.create_target_pose(rotation_matrix, np.array([target_pose[0], target_pose[1], target_pose[2]]))
            
                new_q, _ = self.Kinematics.ik(current_q_pin, target_pose_se3) # TODO: 这里还没写完,运动学中将不再添加夹爪自由度
                # print(f"new_q shape: {new_q.shape}, new_q: {np.round(new_q, 2)}")
                
                return new_q
            
            new_q = compute_ik_from_pose(target_pose)
            
            self.Kinematics.last_q = new_q.copy()
            # print(f"laset_q: {np.round(self.Kinematics.last_q, 2)}")

            # 更新控制器
            self.controller.update_control(new_q[7:]) # 只传入机械臂的关节角

            # 推进仿真
            for _ in range(100):  # 增加仿真步长以提高控制频率
                mujoco.mj_step(self.model, self.data) # 进行一次仿真步长更新

            # 同步更新界面
            self.viewer.sync()
            self.sensor.update_camera_view()
            
            # 获取力传感器数据，与环境控制频率保持 20 Hz
            # TF = self.sensor.get_wrist_force_torque()
            # print(f"FT Sensor: {np.round(TF.tolist(), 6)}")
            # 更新， renderer的camera, 更新viewer, self.update_camera_view()
            
            
if __name__ == "__main__":
    SCENE_XML_PATH = './model/scene.xml'
    ARM_XML_PATH = './model/dual_space_ur10e_2f85_wo_gripper.xml'
    frame_name = "left_attachment"
    # controller = XboxController()
    # if not controller.is_connected():
    #     print("控制器连接失败，程序将退出。")
    #     exit(1)
    
    # try:   right_attachment
    #     robot = RobotController(SCENE_XML_PATH, ARM_XML_PATH, controller)
    #     robot.run_loop()
    # finally:
    #     controller.cleanup()

    robot = Robot(SCENE_XML_PATH, ARM_XML_PATH, frame_name)
    try:
        robot.run_loop()
    finally:
        robot._running = False
        robot.viewer.close()
        print("关闭viewer")
        robot.sensor.close()
        print("关闭cv窗口")
        robot.xbox.disconnect()
        print("关闭xbox手柄连接")
        


import pygame
import numpy as np

class XboxController:
    """Xbox手柄控制器类，负责处理所有手柄输入
    
    事实上，我想要xbox输出是相对于末端坐标系或者相对于基座坐标系的增量变化
    """
    
    def __init__(self):

        self.actions = np.zeros(6)  # [x, y, z, pitch, roll, yaw]
        
        # 控制灵敏度
        self.tran_sensitivity = 0.003
        self.rot_sensitivity = 0.03
        self.gripper_sensitivity = 1
        
        # 死区阈值（过滤摇杆中心微小漂移）
        self.deadzone = 0.1
        
        # 储存夹爪目标值，用于夹爪开合控制
        self.gripper_target = 0.0
        
        # 跟踪按钮状态，避免重复触发
        self.button_shift_rot = False
        self.control_rot = False
        self.button_tare_pressed = False
        self.trigger_tare = False

        # 初始化手柄
        self.controller = self.connect()
        
    def connect(self):
        """初始化Xbox手柄（通过pygame访问js设备）"""
        
        # 初始化pygame和游戏杆模块
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            print("未检测到任何游戏杆设备")
            return None
            
        # 对应/dev/input/js0
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"检测到手柄: {joystick.get_name()}")
        print(f"按钮数量: {joystick.get_numbuttons()}")
        return joystick
        
    def is_connected(self):
        """检查手柄是否连接"""
        return self.controller is not None
        
    def handle_input(self) -> np.ndarray:
        """处理手柄输入并更新控制参数"""
        if not self.is_connected():
            return
            
        # 处理pygame事件（必须调用，否则无法更新轴值和按钮状态）
        pygame.event.pump()
        
        # 旋转控制
        button1 = self.controller.get_button(0)
        if button1 and not self.button_shift_rot: # 防止重复触发
            print("Button1 pressed - 切换平移/旋转模式")
            self.control_rot = not self.control_rot
            self.button_shift_rot = True
        elif not button1:
            self.button_shift_rot = False
            
        # 去皮控制 (Back 按钮 / 按钮 6)
        button_back = self.controller.get_button(6)
        if button_back and not self.button_tare_pressed:
            print("Button Back pressed - 请求传感器去皮")
            self.trigger_tare = True
            self.button_tare_pressed = True
        elif not button_back:
            self.button_tare_pressed = False


        if self.control_rot:
            # pitch, roll, yaw
            pitch_axis = self.controller.get_axis(0) # 绕x轴旋转
            if abs(pitch_axis) < self.deadzone:
                pitch_axis = 0.0

            roll_axis = self.controller.get_axis(1) # 绕y轴旋转
            if abs(roll_axis) < self.deadzone:
                roll_axis = 0.0

            yaw_axis = self.controller.get_axis(4) # 绕z轴旋转
            if abs(yaw_axis) < self.deadzone:
                yaw_axis = 0.0

            x_axis = 0.0
            y_axis = 0.0
            z_axis = 0.0
        else:
              # 读取左摇杆X轴（0号轴）
            x_axis = -self.controller.get_axis(0)
            # 应用死区过滤
            if abs(x_axis) < self.deadzone:
                x_axis = 0.0

            # 读取左摇杆Y轴（1号轴）
            y_axis = -self.controller.get_axis(1)
            if abs(y_axis) < self.deadzone:
                y_axis = 0.0
                
            # 读取右摇杆X轴（3号轴）
            z_axis = self.controller.get_axis(4)
            if abs(z_axis) < self.deadzone:
                z_axis = 0.0

            pitch_axis = 0.0
            roll_axis = 0.0
            yaw_axis = 0.0

        # 读取夹爪控制轴（4号和5号轴）
        gripper_axis_1 = -(self.controller.get_axis(2) + 1.0) /2  # 绕z轴旋转,由于默认值为1，所以要-1
        gripper_axis_2 = (self.controller.get_axis(5) + 1.0) /2  # 绕z轴旋转,由于默认值为1，所以要-1


        if gripper_axis_1 == 0.0:
            gripper_axis = gripper_axis_2
        else:
            gripper_axis = gripper_axis_1
        if abs(gripper_axis) < self.deadzone:
            gripper_axis = 0.0


        # ✅ 应用工具坐标系下的增量
        delta_trans = np.array([
            x_axis * self.tran_sensitivity,
            y_axis * self.tran_sensitivity,
            z_axis * self.tran_sensitivity
        ])

        delta_rot = np.array([
            pitch_axis * self.rot_sensitivity,
            roll_axis * self.rot_sensitivity,
            yaw_axis * self.rot_sensitivity
        ])

        delta_gripper = gripper_axis * self.gripper_sensitivity


        self.actions = np.array([
            delta_trans[0], delta_trans[1], delta_trans[2], 
            delta_rot[0], delta_rot[1], delta_rot[2],
            delta_gripper
        ])

        return self.actions

        # self.apply_tool_delta(delta_trans, delta_rot)

        # self.gripper_target += gripper_axis * 0.005
        # self.gripper_target = np.clip(self.gripper_target, 0.0, 1.0) # 假设夹爪dof范围是0.0~0.8

    def disconnect(self):
        pygame.quit()

    def consume_tare_request(self):
        """拨动 Back 键后返回 True，供控制器触发一次去皮"""
        if self.trigger_tare:
            self.trigger_tare = False
            return True
        return False

    def get_position_target(self):
        return self.x, self.y, self.z, self.pitch, self.roll, self.yaw
        
    def get_gripper_target(self):
        return self.gripper_target
        


    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        """欧拉角转旋转矩阵（ZYX顺序）"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
    
    def rotation_matrix_to_euler(self, R):
        """旋转矩阵转欧拉角（ZYX顺序）"""
        pitch = np.arcsin(-R[2, 0])
        
        if np.cos(pitch) > 1e-6:  # 正常情况
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:  # 万向锁情况
            yaw = 0.0
            roll = np.arctan2(-R[0, 1], R[1, 1])
        
        return roll, pitch, yaw
    
    def pose_to_transform(self, x, y, z, roll, pitch, yaw):
        """位姿转4x4变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.euler_to_rotation_matrix(roll, pitch, yaw)
        T[:3, 3] = [x, y, z]
        return T
    
    def transform_to_pose(self, T):
        """4x4变换矩阵转位姿"""
        x, y, z = T[:3, 3]
        roll, pitch, yaw = self.rotation_matrix_to_euler(T[:3, :3])
        return x, y, z, roll, pitch, yaw
    
    def apply_tool_delta(self, delta_trans, delta_rot):
        """在工具坐标系下应用增量变换"""
        # 1. 获取当前位姿的变换矩阵
        T_current = self.pose_to_transform(
            self.x, self.y, self.z,
            self.roll, self.pitch, self.yaw
        )
        
        # 2. 构建工具坐标系下的增量变换
        T_delta = np.eye(4)
        T_delta[:3, 3] = delta_trans
        T_delta[:3, :3] = self.euler_to_rotation_matrix(*delta_rot)
        
        # 3. 右乘增量变换（工具坐标系）
        T_new = T_current @ T_delta
        
        # 4. 提取新位姿
        x, y, z, roll, pitch, yaw = self.transform_to_pose(T_new)
        
        # 5. 应用限制
        self.x = np.clip(x, self.x_min, self.x_max)
        self.y = np.clip(y, self.y_min, self.y_max)
        self.z = np.clip(z, self.z_min, self.z_max)
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw


if __name__ == "__main__":
    import time
    controller = XboxController()
    if controller.is_connected():
        while True:
            actions = controller.handle_input(None)
            print("Actions:", actions)
            time.sleep(0.1)
    controller.disconnect()
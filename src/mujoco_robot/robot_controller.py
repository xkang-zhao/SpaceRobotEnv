
import numpy as np

class RobotController:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # 主要用于储存目标位置和姿态    
        self.target_pose = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.target_gripper = 0.0

        # 位置限制,保护末端执行器不超出工作空间，需要计算工作空间
        self.x_min, self.x_max = -5.0, 5.0
        self.y_min, self.y_max = -5.0, 5.0
        self.z_min, self.z_max = -5.0, 5.0

        self.last_q = None  # 储存上一次的关节角度
        self.last_delta_gripper = 0  # 记录上次的夹爪动作状态
        self.max_gripper_target = 255.0

    def update_target_pose(self, actions):
        """在工具坐标系下应用增量变换"""
        # 1. 获取当前工件坐标系位姿的变换矩阵
        T_current_pose = RobotController.pose_to_transform(self.target_pose)
        
        # 2. 构建工具坐标系下的增量变换
        T_delta = np.eye(4)
        T_delta[:3, 3] = actions[:3]
        T_delta[:3, :3] = self.euler_to_rotation_matrix(*actions[3:6])
        
        # 3. 右乘增量变换（工具坐标系）
        T_new = T_current_pose @ T_delta
        
        # 4. 提取新位姿
        new_pose = RobotController.transform_to_pose(T_new)
        
        # 5. 笛卡尔空间限制
        x = np.clip(new_pose[0], self.x_min, self.x_max)
        y = np.clip(new_pose[1], self.y_min, self.y_max)
        z = np.clip(new_pose[2], self.z_min, self.z_max)

        self.target_pose = np.array([x, y, z, new_pose[3], new_pose[4], new_pose[5]])    
        
    def update_target_gripper(self, actions, reset=None):
        """更新夹爪目标位置"""
        # ------------------------ 01 控制（瞬间翻转）-----------
        # if reset is not None:
        #     self.target_gripper = 0
        #     return
        # delta_gripper = actions[6]  # 假设第7个元素是夹爪的增量
        #
        # # 如果从0变为1或者从1变为0（状态发生改变），则翻转夹爪的目标位置
        # if delta_gripper != self.last_delta_gripper:
        #     self.target_gripper = self.max_gripper_target if self.target_gripper == 0.0 else 0.0
        #     self.last_delta_gripper = delta_gripper

        # --------------------连续增量模式（慢速闭合）---------------
        # actions[6] ∈ [-1, 1]:  正值驱动夹爪闭合,  负值驱动夹爪张开
        # 每帧累加 GRIPPER_SPEED * actions[6],  到达目标值 255(全闭) / 0(全开) 后停止
        GRIPPER_SPEED = 25.0  # 每步增量，越大越快；调到 5~10 则为很慢

        if reset is not None:
            self.target_gripper = 0
            return

        delta = actions[6]                       # 期望方向与强度
        self.target_gripper += delta * GRIPPER_SPEED
        self.target_gripper = np.clip(
            self.target_gripper,
            0,
            self.max_gripper_target,
        )
        

    def get_target_pose(self):
        return self.target_pose
    
    def update_control(self, new_q):
        self.data.ctrl[0:6] = new_q
        self.data.ctrl[6] = self.target_gripper

    def update_qpos(self, new_q):
        self.data.qpos[7:7+6] = new_q
    

    @staticmethod
    def euler_to_rotation_matrix(roll, pitch, yaw):
        """欧拉角转旋转矩阵（ZYX顺序）"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
    
    @staticmethod
    def rotation_matrix_to_euler(R):
        """旋转矩阵转欧拉角（ZYX顺序）"""
        pitch = np.arcsin(-R[2, 0])
        
        if np.cos(pitch) > 1e-6:  # 正常情况
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        else:  # 万向锁情况
            yaw = 0.0
            roll = np.arctan2(-R[0, 1], R[1, 1])
        
        return roll, pitch, yaw


    @staticmethod
    def pose_to_transform(target_pose): # 表示当前的pose, 在世界坐标系下
        """位姿转4x4变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = RobotController.euler_to_rotation_matrix(target_pose[3],
                                                            target_pose[4], 
                                                            target_pose[5])
        T[:3, 3] = target_pose[:3]
        
        return T
    
    @staticmethod
    def transform_to_pose(T):
        """4x4变换矩阵转位姿"""
        x, y, z = T[:3, 3]
        roll, pitch, yaw = RobotController.rotation_matrix_to_euler(T[:3, :3])
        
        targe_pose = np.array([x, y, z, roll, pitch, yaw])

        return targe_pose


if __name__ == "__main__":
    print("="*60)
    print("RobotController 单元测试")
    print("="*60)
    
    # 测试 1: 基础初始化（model/data 可为 None）
    print("\n[测试 1] 初始化控制器")
    controller = RobotController(model=None, data=None)
    print(f"✅ 初始目标位姿: {controller.target_pose}")
    print(f"✅ 初始夹爪状态: {controller.target_gripper}")
    
    # 测试 2: 静态方法 - 欧拉角与旋转矩阵转换
    print("\n[测试 2] 欧拉角 ↔ 旋转矩阵转换")
    test_euler = np.array([np.pi/6, np.pi/4, np.pi/3])  # 30°, 45°, 60°
    R = RobotController.euler_to_rotation_matrix(*test_euler)
    recovered_euler = RobotController.rotation_matrix_to_euler(R)
    error = np.abs(test_euler - np.array(recovered_euler))
    print(f"原始欧拉角: {np.degrees(test_euler)} 度")
    print(f"恢复欧拉角: {np.degrees(recovered_euler)} 度")
    print(f"误差: {error} {'✅ 通过' if np.max(error) < 1e-10 else '❌ 失败'}")
    
    # 测试 3: 位姿与变换矩阵转换
    print("\n[测试 3] 位姿 ↔ 变换矩阵转换")
    test_pose = np.array([0.5, -0.3, 0.8, 0.1, 0.2, 0.3])
    T = RobotController.pose_to_transform(test_pose)
    recovered_pose = RobotController.transform_to_pose(T)
    pose_error = np.abs(test_pose - recovered_pose)
    print(f"原始位姿: {test_pose}")
    print(f"恢复位姿: {recovered_pose}")
    print(f"误差: {pose_error} {'✅ 通过' if np.max(pose_error) < 1e-10 else '❌ 失败'}")
    
    # 测试 4: 工具坐标系增量更新（纯平移）
    print("\n[测试 4] 工具坐标系纯平移")
    controller.target_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions_translation = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])  # 仅平移
    controller.update_target_pose(actions_translation)
    expected_pos = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    pos_error = np.abs(controller.target_pose - expected_pos)
    print(f"期望位姿: {expected_pos}")
    print(f"实际位姿: {controller.target_pose}")
    print(f"{'✅ 通过' if np.max(pos_error[:3]) < 1e-10 else '❌ 失败'}")
    
    # 测试 5: 工具坐标系增量更新（纯旋转）
    print("\n[测试 5] 工具坐标系纯旋转")
    controller.target_pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    actions_rotation = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2])  # 绕Z轴旋转90°
    controller.update_target_pose(actions_rotation)
    print(f"初始位置: [1.0, 0.0, 0.0]")
    print(f"旋转后位置: {controller.target_pose[:3]}")
    print(f"旋转后姿态: {np.degrees(controller.target_pose[3:])} 度")
    # 验证位置是否保持（工具坐标系旋转不改变位置）
    print(f"位置保持: {'✅' if np.allclose(controller.target_pose[:3], [1.0, 0.0, 0.0]) else '❌'}")
    
    # 测试 6: 笛卡尔空间限制
    print("\n[测试 6] 笛卡尔空间限制")
    controller.target_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    extreme_actions = np.array([100.0, -100.0, 50.0, 0.0, 0.0, 0.0])
    controller.update_target_pose(extreme_actions)
    print(f"极限输入: {extreme_actions[:3]}")
    print(f"限幅后位置: {controller.target_pose[:3]}")
    is_clamped = (
        controller.x_min <= controller.target_pose[0] <= controller.x_max and
        controller.y_min <= controller.target_pose[1] <= controller.y_max and
        controller.z_min <= controller.target_pose[2] <= controller.z_max
    )
    print(f"限幅正确: {'✅' if is_clamped else '❌'}")
    
    # 测试 7: 连续增量更新（模拟手柄输入）
    print("\n[测试 7] 连续增量更新")
    controller.target_pose = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    print("模拟10步小幅移动:")
    for i in range(10):
        delta = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.1])
        controller.update_target_pose(delta)
    print(f"最终位姿: {controller.target_pose}")
    print(f"预期X位移约: 0.1, 实际: {controller.target_pose[0]:.4f}")
    print(f"预期Yaw约: {np.degrees(1.0):.2f}°, 实际: {np.degrees(controller.target_pose[5]):.2f}°")
    
    # 测试 8: 更新控制指令（需要模拟 data 对象）
    print("\n[测试 8] 更新控制指令")
    class MockData:
        def __init__(self):
            self.ctrl = np.zeros(7)
    
    mock_data = MockData()
    controller.data = mock_data
    test_q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    controller.target_gripper = 0.8
    controller.update_control(test_q)
    print(f"关节角度指令: {controller.data.ctrl[:6]}")
    print(f"夹爪指令: {controller.data.ctrl[6]}")
    print(f"{'✅ 通过' if np.allclose(controller.data.ctrl[:6], test_q) else '❌ 失败'}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

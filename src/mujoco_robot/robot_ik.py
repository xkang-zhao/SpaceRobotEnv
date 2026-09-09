from math import e
from turtle import st
import pinocchio as pin
import numpy as np
from numpy.linalg import inv, norm, pinv


class Kinematics:
    def __init__(self, model_path, ee_frame) -> None:
        """
        初始化运动学求解器
        
        参数:
            model_path: MJCF模型文件路径
            ee_frame: 末端执行器的frame名称
        """
        self.model_path = model_path
        self.frame_name = ee_frame  # 末端执行器的frame名称
        self.create_model()  # 创建pinocchio机器人模型

    def create_model(self):
        self.arm = pin.RobotWrapper.BuildFromMJCF(self.model_path)

        self.model = self.arm.model
        self.data = self.arm.data
        self._error_data = self.model.createData()

    @staticmethod
    def create_target_pose(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> pin.SE3:
        '''
        创建目标末端位姿
        
        参数:
            rotation_matrix: 3x3旋转矩阵
            translation_vector: 3维平移向量
            
        返回:
            pin.SE3对象，表示目标末端位姿
        '''
        return pin.SE3(rotation_matrix, translation_vector)

    def compute_matrices(self, q, frame_id):
        '''
        核心计算：计算广义雅可比矩阵
        
        该函数计算考虑了基座反向漂移的末端执行器雅可比矩阵。
        基于动量守恒原理，当机械臂运动时，无法固定的基座会反向移动。
        
        参数:
            q: 当前关节配置 (13维: 6维基座 + 7维机械臂)
            frame_id: 末端执行器的frame ID
            
        返回:
            J_g: 广义雅可比矩阵 (6x7)，描述关节速度到末端速度的映射，维度为7说明是4元数表示
            M_bb_inv: 基座质量矩阵的逆 (6x6)
            M_bm: 基座-机械臂耦合项 (6x7)
        '''
        model = self.model
        data = self.data
        
        # 1. 计算惯性矩阵
        # CRBA (Composite Rigid Body Algorithm) 高效计算整个系统的质量/惯性矩阵
        pin.crba(model, data, q)
        M = data.M  # 13x13 系统惯性矩阵
        
        # 2. 提取基座和机械臂的耦合项
        # M_bb: 基座的惯性矩阵 (6x6)
        # M_bm: 基座与机械臂的耦合项 (6x7)
        M_bb_inv = pinv(M[0:6, 0:6])  # 基座惯性矩阵的伪逆
        M_bm = M[0:6, 6:]  # 基座-机械臂耦合项
        
        # 3. 计算空间雅可比矩阵
        # 计算所有关节的雅可比矩阵 (预计算阶段)
        pin.computeJointJacobians(model, data, q)
        # 更新所有frame的位置和姿态
        pin.framesForwardKinematics(model, data, q)
        # 获取末端执行器相对于世界坐标系的雅可比矩阵 (6x13)
        # 前6列对应基座，后7列对应机械臂关节
        J_spatial = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        
        # 4. 分离基座和机械臂部分
        J_b = J_spatial[:, 0:6]  # 末端对基座的雅可比 (6x6)
        J_m = J_spatial[:, 6:]   # 末端对机械臂的雅可比 (6x7)
        
        # 5. 计算广义雅可比矩阵
        # 关键公式: J_g = J_m - J_b @ M_bb_inv @ M_bm
        # 物理意义: 末端速度 = 机械臂直接贡献 - 基座反向漂移的影响
        J_g = J_m - J_b @ M_bb_inv @ M_bm
        
        return J_g, M_bb_inv, M_bm

    def _compute_coupled_jacobian(self, q, frame_id):
        """Update placements and return the Jacobian and momentum coupling.

        Keep compute_matrices' inverse-returning interface for report consumers;
        the control loop only needs the product solve(M_bb, M_bm).
        """
        model, data = self.model, self.data
        pin.crba(model, data, q)
        m_bb, m_bm = data.M[:6, :6], data.M[:6, 6:]
        try:
            coupling = np.linalg.solve(m_bb, m_bm)
        except np.linalg.LinAlgError:
            coupling = pinv(m_bb) @ m_bm
        pin.computeJointJacobians(model, data, q)
        # Joint Jacobians already update joint placements for this q.
        pin.updateFramePlacements(model, data)
        jacobian = pin.getFrameJacobian(
            model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        return jacobian[:, 6:] - jacobian[:, :6] @ coupling, coupling

    def step_ik(self, q, target_pose, frame_id, damping=1e-1, dt=0.05):
        '''单步 IK (Safety Tuned)'''
        # 1. Error
        # 计算正运动学。作用：根据当前的 q，更新所有连杆和 Frame 的位置。必须先做这一步才能知道现在手在哪。
        model = self.model
        data = self.data
        J_g, coupling = self._compute_coupled_jacobian(q, frame_id)
        curr_pose = data.oMf[frame_id]
        
        # 计算位置误差（目标位置 - 当前位置）。这是一个 3D 向量
        err_pos = target_pose.translation - curr_pose.translation
        
        # 计算旋转误差（在局部坐标系下）。R_curr.T @ R_target 计算了从当前姿态到目标姿态的相对旋转矩阵。pin.log3 将这个旋转矩阵转换为旋转向量（轴角表示）。
        err_rot_local = pin.log3(curr_pose.rotation.T @ target_pose.rotation)
        err_rot = curr_pose.rotation @ err_rot_local

        #将局部旋转误差转换到世界坐标系。这样位置误差和旋转误差就在同一个参考系下了。
        error = np.concatenate([err_pos, err_rot])
        
        # 2. Jacobian
        # J_g描述了“如果我动一下关节，考虑到基座的反向漂移，末端会怎么动”
        
        # 3. Velocity
        # print(f"q: {q}")

        # while True:
        #     pass

        H = J_g.T @ J_g + (damping**2) * np.eye(6)

        # 计算梯度方向。这是为了让误差平方和下降最快的方向。
        g = J_g.T @ error

        # 求解线性方程组，得到机械臂关节的目标速度
        v_m = np.linalg.solve(H, g)
        
        # 4. Momentum Coupling
        # 根据动量守恒，计算基座的被动速度
        v_b = -coupling @ v_m
        v_total = np.concatenate([v_b, v_m])
        
        # [关键安全限制] 限制最大关节速度
        # 0.2 rad/step 约等于 11度，这已经很快了，超过这个值通常意味着数学爆炸
        max_step_vel = 0.2 
        v_norm = norm(v_total)
        if v_norm > max_step_vel:
            scale = max_step_vel / v_norm
            v_total = v_total * scale
            
        # 5. Integrate
        q_next = pin.integrate(model, q, v_total * dt)
        return q_next, norm(error)

    def _target_error(self, q, target_pose, frame_id):
        """Residual at q, in the same world-aligned convention as step_ik."""
        pin.framesForwardKinematics(self.model, self._error_data, q)
        pose = self._error_data.oMf[frame_id]
        return norm(np.r_[
            target_pose.translation - pose.translation,
            pose.rotation @ pin.log3(pose.rotation.T @ target_pose.rotation),
        ])

    def solve_trajectory_ik(self, q_init, final_pose, steps=10, *, early_stop=True):
        '''
            q_init: np.ndarray 初始自由度(13维: 7维基座 + 6维机械臂)
            final_pose: pin.SE3 目标末端位姿
            steps: int 插值步数
            early_stop: 在初始状态和每组10次迭代后检查最终目标；混合位姿
                残差不超过1e-4时提前返回。False保留原有固定预算求解。
        
        '''
        model = self.model
        data = self.data

        q = q_init.copy()
        frame_id = model.getFrameId(self.frame_name)
        if steps < 1:
            raise ValueError('IK interpolation steps must be positive')
        # A stricter threshold than the existing 1e-3 refinement tolerance.
        # Test the final target, never an intermediate interpolation waypoint.
        if early_stop:
            final_error = self._target_error(q, final_pose, frame_id)
            if final_error <= 1e-4:
                return q, final_error
        
        # 获取起点
        pin.framesForwardKinematics(model, data, q)
        start_pose = data.oMf[frame_id]
        # 记录初始基座位置，用于后续监测基座漂移（当前未使用，保留备用）
        # start_base = q[0:3]
        
        # 主插值循环：从起点逐步插值到目标位置
        for i in range(steps):
            # 计算当前插值系数 alpha，范围从 0 到 1
            alpha = (i + 1) / steps 
            
            # 线性插值位置：在起点和目标点之间做线性分插
            target_pos_i = start_pose.translation + alpha * (final_pose.translation - start_pose.translation)
            # 四元数球面线性插值：平滑地插值两个旋转姿态
            quat_start = pin.Quaternion(start_pose.rotation)
            quat_end = pin.Quaternion(final_pose.rotation)
            quat_i = quat_start.slerp(alpha, quat_end)
            # 组合成当前插值点的目标位姿
            target_pose_i = pin.SE3(quat_i.matrix(), target_pos_i)
            
            # 对每个插值点执行多个小步迭代，确保收敛性和稳定性
            for _ in range(10): 
                q, err = self.step_ik(q, target_pose_i, frame_id, dt=0.05, damping=5e-2)
            if early_stop:
                # A separate Data object avoids changing the legacy interpolation
                # start_pose alias while checking the newly integrated state.
                final_error = self._target_error(q, final_pose, frame_id)
                if final_error <= 1e-4:
                    return q, final_error
                
            # 可选：输出调试信息（已注释）
            # if i % 20 == 0:
            #     drift = norm(q[0:3] - start_base)
            #     print(f"Step {i:3d}/{steps}: Local Err={err:.4f} | Base Drift={drift:.3f}m")

        # 精调阶段：使用更小的阻尼以提高精度
        # print("Final finetuning (High Precision)...")
        
        # 执行最多200次精调迭代，当误差小于阈值时提前停止
        _fine_warn_counter = getattr(self, '_fine_warn_counter', 0)
        for _ in range(50):
            if err <= 1e-3:
                break
            # 减小阻尼参数（从 1e-2 减至 0.01）以获得更精确的解
            q, err = self.step_ik(q, final_pose, frame_id, dt=0.05, damping=0.05)
        
        # 如果精调未完全收敛，输出警告但继续执行（每50次才打印一次）
        if err > 1e-3:
            _fine_warn_counter += 1
            self._fine_warn_counter = _fine_warn_counter
            if _fine_warn_counter % 50 == 1:
                print(f"[WARN] IK Fine-tuning did not fully converge (err={err:.4f}, cnt={_fine_warn_counter}), but proceeding.")
                
        return q, err

    def _try_direct_ik(self, q_init, target_pose, frame_id):
        """Solve a small target change; return None to retry from q_init.

        dt is an internal numerical step, unrelated to MuJoCo's timestep.
        Keep the existing damping and velocity norm limit. Accept only a
        finite, decreasing residual at the same 1e-4 early-stop tolerance.
        """
        error = self._target_error(q_init, target_pose, frame_id)
        if not 1e-4 < error <= 0.02:
            return None
        q = q_init.copy()
        for _ in range(4):
            for _ in range(10):
                q, _ = self.step_ik(q, target_pose, frame_id, dt=0.2, damping=0.05)
                if not np.isfinite(q).all():
                    return None
            next_error = self._target_error(q, target_pose, frame_id)
            if not np.isfinite(next_error):
                return None
            if next_error >= error:
                return None
            if next_error <= 1e-4:
                return q, next_error
            error = next_error
        return None

    def ik(self, q_init: np.ndarray, target_pose: pin.SE3, steps=10, *,
           early_stop=True, fast_path=True):
        """
        Solve the inverse kinematics problem to find joint angles that achieve the target end-effector pose.
        This method computes the joint configuration needed to reach a desired end-effector pose
        using an iterative trajectory-based inverse kinematics approach.
        Args:
            q_init (np.ndarray): Initial joint configuration (starting point for IK solver).
            target_pose (pin.SE3): Target end-effector pose in SE3 format (position and orientation).
            steps (int, optional): Number of interpolation points. Defaults to 10.
            early_stop (bool): Check the final target after each interpolation
                block and stop at residual <= 1e-4. Set False for the legacy
                iteration schedule. This does not change grasp success criteria.
            fast_path (bool): With early_stop and steps >= 8, try up to 40 direct
                iterations for a mixed pose residual <= 0.02; otherwise use the
                trajectory solver. Failure restarts from the original input.
        Returns:
            tuple: A tuple containing:
                - q_final (np.ndarray): Joint configuration that achieves the target pose.
                - final_err (float): Final error/residual of the IK solution.
        Notes:
            - Uses pinocchio (pin) for forward kinematics calculations.
            - The actual IK solving is delegated to the solve_trajectory_ik method.
        """
        if fast_path and early_stop and steps >= 8:
            frame_id = self.model.getFrameId(self.frame_name)
            result = self._try_direct_ik(q_init, target_pose, frame_id)
            if result is not None:
                return result
        q_final, final_err = self.solve_trajectory_ik(
            q_init, target_pose, steps=steps, early_stop=early_stop
        )

        return q_final, final_err


    def step_ik_fixed_base(self, q, target_pose, frame_id, damping=1e-1, dt=0.05):
        '''固定基座的单步 IK
        
        '''
        model = self.model
        data = self.data
        
        # 1. 误差计算 (同原代码)
        pin.framesForwardKinematics(model, data, q)  # 计算正运动学
        curr_pose = data.oMf[frame_id]  # 当前末端执行器位姿
        err_pos = target_pose.translation - curr_pose.translation  # 位置误差
        err_rot_local = pin.log3(curr_pose.rotation.T @ target_pose.rotation)  # 旋转误差（局部坐标系下）
        err_rot = curr_pose.rotation @ err_rot_local  # 将局部旋转误差转换到世界坐标系
        error = np.concatenate([err_pos, err_rot])  # 合并位置和旋转误差
        
        # 2. 雅可比矩阵 - 只取机械臂部分
        pin.computeJointJacobians(model, data, q)  # 计算所有关节的雅可比矩阵
        J_spatial = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)  # 获取末端执行器的雅可比矩阵
        
        # **关键修改**: 只使用机械臂的雅可比 (列索引 6:)
        '''
        代码通过 J_m = J_spatial[:, 6:] 强行丢弃了前6列。这意味着在数学计算中，算法完全不考虑基座移动的可能性，只利用机械臂的自由度来消除误差。
        '''
        J_m = J_spatial[:, 6:]  # 忽略基座的6列
        
        # 3. 速度求解 - 只求解机械臂速度
        H = J_m.T @ J_m + (damping**2) * np.eye(J_m.shape[1])  # 计算H矩阵
        g = J_m.T @ error  # 计算梯度方向
        v_m = inv(H) @ g  # 求解机械臂的目标速度
        
        # 4. 速度限制
        max_step_vel = 0.2  # 最大关节速度限制
        v_norm = norm(v_m)  # 计算速度的范数
        if v_norm > max_step_vel:  # 如果超过最大速度限制
            v_m = v_m * (max_step_vel / v_norm)  # 按比例缩放速度
        
        # 5. 积分 - 只更新机械臂关节
        # 基座保持不动: v_b = [0,0,0,0,0,0]
        v_total = np.concatenate([np.zeros(6), v_m])  # 合并基座和机械臂的速度
        q_next = pin.integrate(model, q, v_total * dt)  # 更新关节配置
        
        return q_next, norm(error)  # 返回更新后的关节配置和误差

    def solve_trajectory_ik_fixed_base(self, q_init: np.ndarray, final_pose, steps=5):
        '''使用 step_ik_fixed_base 的轨迹求解

            final_pose: pin.SE3 目标末端位姿

        '''
        model = self.model
        # 创建数据结构用于运动学计算
        data = self.data
        # 复制初始配置，避免修改原始输入
        q = q_init.copy()
        # print(f"q: {q}")
        # 获取末端执行器的frame ID
        frame_id = model.getFrameId(self.frame_name)
        
        # 计算起始位姿
        pin.framesForwardKinematics(model, data, q)
        start_pose = data.oMf[frame_id]
        
        # 主插值循环：从起点逐步插值到目标位置
        for i in range(steps):
            # 计算当前插值系数 alpha，范围从 0 到 1
            alpha = (i + 1) / steps
            # 线性插值位置：在起点和目标点之间做线性分插
            target_pos_i = start_pose.translation + alpha * (final_pose.translation - start_pose.translation)
            # 四元数球面线性插值：平滑地插值两个旋转姿态
            quat_start = pin.Quaternion(start_pose.rotation)
            quat_end = pin.Quaternion(final_pose.rotation)
            quat_i = quat_start.slerp(alpha, quat_end)
            # 组合成当前插值点的目标位姿
            target_pose_i = pin.SE3(quat_i.matrix(), target_pos_i)
            
            # 对每个插值点执行多个小步迭代，确保收敛性和稳定性
            for _ in range(10):
                q, err = self.step_ik_fixed_base(q, target_pose_i, frame_id, damping=5e-2)
        
        # 精调阶段：使用更小的阻尼以提高精度
        # 执行最多200次精调迭代，当误差小于阈值时提前停止
        for _ in range(200):
            # 如果误差已经足够小，提前退出循环
            if err <= 1e-3:
                break
            # 减小阻尼参数（0.01）以获得更精确的解
            q, err = self.step_ik_fixed_base(q, final_pose, frame_id, damping=0.05)
        
        # 返回最终关节配置和误差
        return q, err
    
    def ik_with_fixed_base(self, q_arm_init: np.ndarray, base_pose: np.ndarray, target_ee_pose: pin.SE3, steps=100):
        '''
        参数:
            q_arm_init: 机械臂初始关节角 (不含基座的6维)
            base_pose: 固定基座位姿 [x, y, z, 四元数]，这个应该为7维
            target_ee_pose: 目标末端位姿 (SE3)
        '''
        # 构造完整初始配置
        q_init = np.concatenate([base_pose, q_arm_init])
        
        # 使用修改后的 step_ik
        q_final, err = self.solve_trajectory_ik_fixed_base(q_init, target_ee_pose, steps)
        
        # 返回基座+机械臂关节角
        return q_final, err
    
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


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    kin = Kinematics("./model/ur10e_2f85_wo_gripper.xml", "wrist_3_link")

    # # print(kin.model)
    # # 打印自由度信息  
    # print(f"配置空间维度 (nq): {kin.model.nq}")  
    # print(f"速度空间维度/自由度数 (nv): {kin.model.nv}")  
    # print(f"关节数量: {kin.model.njoints}") 
    
    # # 打印关节连接情况  
    # for i in range(kin.model.njoints):  
    #     joint_name = kin.model.names[i]  
    #     parent_id = kin.model.parents[i] if i > 0 else 0  
    #     parent_name = kin.model.names[parent_id]  
    #     print(f"关节 {i}: {joint_name} -> 父关节: {parent_name} (ID: {parent_id})")

    # print("关节类型详情:")  
    # for i in range(1, kin.model.njoints):  # 跳过universe关节  
    #     joint = kin.model.joints[i]  
    #     joint_name = kin.model.names[i]  
    #     print(f"  {joint_name}: {joint.nq}q, {joint.nv}v")  
    # quit()

    
    q_init: np.ndarray = pin.neutral(kin.model) # 13维：7维基座 + 6维机械臂
    # print(type(q_init))

    pin.framesForwardKinematics(kin.model, kin.data, q_init)
    start_pose = kin.data.oMf[kin.model.getFrameId("wrist_3_link")]
    print(f"Start Pos: {start_pose.translation}")
    
    # 设定目标
    target_pos = start_pose.translation + np.array([-0.02, 0.03, 0.01])
    rot_change = pin.AngleAxis(np.deg2rad(0.05), np.array([0., 1., 0.])).matrix()
    target_rot = rot_change @ start_pose.rotation
    final_pose = pin.SE3(target_rot, target_pos)


    q_final, final_err = kin.ik(q_init, final_pose, steps=100)
    # print(f"Goal Pos: {target_pos}")
    
    # q_final, final_err = solve_trajectory_ik(model, q_init, final_pose, steps=100)
    
    # pin.framesForwardKinematics(model, data, q_final)
    # final_result_pose = data.oMf[model.getFrameId("wrist_3_link")]
    
    print("\n" + "="*30)
    print(f"Final Error: {final_err:.6f}")
    # print(f"Result Pos : {final_result_pose.translation}")
    print(f"Base Drift : {q_final[0:3] - q_init[0:3]}")
    
    if final_err < 1e-2:
        print("✅ SUCCESS! Trajectory IK converged.")
    else:
        print("❌ Failed. Still unreachable, but at least didn't explode.")

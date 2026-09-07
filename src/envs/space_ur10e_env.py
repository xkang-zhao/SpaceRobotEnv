"""Configurable Gymnasium environment for SpaceUR10e grasping tasks."""

import os

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

# 复用现有机器人模块
from mujoco_robot.robot_controller import RobotController
from mujoco_robot.robot_ik import Kinematics
from mujoco_robot.robot_sensor import RobotSensor
from mujoco_robot.robot_viewer import RobotViewer


DEFAULT_TARGET_INIT_RANGE = {
    "x": (1.5, 1.7),
    "y": (-0.2, 0.2),
    "z": (2.6, 2.7),
}

DEFAULT_VIEWER_CONFIG = {
    "distance": 5.0,
    "azimuth": 135,
    "elevation": -30,
    "lookat": (1.6, 0.0, 2.65),
}

class SpaceUR10eEnv(gym.Env):
    """可配置的 MuJoCo UR10e 空间机器人抓取环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        scene_path="./mjcf/1_cube_scene.xml",
        arm_path="./mjcf/arm.xml",
        frame_name="left_attachment",
        use_depth=False,
        observation_mode="rgb",
        target_init_range=None,
        viewer_config=None,
        target_body_name="cube",
        target_joint_name="cube:joint",
    ):
        super().__init__()

        self.scene_path = scene_path
        self.arm_path = arm_path
        self.render_mode = render_mode
        if observation_mode not in {"state", "rgb"}:
            raise ValueError(
                "observation_mode must be either 'state' or 'rgb', "
                f"got {observation_mode!r}"
            )
        self.observation_mode = observation_mode
        init_range = target_init_range or DEFAULT_TARGET_INIT_RANGE
        self.target_init_range = {
            axis: tuple(init_range[axis]) for axis in ("x", "y", "z")
        }
        self.viewer_config = {**DEFAULT_VIEWER_CONFIG, **(viewer_config or {})}
        self.target_body_name = target_body_name
        self.target_joint_name = target_joint_name

        # 1. 初始化 MuJoCo
        # 确保路径是绝对路径或相对于当前工作目录正确
        if not os.path.exists(scene_path):
            raise FileNotFoundError(f"Scene file not found: {scene_path}")
            
        self.model = mujoco.MjModel.from_xml_path(scene_path) # type: ignore
        self.data = mujoco.MjData(self.model) # type: ignore

        # 2. 初始化子模块
        self.kinematics = Kinematics(arm_path, frame_name)
        self.controller = RobotController(self.model, self.data)
        self.sensor = None
        if observation_mode == "rgb":
            self.sensor = RobotSensor(
                self.model,
                self.data,
                depth_rendering=use_depth,
            )
        
        # Viewer 初始化 (仅在 human 模式下)
        self.viewer = None
        if render_mode == "human":
            self.viewer = RobotViewer(self.model, self.data, **self.viewer_config)

        # 3. 定义动作空间 (Action Space)
        # [dx, dy, dz, droll, dpitch, dyaw, gripper_ctrl]
        # 范围归一化到 [-1, 1]，在 step 中再缩放
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # 4. 定义观测空间 (Observation Space)
        # 非摄像头数据
        observation_spaces = {
            "joint_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "base_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "ee_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
        }
        if observation_mode == "state":
            observation_spaces["gripper_pos"] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )
        else:
            observation_spaces["target_pose"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7,),
                dtype=np.float64,
            )
        # 根据传感器中实际的相机名称动态添加摄像头空间（与 update_camera_view 行为一致）
        if self.sensor is not None:
            for _, cam_name in self.sensor.cameras_id:
                observation_spaces[cam_name] = spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8)
                if use_depth and "left_wrist" in cam_name:
                    observation_spaces[f"{cam_name}_depth"] = spaces.Box(low=0, high=np.inf, shape=(480, 640), dtype=np.float32)
        self.observation_space = spaces.Dict(observation_spaces)

        # 缓存一些 ID 以便快速访问
        self.target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.target_body_name,
        )
        if self.target_body_id == -1:
            raise ValueError(f"Target body not found in model: {self.target_body_name}")
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_attachment_site")
        self.gripper_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "left_right_driver_joint",
        )
        if self.gripper_joint_id == -1:
            raise ValueError("Gripper joint not found in model: left_right_driver_joint")
        self.gripper_qpos_adr = self.model.jnt_qposadr[self.gripper_joint_id]
        self.gripper_joint_range = self.model.jnt_range[self.gripper_joint_id].copy()
        if (
            not np.all(np.isfinite(self.gripper_joint_range))
            or self.gripper_joint_range[1] <= self.gripper_joint_range[0]
        ):
            raise ValueError(
                "Gripper joint must have an increasing finite position range"
            )

        # ===== 抓取检测与奖励相关的 ID 缓存 =====
        # 夹爪 pad geom ID (左臂 2F-85 夹爪的左右手指)
        self.finger_A_pad_ids = set()  # 左臂夹爪的「左手指」pad
        for name in ["left_left_pad1", "left_left_pad2"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self.finger_A_pad_ids.add(gid)

        self.finger_B_pad_ids = set()  # 左臂夹爪的「右手指」pad
        for name in ["left_right_pad1", "left_right_pad2"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                self.finger_B_pad_ids.add(gid)

        # 目标物体的 geom ID（通过 body 查找，因为目标 geom 可能没有命名）
        target_geom_start = self.model.body_geomadr[self.target_body_id]
        target_geom_num = self.model.body_geomnum[self.target_body_id]
        self.target_geom_ids = set(range(target_geom_start, target_geom_start + target_geom_num))

        # 目标物体 freejoint 在 qvel 中的地址（6 DOF: 3线速度 + 3角速度）
        self.target_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            self.target_joint_name,
        )
        if self.target_joint_id == -1:
            raise ValueError(f"Target joint not found in model: {self.target_joint_name}")
        self.target_dof_adr = self.model.jnt_dofadr[self.target_joint_id]

        # 夹爪中心点 site (left_pinch 位于两指中间，用于距离计算)
        self.pinch_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_pinch")
        if self.pinch_site_id == -1:
            self.pinch_site_id = self.ee_site_id  # fallback

        # ===== 奖励函数超参数 =====
        self.reward_weights = {
            "reach": 1.0,           # 接近奖励权重
            "align": 0.3,           # 姿态对齐权重
            "contact_single": 0.5,  # 单侧接触 bonus
            "contact_dual": 0.5,    # 双侧接触额外 bonus
            "success_bonus": 10.0,  # 抓取成功 bonus
            "action_penalty": 0.01, # 动作幅度惩罚
        }
        self.align_dist_threshold = 0.15       # 启用姿态对齐奖励的距离阈值 (m)
        self.grasp_gripper_threshold = 100.0   # 夹爪闭合判定阈值 (ctrl range 0-255)
        self.target_stable_vel_threshold = 0.1 # 物体稳定速度阈值
        self.fail_dist_threshold = 2.0         # 任务失败距离阈值 (m)

        # ===== 连续成功判定 =====
        self.success_consec_threshold = 10     # 连续 N 步 is_success 才确认抓取成功
        self._success_counter = 0              # 当前连续成功计数

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置 MuJoCo 仿真状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 随机初始化抓取目标位置
        self._init_target()
        
        # 设置初始姿态 (参考 main.py 中的 init_pose)
        init_pose = np.array([1.2, 0.1, 2.7, 1.57, 0.0, 1.57])
        
        # 构造初始配置用于 IK (参考 main.py)
        def compute_ik_from_pose_with_fixed_base(init_pose):
            
            current_q_pin = np.zeros(13)
            current_q_pin[2] = 2
            current_q_pin[6] = 1 # w 
            current_q_pin[8] = -1.24 

            rotation_matrix = self.kinematics.euler_to_rotation_matrix(init_pose[3], init_pose[4], init_pose[5])
            target_se3 = self.kinematics.create_target_pose(rotation_matrix, np.array([init_pose[0], init_pose[1], init_pose[2]]))  #target_pos

            # 使用 ik_with_fixed_base 计算初始关节角 (与 main.py 一致)
            init_q, _ = self.kinematics.ik_with_fixed_base(current_q_pin[7:], current_q_pin[:7], target_se3, steps=100)

            return init_q
        
        init_q = compute_ik_from_pose_with_fixed_base(init_pose)
        
        # 更新控制器目标
        self.controller.target_pose = init_pose

        # 直接设置关节状态
        self.controller.update_qpos(init_q[7:])
        self.controller.update_control(init_q[7:])
        self.controller.update_target_gripper(actions=None, reset=0)
        mujoco.mj_forward(self.model, self.data) # 正向动力学

        # 重置连续成功计数器
        self._success_counter = 0
        
        # 同步 Viewer
        if self.viewer:
            self.viewer.sync()

        return self._get_obs(), {}

    def step(self, action):
        # 1. 处理动作
        # 假设 action 是归一化的 [-1, 1]，我们需要将其缩放到实际的控制灵敏度
        # 参考 XboxController 的灵敏度
        delta_trans = action[:3]
        delta_rot = action[3:6] 
        delta_gripper = action[6]

        real_action = np.concatenate([delta_trans, delta_rot, [delta_gripper]])

        # 2. 更新控制器目标
        self.controller.update_target_pose(real_action)
        self.controller.update_target_gripper(real_action)
        target_pose = self.controller.get_target_pose()

        # 3. 执行 IK 和 控制 (核心逻辑复用 main.py)
        # import time
        # start_time = time.perf_counter()
        self._apply_ik_control(target_pose, steps=100) # 这里的 steps 是仿真步数
        # print(f"ik 时间：{time.perf_counter() - start_time}")

        # 4. 获取观测
        obs = self._get_obs()

        # 5. 计算抓取诊断量。state 观测不公开 target_pose，因此奖励
        # 计算时在内部补全。shaped reward 仅保留在 info 中用于调试，
        # 环境对智能体返回的奖励在连续抓取确认后统一为 0/1。
        reward_obs = obs
        if "target_pose" not in reward_obs:
            reward_obs = {**reward_obs, "target_pose": self._get_target_pose()}
        shaped_reward, reward_info = self.compute_reward(reward_obs, action)

        # 6. 检查终止条件 (连续成功判定)
        if reward_info["is_success"]:
            self._success_counter += 1
        else:
            self._success_counter = 0

        grasp_confirmed = self._success_counter >= self.success_consec_threshold
        terminated = grasp_confirmed                                # 连续 N 步成功 → terminated
        truncated = reward_info["dist"] > self.fail_dist_threshold  # 物体飞太远 → truncated
        reward = self._sparse_reward(grasp_confirmed)

        info = {
            "distance": reward_info["dist"],
            "is_success": grasp_confirmed,
            "success_counter": self._success_counter,
            "shaped_reward": shaped_reward,
            "left_contact": reward_info["left_contact"],
            "right_contact": reward_info["right_contact"],
            "r_reach": reward_info["r_reach"],
            "r_align": reward_info["r_align"],
            "r_contact": reward_info["r_contact"],
            "r_success": reward_info["r_success"],
            "r_action": reward_info["r_action"],
        }

        # 渲染
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()

        return obs, reward, terminated, truncated, info

    # ===== 抓取检测与奖励计算 =====

    @staticmethod
    def _sparse_reward(grasp_confirmed):
        """Return the terminal binary reward used by all grasp tasks."""
        return 1.0 if bool(grasp_confirmed) else 0.0

    def _check_gripper_target_contacts(self):
        """
        检查夹爪两侧手指的 pad 是否分别与目标物体发生接触。
        通过遍历 MuJoCo data.contact 实现，无需额外 XML touch sensor。

        Returns:
            (finger_A_contact, finger_B_contact): 左右手指是否分别与目标物体接触
        """
        finger_A_contact = False  # 左手指 (left_left_pad)
        finger_B_contact = False  # 右手指 (left_right_pad)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # 检查接触对中是否包含目标物体 geom
            if geom1 in self.target_geom_ids:
                other = geom2
            elif geom2 in self.target_geom_ids:
                other = geom1
            else:
                continue

            # 判断接触的是哪侧手指
            if other in self.finger_A_pad_ids:
                finger_A_contact = True
            elif other in self.finger_B_pad_ids:
                finger_B_contact = True

            # 两侧都有接触，提前退出
            if finger_A_contact and finger_B_contact:
                break

        return finger_A_contact, finger_B_contact

    def is_success(self):
        """
        判断抓取任务是否成功。
        三重条件:
          1. 夹爪双侧 pad 均与目标物体接触
          2. 夹爪执行器已发出闭合指令 (target_gripper > 阈值)
          3. 物体 6D 速度足够小 (排除偶然碰撞)
        """
        finger_A_contact, finger_B_contact = self._check_gripper_target_contacts()
        gripper_closed = self.controller.target_gripper > self.grasp_gripper_threshold
        target_vel = self.data.qvel[self.target_dof_adr:self.target_dof_adr + 6]
        target_stable = np.linalg.norm(target_vel) < self.target_stable_vel_threshold

        return finger_A_contact and finger_B_contact and gripper_closed and target_stable

    def compute_reward(self, obs, action):
        """
        计算抓取过程诊断量和仅用于日志的 shaped reward。

        Gym ``step`` 不会将这里的 shaped reward 返回给智能体；它只在连续
        抓取确认的终止步返回 1.0，其余步骤返回 0.0。

        阶段1 - 接近: pinch site 到目标物体的距离惩罚 (始终生效)
        阶段2 - 对齐: 末端姿态与目标姿态角度差惩罚 (距离 < 阈值时生效)
        阶段3 - 抓取: 接触 bonus (单侧/双侧) + 成功 bonus
        额外  - 动作惩罚: 抑制过大动作幅度

        Returns:
            (shaped_reward, reward_info_dict)
        """
        w = self.reward_weights

        # --- 阶段1: 接近奖励 (始终生效) ---
        pinch_pos = self.data.site_xpos[self.pinch_site_id]
        target_pos = obs["target_pose"][:3]
        dist = np.linalg.norm(pinch_pos - target_pos)
        r_reach = -w["reach"] * dist

        # --- 阶段2: 姿态对齐奖励 (距离较近时生效) ---
        r_align = 0.0
        if dist < self.align_dist_threshold:
            ee_quat = obs["ee_pose"][3:]
            target_quat = obs["target_pose"][3:]
            dot_product = np.clip(np.dot(ee_quat, target_quat), -1.0, 1.0)
            rot_dist = 2.0 * np.arccos(np.abs(dot_product))
            r_align = -w["align"] * rot_dist

        # --- 阶段3: 接触与抓取奖励 ---
        finger_A_contact, finger_B_contact = self._check_gripper_target_contacts()
        r_contact = 0.0
        if finger_A_contact or finger_B_contact:
            r_contact += w["contact_single"]    # 单侧接触 bonus
        if finger_A_contact and finger_B_contact:
            r_contact += w["contact_dual"]      # 双侧接触额外 bonus

        # 成功判断 (复用接触检测结果，避免重复遍历 data.contact)
        gripper_closed = self.controller.target_gripper > self.grasp_gripper_threshold
        target_vel = self.data.qvel[self.target_dof_adr:self.target_dof_adr + 6]
        target_stable = np.linalg.norm(target_vel) < self.target_stable_vel_threshold
        success = finger_A_contact and finger_B_contact and gripper_closed and target_stable

        r_success = w["success_bonus"] if success else 0.0

        # --- 动作幅度惩罚 ---
        r_action = -w["action_penalty"] * float(np.linalg.norm(action) ** 2)

        # 总奖励
        reward = r_reach + r_align + r_contact + r_success + r_action

        reward_info = {
            "dist": dist,
            "r_reach": r_reach,
            "r_align": r_align,
            "r_contact": r_contact,
            "r_success": r_success,
            "r_action": r_action,
            "left_contact": finger_A_contact,
            "right_contact": finger_B_contact,
            "is_success": success,
        }

        return reward, reward_info

    def _apply_ik_control(self, target_pose, steps=100):
        """
        封装 main.py 中的 IK 计算和仿真步进逻辑
        """
        # 准备 Pinocchio 需要的状态向量 (处理四元数顺序)
        sim_q = self.data.qpos.copy()
        current_q_pin = np.zeros(13)
        current_q_pin[:3] = sim_q[:3]
        current_q_pin[3:6] = sim_q[4:7]
        current_q_pin[6] = sim_q[3] # w (MuJoCo w 在前，Pinocchio w 在后)
        current_q_pin[7:] = sim_q[7:13]

        # 转换目标位姿
        rotation_matrix = self.kinematics.euler_to_rotation_matrix(target_pose[3], target_pose[4], target_pose[5])
        target_pose_se3 = self.kinematics.create_target_pose(rotation_matrix, np.array([target_pose[0], target_pose[1], target_pose[2]]))

        # 计算 IK —— 使用浮动基座模式
        new_q, _ = self.kinematics.ik(current_q_pin, target_pose_se3, steps=20)
        
        # 更新控制信号
        self.controller.update_control(new_q[7:])

        # 仿真步进 (Frame Skip)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data) # type: ignore
        
    def _init_target(self):
        """Randomize the target pose using the configured XYZ ranges."""
        target_qpos_addr = self.model.jnt_qposadr[self.target_joint_id]
        target_position = np.array(
            [
                self.np_random.uniform(*self.target_init_range[axis])
                for axis in ("x", "y", "z")
            ],
            dtype=float,
        )
        self.data.qpos[target_qpos_addr:target_qpos_addr + 3] = target_position
        self.data.qpos[target_qpos_addr + 3:target_qpos_addr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0]
        )

    def _get_obs(self):
        # 获取机械臂关节角 (后6个)
        joint_pos = self.data.qpos[7:13]
        # 获取基座姿态 (前7个: 3位置 + 4四元数)
        base_pose = self.data.qpos[:7]
        
        # 获取末端位置和姿态,四元数顺序为w, x, y, z
        ee_pos = self.data.site_xpos[self.ee_site_id]
        ee_mat = self.data.site_xmat[self.ee_site_id].reshape(9)
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, ee_mat) # type: ignore
        ee_pose = np.concatenate([ee_pos, ee_quat])
        
        images = (
            self.sensor.update_camera_view(viewer_mode="human")
            if self.sensor is not None
            else {}
        )
        gripper_min, gripper_max = self.gripper_joint_range
        gripper_pos = np.array(
            [
                np.clip(
                    (
                        self.data.qpos[self.gripper_qpos_adr] - gripper_min
                    )
                    / (gripper_max - gripper_min),
                    0.0,
                    1.0,
                )
            ],
            dtype=np.float32,
        )

        observations = {
            "joint_pos": joint_pos,
            "base_pose": base_pose,
            "ee_pose": ee_pose,
        }
        if self.observation_mode == "state":
            observations["gripper_pos"] = gripper_pos
        else:
            # 目标物体四元数顺序为 w, x, y, z。
            observations["target_pose"] = self._get_target_pose()
            observations.update(images)
        return observations

    def _get_target_pose(self):
        target_pos = self.data.xpos[self.target_body_id]
        target_quat = self.data.xquat[self.target_body_id]
        return np.concatenate([target_pos, target_quat])

    def render(self):
        if self.render_mode == "rgb_array" and self.sensor is not None:
            return self.sensor.renderer.render()
        
    def close(self):
        if self.viewer:
            self.viewer.close()
        if self.sensor is not None:
            self.sensor.close()

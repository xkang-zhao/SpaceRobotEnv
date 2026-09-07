lerobot 0.4.2 pi0 transformers

pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"

myvla need the leastest version 5.4.0

## mujoco_robot解读

位置与世界坐标系不对应的问题：
这是由于基座坐标系与世界坐标系是不对齐的，所以修改xml文件的base的quat为全0

隐式积分器不是很稳定，所以选择RK4积分器，时间步长改为0.0005s，选择一个合适的步长。

在设置红色的cube的时候，下面坐标有穿模，提高一下z轴坐标即可。

添加卫星基座后，确实可以自动做动力学计算模拟，动关节，基座发生变化。
问题在于如何采集数据

没有完成的：
- 使用xbox控制关节变量，现在gui界面拖动
- 安全没有逆运动学计算，因为不知道末端执行器相对于谁来求

2025-12-31 下午

添加了获取body，joint信息，初始化关节位置等共功能

代码块

```python
if not hasattr(self, '_debug_printed'):
    print("\n" + "="*60)
    print("MuJoCo 模型关节信息:")
    print("="*60)
     for i in range(self.model.nq):
        joint_id = self.model.jnt_qposadr[i] if i < self.model.njnt else -1
        joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) if i < self.model.njnt else "N/A"
        print(f"qpos[{i}]: {joint_name}")
    print("="*60 + "\n")
    self._debug_printed = True
quit()
```


```python
    mujoco.mj_forward(self.model, self.data)
    base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chasersat")
    print(f"Base ID: {base_id}")
    base_pos = self.data.body(base_id).xpos
    base_quat = self.data.body(base_id).xquat
    print(f"Base Position: {base_pos}, Base Quaternion: {base_quat}")

```

2026-1-29

问题1：写的快速验证的框架如何与lerobot的lerobot-record的代码联系起来，这个代码没有dataset的记录
问题2：再次看懂ik的代码，然后考虑是否能够添加轨迹规划功能，从而实现自动采集
问题3：以上测试完成后，要考虑完整的两臂机械臂的mjcf的建模，考虑好质量问题，动作空间问题

还有学习一下如何使用ai写代码的时候，加入仓库代码参考

如何把mujoco的仿真包装成gym风格，step()接


这个文件夹包含一个 MuJoCo 仿真环境，用于模拟 UR10e 机器人手臂和 2F-85 夹爪。它包含以下几个主要部分：

main.py: 主程序文件，负责初始化 MuJoCo 模型、设置控制器、运行仿真循环等。
model/scene.xml: MuJoCo 场景描述文件，定义了仿真环境中的物体、光源、相机等。
model/ur10e_2f85_wo_gripper.xml: MuJoCo 机器人模型文件，描述了 UR10e 机械臂的结构、关节、连杆等。
robot_viewer.py: (未提供) 用于可视化 MuJoCo 仿真的类。
xbox_controller.py: 用于处理 Xbox 控制器输入的类。
ik.py: (未提供) 包含运动学计算，特别是逆运动学 (IK) 求解器的类。
robot_controller.py: (未提供) 用于控制机器人运动的类。
robot_sensor.py: (未提供) 用于处理传感器数据的类。
README.md: 包含项目说明和一些待办事项

# 逆运动学解释

[开始: ik 函数]
  |
  v
[调用 solve_trajectory_ik]
  |
  +--> [1. 初始化]
  |      - 复制 q_init 到 q
  |      - 获取末端 Frame ID
  |      - 执行正向运动学 (FK) 获取当前末端位姿 start_pose
  |
  +--> [2. 主插值循环 (i 从 1 到 steps)]
  |      |
  |      +-- (A) 计算插值系数: alpha = i / steps
  |      |
  |      +-- (B) 目标位姿插值:
  |      |      - 位置: 线性插值 (Lerp)
  |      |      - 姿态: 球面线性插值 (Slerp) -> 生成 target_pose_i
  |      |
  |      +-- (C) 子迭代循环 (执行 5 次):
  |             |
  |             +-- 调用 step_ik(q, target_pose_i, ...)
  |                 |
  |                 +-- [step_ik 内部逻辑]:
  |                     1. FK 更新: 计算当前位姿 curr_pose
  |                     2. 误差计算: error = [位置误差; 旋转向量误差]
  |                     3. 计算广义雅可比 J_g:
  |                        J_g = J_m - J_b * inv(M_bb) * M_bm (考虑基座反向漂移)
  |                     4. 阻尼最小二乘求解:
  |                        v_m = inv(J_g.T * J_g + damping^2 * I) * J_g.T * error
  |                     5. 动量耦合计算基座速度: v_b = -inv(M_bb) * M_bm * v_m
  |                     6. 安全限幅: 若速度过大则等比例缩放
  |                     7. 状态积分: q_next = integrate(q, v_total * dt)
  |                     8. 返回 q_next 和误差范数
  |
  +--> [3. 精调阶段 (最多 200 次迭代)]
  |      |
  |      +-- 检查误差: 若 err < 1e-3 则提前跳出
  |      |
  |      +-- 调用 step_ik: 使用低阻尼 (damping=0.01) 以提高最终精度
  |
  +--> [4. 收敛检查]
  |      - 若 err > 1e-3，输出警告 (未完全收敛)
  |
  v
[返回 q_final, final_err]

1 轨迹插值 (Trajectory Interpolation): 算法不是直接从起点跳到终点，而是构造了一系列中间目标点。这样做可以避免数值求解器因为目标过远而发散，确保机械臂沿着平滑的路径运动。

2 广义雅可比 (Generalized Jacobian, $J_g$): 这是针对空间机器人的特殊处理。公式 $J_g = J_m - J_b M_{bb}^{-1} M_{bm}$ 考虑了“动量守恒”：当机械臂向左动时，浮动基座会因为反作用力向右漂移。$J_g$ 描述了关节运动对末端在世界坐标系下产生的净影响。

3 阻尼最小二乘 (Damped Least Squares): 在求解关节速度时加入了 damping 项。这能防止机械臂在接近奇异位姿（Singularity）时产生无穷大的关节速度，保证了数学上的稳定性。
4 精调 (Fine-tuning): 在完成大步长的轨迹追踪后，使用更小的阻尼进行原地迭代，目的是消除插值过程中累积的微小残差，达到毫米级的精度。





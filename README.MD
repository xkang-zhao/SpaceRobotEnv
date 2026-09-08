# SpaceUR10e MuJoCo Simulation

这是一个基于 MuJoCo 和 Gymnasium 的双臂空间 UR10e 仿真项目。当前主要支持：

- 键盘遥操作；
- Cube、卫星和空间碎片抓取环境；
- 8 个已标定的自动抓取任务；
- LeRobot 数据集采集；
- LeRobot 策略的本地或 ZMQ 分离式评估。

## 目录结构

```text
.
├── mjcf/                         # MuJoCo 场景、机器人模型和网格
├── scripts/
│   ├── auto_collect_dataset/    # LeRobot v2.1 自动采集入口
│   ├── evals/lerobot/           # LeRobot 策略评估
│   ├── keyboard/                # 键盘遥操作
│   ├── auto_grasp.py            # 统一自动抓取入口
│   ├── collect_dataset_keyboard_lerobot.py
│   ├── get_offset.py
│   └── get_offset_mac.py
├── src/
│   ├── data_collection/         # v2.1 writer 和多进程采集实现
│   ├── envs/                    # Gymnasium 环境注册与实现
│   ├── mujoco_robot/            # 控制器、IK、传感器和 Viewer
│   └── planners/auto_grasp/     # 自动抓取 planner 与任务配置
├── test/                        # 单元测试和集成测试
└── data/local/                  # 本地数据集，不应手动删除
```

## 安装

建议使用 Conda 管理 Pinocchio 和 Python 环境。下面以 Python 3.10 为例：

```bash
conda create -n robot python=3.10
conda activate robot

# macOS、Windows 以及遇到 pip ABI 问题的 Linux 环境，优先使用 conda-forge。
# conda install -c conda-forge pinocchio

pip install -r requirements.txt
pip install -e .
pip install pin
# pip installlerobot==0.4.4
```

## 环境

环境在 `src/envs/__init__.py` 中注册。所有环境共用
`src/envs/space_ur10e_env.py` 的控制、观测、奖励和成功判定逻辑，仅场景模型、
目标物体和随机初始范围不同。

| Gymnasium ID | 场景 | 抓取对象 | 随机初始位置范围 `(m)` |
| --- | --- | --- | --- |
| `SpaceUR10e-Cube-v0` | `mjcf/1_cube_scene.xml` | 红色 Cube | X `[1.50, 1.70]`，Y `[-0.20, 0.20]`，Z `[2.60, 2.70]` |
| `SpaceUR10e-Satellite-v0` | `mjcf/2_satellite_scene.xml` | Satellite | X `[1.80, 1.95]`，Y `[-0.10, 0.10]`，Z `[2.65, 2.70]` |
| `SpaceUR10e-Satellite2-v0` | `mjcf/3_satellite_2_scene.xml` | Satellite2 | X `[2.65, 2.80]`，Y `[-0.10, 0.10]`，Z `[2.65, 2.70]` |
| `SpaceUR10e-Satellite3-v0` | `mjcf/4_satellite3_scene.xml` | Satellite3 | X `[2.15, 2.30]`，Y `[-0.10, 0.10]`，Z `[2.65, 2.70]` |
| `SpaceUR10e-DebrisAntennaPanel-v0` | `mjcf/5_debris_antenna_panel_scene.xml` | 天线板碎片 | X `[1.85, 2.00]`，Y `[-0.10, 0.10]`，Z `[2.65, 2.70]` |
| `SpaceUR10e-DebrisTruss-v0` | `mjcf/6_debris_truss_scene.xml` | 桁架碎片 | X `[1.50, 1.65]`，Y `[-0.10, 0.10]`，Z `[2.65, 2.70]` |

### 通用动作与观测

动作是 7 维向量：

```text
[dx, dy, dz, droll, dpitch, dyaw, gripper]
```

通用数值观测：

| Key | Shape | 含义 |
| --- | --- | --- |
| `joint_pos` | `[6]` | 左臂 6 个关节位置 |
| `base_pose` | `[7]` | 机器人基座位置和 `wxyz` 四元数 |
| `ee_pose` | `[7]` | 左臂末端位置和 `wxyz` 四元数 |
| `target_pose` | `[7]` | 抓取目标位置和 `wxyz` 四元数 |

默认 RGB 相机均输出 `[480, 640, 3]` 的 `uint8` 图像：

```text
third_left_camera
third_right_camera
left_wrist_camera
```

开启深度渲染时会额外提供 `left_wrist_camera_depth`。即使不创建交互式
Viewer，`RobotSensor` 仍会创建 MuJoCo 离屏渲染器并产生相机观测。

### 控制频率与成功条件

- MuJoCo timestep 为 `0.0005 s`；
- 每次 `env.step()` 执行 100 次物理步进；
- 每个控制周期对应 `0.05 s`，即 `20 Hz`；
- 自动采集不使用 `sleep`，仿真按机器允许的速度运行，但数据时间仍按
  `20 FPS` 记录；
- 通用成功判定要求双指接触、夹爪闭合命令大于 100、目标 6D 速度小于
  `0.1`，并连续满足 10 个控制周期。

## 快速开始（有显示器）

键盘遥操作：

```bash
python scripts/keyboard/gym_env_keyboard.py \
  --env SpaceUR10e-Cube-v0
```

运行一个默认自动抓取任务：

```bash
python scripts/auto_grasp.py cube
python scripts/auto_grasp.py satellite3_left_antenna_panel
```

自动采集一个 LeRobot v2.1 数据集：

```bash
python scripts/auto_collect_dataset/collect_autograb_lerobot_v21.py cube \
  --output-dir data/cube_lerobot_v21 \
  --episodes 10 \
  --num-envs 4
```

## 脚本索引

| 脚本 | 用途 |
| --- | --- |
| `scripts/keyboard/gym_env_keyboard.py` | 使用 pynput 和 MuJoCo Viewer 遥操作 |
| `scripts/keyboard/gym_env_keyboard_mac.py` | macOS 原生 Viewer 键盘遥操作 |
| `scripts/get_offset.py` | 遥操作并输出抓取标定偏移 |
| `scripts/get_offset_mac.py` | macOS 版抓取偏移标定 |
| `scripts/auto_grasp.py` | 统一运行 8 个自动抓取任务 |
| `scripts/collect_dataset_keyboard_lerobot.py` | 键盘采集 LeRobot 数据集 |
| `scripts/collect_dataset_keyboard_teleop.sh` | 键盘采集示例包装脚本 |
| `scripts/auto_collect_dataset/collect_autograb_lerobot_v21.py` | 多进程自动采集 LeRobot v2.1 |
| `scripts/auto_collect_dataset/collect_dataset.sh` | 一次采集全部 8 个 autograb 任务 |
| `scripts/evals/lerobot/evaluate_lerobot_success_rate.py` | 本地 LeRobot 策略成功率评估 |
| `scripts/evals/lerobot/lerobot_zmq_server.py` | ZMQ 策略推理服务端 |
| `scripts/evals/lerobot/lerobot_zmq_client.py` | ZMQ MuJoCo 环境客户端 |


## 自动采集全部 8 个任务
scripts/auto_collect_dataset/collect_dataset.sh

运行：

```bash
bash scripts/auto_collect_dataset/collect_dataset.sh
```

CLI 参数：无。脚本顶部集中提供三个主要配置：

| Shell 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EPISODES_PER_TASK` | `80` | 每个任务最终保存的 episode 数 |
| `NUM_ENVS` | `4` | 每个任务内部使用的并行 MuJoCo worker 数 |
| `OUTPUT_ROOT` | `data/local` | 8 个任务数据集的共同输出根目录 |

脚本将依次采集以下任务：

| 顺序 | 任务 | 输出子目录 |
| --- | --- | --- |
| 1 | `cube` | `cube_lerobot_v21` |
| 2 | `satellite_handle` | `satellite_handle_lerobot_v21` |
| 3 | `satellite_left_antenna_panel` | `satellite_left_antenna_panel_lerobot_v21` |
| 4 | `satellite2_left_truss_connection` | `satellite2_left_truss_connection_lerobot_v21` |
| 5 | `satellite3_left_antenna_panel` | `satellite3_left_antenna_panel_lerobot_v21` |
| 6 | `satellite3_upper_rod` | `satellite3_upper_rod_lerobot_v21` |
| 7 | `debris_antenna_panel` | `debris_antenna_panel_lerobot_v21` |
| 8 | `debris_truss` | `debris_truss_lerobot_v21` |

终端会输出当前任务序号、任务名、输出目录、开始时间、成功/失败和耗时。
单个任务失败后脚本会继续执行剩余任务，最后打印成功/失败汇总，并在存在
失败任务时返回非零退出码。

采集器默认不会覆盖已有的非空数据集目录。重新批量采集时，请设置新的
`OUTPUT_ROOT`，或者先自行处理已存在的目标目录。

Linux headless 示例：

```bash
MUJOCO_GL=egl \
  bash scripts/auto_collect_dataset/collect_dataset.sh
```


## 脚本参数说明

<details>
<summary><code>scripts/keyboard/gym_env_keyboard.py</code>：通用键盘遥操作</summary>

运行：

```bash
python scripts/keyboard/gym_env_keyboard.py \
  --env SpaceUR10e-Cube-v0
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | `SpaceUR10e-Cube-v0` | 要创建的 Gymnasium 环境 ID |

按键：

| 按键 | 功能 |
| --- | --- |
| `8 / 5` | 末端 X 正/负方向 |
| `6 / 4` | 末端 Y 正/负方向 |
| `7 / 9` | 末端 Z 正/负方向 |
| `1` | 切换夹爪开合 |
| `Esc` | 退出 |

脚本固定以 20 Hz 运行并创建 `human` Viewer。

</details>

<details>
<summary><code>scripts/keyboard/gym_env_keyboard_mac.py</code>：macOS 键盘遥操作</summary>

运行：

```bash
python scripts/keyboard/gym_env_keyboard_mac.py \
  --env SpaceUR10e-Satellite3-v0
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | `SpaceUR10e-Cube-v0` | 要创建的 Gymnasium 环境 ID |

按键与通用版本一致。该脚本通过 `mujoco.viewer.launch_passive()` 获取
macOS Viewer 的原生键盘状态，支持长按连续移动。使用前需要先点击 Viewer
窗口使其获得焦点。

</details>

<details>
<summary><code>scripts/get_offset.py</code>：抓取位置标定</summary>

运行：

```bash
python scripts/get_offset.py --env SpaceUR10e-Satellite-v0
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | `SpaceUR10e-Satellite-v0` | 要标定的 Gymnasium 环境 ID |

按键：

| 按键 | 功能 |
| --- | --- |
| `1` | 切换平移/旋转模式 |
| `8/5`、`6/4`、`7/9` | 控制当前模式下的 X/Y/Z |
| `D` | 打印世界坐标和目标 body 坐标系 offset，然后退出 |
| `Esc` | 不打印并退出 |

</details>

<details>
<summary><code>scripts/get_offset_mac.py</code>：macOS 抓取位置与姿态标定</summary>

运行：

```bash
python scripts/get_offset_mac.py --env SpaceUR10e-Satellite3-v0
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | `SpaceUR10e-Satellite-v0` | 要标定的 Gymnasium 环境 ID |

按键：

| 按键 | 功能 |
| --- | --- |
| `-` | 切换平移/旋转模式 |
| `8/5`、`6/4`、`7/9` | 控制当前模式下的 X/Y/Z |
| `D` | 打印位置 offset、四元数和相对 RPY 后退出 |
| `Esc` | 不打印并退出 |

该脚本使用 macOS 原生 Viewer 键盘处理并支持长按。

</details>

<details>
<summary><code>scripts/auto_grasp.py</code>：统一自动抓取入口</summary>

查看任务：

```bash
python scripts/auto_grasp.py --help
python scripts/auto_grasp.py TASK --help
```

任务与环境：

| `TASK` | 环境 | 抓取目标 |
| --- | --- | --- |
| `cube` | `SpaceUR10e-Cube-v0` | 红色 Cube |
| `satellite_handle` | `SpaceUR10e-Satellite-v0` | 卫星把手 |
| `satellite_left_antenna_panel` | `SpaceUR10e-Satellite-v0` | 卫星左天线翼板 |
| `satellite2_left_truss_connection` | `SpaceUR10e-Satellite2-v0` | Satellite2 左桁架连接处 |
| `satellite3_left_antenna_panel` | `SpaceUR10e-Satellite3-v0` | Satellite3 左天线翼板 |
| `satellite3_upper_rod` | `SpaceUR10e-Satellite3-v0` | Satellite3 上杆 |
| `debris_antenna_panel` | `SpaceUR10e-DebrisAntennaPanel-v0` | 天线板碎片 |
| `debris_truss` | `SpaceUR10e-DebrisTruss-v0` | 桁架碎片 |


默认运行：

```bash
python scripts/auto_grasp.py cube
python scripts/auto_grasp.py satellite_handle
python scripts/auto_grasp.py satellite_left_antenna_panel
python scripts/auto_grasp.py satellite2_left_truss_connection
python scripts/auto_grasp.py satellite3_left_antenna_panel
python scripts/auto_grasp.py satellite3_upper_rod
python scripts/auto_grasp.py debris_antenna_panel
python scripts/auto_grasp.py debris_truss
```

向量参数必须用引号包裹，例如：

```bash
python scripts/auto_grasp.py satellite3_left_antenna_panel \
  --grasp_offset "[-0.776456, 0.576061, -0.007349]"
```

</details>

<details>
<summary><code>scripts/collect_dataset_keyboard_lerobot.py</code>：键盘采集数据集</summary>

运行：

```bash
python scripts/collect_dataset_keyboard_lerobot.py \
  --env SpaceUR10e-Cube-v0 \
  --repo_id local/spaceur10e_keyboard_cube \
  --num_episodes 10 \
  --task "Grab the cube"
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--env` | `SpaceUR10e-Cube-v0` | 采集使用的 Gymnasium 环境 |
| `--repo_id` | `local/spaceur10e_keyboard_cube` | 数据集 ID；数据写入 `data/<repo_id>` |
| `--num_episodes` | `10` | 数据集最终 episode 数 |
| `--enable_rerun` | 关闭 | 启用 Rerun 可视化 |
| `--task` | `Grab the object` | 写入数据集的语言任务 |
| `--cam1_key` | `third_left_camera` | `observation.images.cam1` 的观测 key |
| `--cam2_key` | `third_right_camera` | `observation.images.cam2` 的观测 key |
| `--cam3_key` | `left_wrist_camera` | `observation.images.cam3` 的观测 key |

采集按键：

| 按键 | 功能 |
| --- | --- |
| `8/5`、`6/4`、`7/9` | 控制末端 XYZ |
| `1` | 切换夹爪开合 |
| `←` 或 `Backspace` | 丢弃当前 episode 并重置 |
| `→` | 将当前 episode 标记成功并保存 |
| `Esc` | 退出 |

该入口使用已安装 LeRobot 版本提供的 `LeRobotDataset.create()`，输出格式跟随
该版本的默认 writer。需要严格 v2.1 时应使用自动采集入口。

</details>

<details>
<summary><code>scripts/collect_dataset_keyboard_teleop.sh</code>：键盘采集示例</summary>

运行：

```bash
bash scripts/collect_dataset_keyboard_teleop.sh
```

CLI 参数：无。脚本内固定调用：

| 配置 | 值 |
| --- | --- |
| 环境 | `SpaceUR10e-Cube-v0` |
| 数据集 ID | `local/my_3cam_dataset_keyboard` |
| episode 数 | `2` |
| 任务 | `Grab the cube` |
| 相机 | `third_left_camera`、`third_right_camera`、`left_wrist_camera` |
| Rerun | 开启 |

如需修改配置，请直接编辑该 Shell 文件，或使用上面的 Python 入口传参。

</details>

<details>
<summary><code>scripts/auto_collect_dataset/collect_autograb_lerobot_v21.py</code>：自动采集 LeRobot v2.1</summary>

单 worker 示例：

```bash
python scripts/auto_collect_dataset/collect_autograb_lerobot_v21.py cube \
  --output-dir data/local/cube_lerobot_v21 \
  --episodes 10 \
  --instruction "Grab the red cube"
```

多 worker 示例：

```bash
python scripts/auto_collect_dataset/collect_autograb_lerobot_v21.py debris_truss \
  --output-dir data/local/debris_truss_lerobot_v21 \
  --episodes 40 \
  --num-envs 4
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `task_name` | 必填 | 与 `auto_grasp.py` 相同的 8 个任务名之一 |
| `--output-dir` | `data/local/<task>_lerobot_v21` | 数据集输出根目录 |
| `--episodes` | `10` | 最终保存的 episode 数 |
| `--num-envs` | `1` | 独立 spawn MuJoCo worker 数 |
| `--instruction` | 任务注册的默认指令 | 写入 `tasks.jsonl` 的语言指令 |
| `--seed` | `0` | 确定性 attempt seed 起点 |
| `--max-attempts-per-episode` | `10` | 每个目标 episode 的最大尝试次数 |
| `--camera NAME=OBS_KEY` | 三路默认 RGB | 相机 feature 与环境观测 key 映射，可重复 |
| `--keep-failed` | 关闭 | 允许失败 attempt 占用一个 episode |
| `--overwrite` | 关闭 | 新数据集完整生成后事务式替换旧目录 |
| `--render` | 关闭 | 显示 Viewer，只允许 `--num-envs 1` |
| `--enable-rerun` | 关闭 | 启用 Rerun，只允许 `--num-envs 1` |
| `--debug-planner` | 关闭 | 输出 planner 调试信息 |

默认相机映射：

```text
cam1=third_left_camera
cam2=third_right_camera
cam3=left_wrist_camera
```

只选择两个相机：

```bash
python scripts/auto_collect_dataset/collect_autograb_lerobot_v21.py debris_truss \
  --episodes 40 \
  --num-envs 4 \
  --camera cam1=third_left_camera \
  --camera cam3=left_wrist_camera
```

约束：

- 采集器默认只保存环境确认成功的 episode；
- 每帧严格对应一次 `env.step()`，时间戳为 `frame_index / 20`；
- worker 使用 `spawn`，每个进程拥有独立 MuJoCo 和 OpenGL context；
- 输出先写入 staging 目录，完整校验后才安装到目标目录；
- 目标目录非空时必须显式使用 `--overwrite`；
- 不支持向已有 v2.1 数据集追加；
- 多 worker 模式不能使用 Viewer 或 Rerun。

</details>



<details>
<summary><code>scripts/evals/lerobot/evaluate_lerobot_success_rate.py</code>：本地策略评估</summary>

> 该评估入口仍需要针对实际 checkpoint 完成端到端验收。

单进程示例：

```bash
python scripts/evals/lerobot/evaluate_lerobot_success_rate.py \
  --model-path /path/to/pretrained_model \
  --policy-type pi0 \
  --env-id SpaceUR10e-Cube-v0 \
  --num-episodes 30 \
  --max-steps-per-episode 500 \
  --action-chunk-size 1
```

多进程示例：

```bash
python scripts/evals/lerobot/evaluate_lerobot_success_rate.py \
  --model-path /path/to/pretrained_model \
  --policy-type pi0 \
  --num-episodes 30 \
  --num-envs 4 \
  --use-multiprocessing \
  --render-mode rgb_array
```

`human` 渲染不能和多进程评估一起使用。

</details>

<details>
<summary><code>scripts/evals/lerobot/lerobot_zmq_server.py</code>：ZMQ 策略服务端</summary>

运行：

```bash
python scripts/evals/lerobot/lerobot_zmq_server.py \
  --model-path /path/to/pretrained_model \
  --policy-type pi0 \
  --bind tcp://127.0.0.1:5555 \
  --action-chunk-size 8
```

</details>

<details>
<summary><code>scripts/evals/lerobot/lerobot_zmq_client.py</code>：ZMQ MuJoCo 客户端</summary>

运行：

```bash
python scripts/evals/lerobot/lerobot_zmq_client.py \
  --connect tcp://127.0.0.1:5555 \
  --env-id SpaceUR10e-Cube-v0 \
  --render-mode human \
  --num-episodes 5
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--connect` | `tcp://127.0.0.1:5555` | ZMQ REQ 服务端地址 |
| `--env-id` | `SpaceUR10e-Cube-v0` | 客户端运行的 MuJoCo 环境 |
| `--render-mode` | `human` | `human`、`rgb_array` 或 `none` |
| `--scene-path` | 无 | 可选 MuJoCo scene XML 覆盖 |
| `--arm-path` | 无 | 可选 IK robot XML 覆盖 |
| `--frame-name` | 无 | 可选 IK frame 覆盖 |
| `--no-depth` | 关闭 | 禁用深度观测；默认启用 |
| `--task` | 无 | 覆盖服务端默认语言任务 |
| `--robot-type` | 无 | 覆盖服务端默认 robot type |
| `--action-chunk-size` | 无 | 覆盖服务端默认 chunk 大小 |
| `--num-episodes` | `1` | 客户端运行的 episode 数 |
| `--max-steps-per-episode` | `500` | 每个 episode 最大控制步数 |
| `--seed` | 无 | 可选确定性基础 seed |

客户端负责环境 reset/step、发送观测、执行服务端返回的整个动作块并统计成功率。

</details>


## 自动抓取默认标定

这些值是 planner 当前实际使用的默认值，不等同于标定工具输出的原始
末端位姿。

| 任务 | body/world offset | 姿态模式 |
| --- | --- | --- |
| `cube` 前置点 | `[-0.12, 0, 0]`，世界坐标系 | 保持 Cube planner 姿态 |
| `cube` 抓取点 | `[-0.02, 0, 0]`，世界坐标系 | 保持 Cube planner 姿态 |
| `satellite_handle` | `[-0.600, 0.223, 0.004]` | Handle planner |
| `satellite_left_antenna_panel` | `[-0.265751, 0.544939, 0.0]` | absolute，tool Z `π/2` |
| `satellite2_left_truss_connection` | `[-1.267780, 0.476677, 0.426568]` | skip rotation |
| `satellite3_left_antenna_panel` | `[-0.776456, 0.576061, -0.007349]` | absolute，tool Z `π/2` |
| `satellite3_upper_rod` | `[-0.678392, 0.003160, 0.218000]` | skip rotation |
| `debris_antenna_panel` | `[-0.468390, 0.127419, 0.132495]` | relative local Z `-0.61` |
| `debris_truss` | `[-0.136453, 0.075573, 0.118878]` | relative local Z `-0.8` |

<details>
<summary>历史标定原始记录</summary>

标定方法：遥操作末端到目标抓取位置，按 `D` 输出目标中心、末端位置、
body-frame offset 和相对姿态。planner 可能基于实际抓取稳定性进一步微调
原始 offset。

</details>

## 平台说明

### Linux

无显示服务器的离屏采集或评估建议设置：

```bash
export MUJOCO_GL=egl
```

### macOS

- 优先使用 `gym_env_keyboard_mac.py` 和 `get_offset_mac.py`；
- 如果普通 Python 无法创建 CoreGraphics context，使用 `mjpython`；
- 多进程采集必须使用 `spawn`，每个 worker 创建独立的 CGL context。

### Windows

如果遇到 OpenMP 重复加载问题，可以设置：

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
```


## 机器人模型摘要

```text
空间机器人外形尺寸: 1.75 x 1.75 x 1.75 m
卫星本体质量:      200 kg
单条 UR10e:        32.7 kg
单个 2F85 夹爪:    0.9 kg
基座中心:          (0, 0, 2)
初始末端目标:      (1.2, 0.1, 2.7, 1.57, 0.0, 1.57)
```

# SpaceUR10e 仓库协作指南

## 1. 项目定位

本仓库是基于 MuJoCo 与 Gymnasium 的双臂空间 UR10e 仿真项目，左臂末端安装 Robotiq 2F85 夹爪。当前核心工作流包括：

- 键盘遥操作与抓取点标定；
- Cube、卫星和空间碎片抓取环境；
- 8 个标定后的自动抓取任务；
- LeRobot v2.1 数据集采集；
- LeRobot 策略的本地评估与 ZMQ 分离式评估；
- 抓取、IK 和数据采集结果的分析与可视化。

`README.MD` 面向使用者；本文件面向在仓库中分析、修改和验证代码的代理。

## 2. 开始工作前

1. 先阅读与任务直接相关的模块、测试和脚本入口，不要只根据 README 推断行为。
2. 运行 `git status --short`，保留用户已有改动；不要还原、覆盖或顺手整理无关文件。
3. 所有命令默认从仓库根目录运行。环境注册仍使用 `./mjcf/...` 相对路径，从其他目录启动可能找不到场景或机械臂 XML。
4. 优先做最小范围修改。除非任务明确要求，不要同时重构环境、planner、数据格式和脚本接口。
5. 修改前用 `rg` 查找调用点、测试和文档引用；新增实现前先确认仓库内没有可复用逻辑。

## 3. 目录与职责

```text
.
├── mjcf/                         # MuJoCo 场景、机器人 XML 与网格资产
├── scripts/
│   ├── keyboard/                # 键盘遥操作入口
│   ├── auto_collect_dataset/    # 自动抓取数据采集入口
│   ├── evals/lerobot/           # 本地与 ZMQ 策略评估
│   ├── auto_grasp.py            # 8 个自动抓取任务的统一 CLI
│   ├── collect_dataset_keyboard_lerobot.py
│   ├── get_offset.py
│   └── get_offset_mac.py
├── src/
│   ├── envs/                    # Gymnasium 注册及共享环境实现
│   ├── mujoco_robot/            # IK、控制、传感器与 Viewer
│   ├── planners/auto_grasp/     # planner、任务注册、标定 profile 与 CLI
│   ├── data_collection/         # 自动采集循环及 LeRobot v2.1 writer
│   └── utils/                   # 辅助和历史集成代码
├── reports/                     # 分析、绘图和录制工具及结果
├── test/                        # unittest 风格的单元与集成测试
├── data/                        # 用户采集的数据集，通常体积较大
└── assets/                      # 文档和报告使用的图片、GIF 与视频
```

关键模块：

- `src/envs/__init__.py`：6 个 Gymnasium 环境的唯一注册位置，定义场景、目标初始范围和 Viewer 配置。
- `src/envs/space_ur10e_env.py`：共享的动作、观测、reset、控制、奖励、终止与成功判定。
- `src/mujoco_robot/robot_ik.py`：Pinocchio IK，依赖和坐标系最敏感。
- `src/mujoco_robot/robot_controller.py`：维护末端与夹爪目标并写入 MuJoCo control。
- `src/mujoco_robot/robot_sensor.py`：RGB/深度离屏相机观测。
- `src/planners/auto_grasp/tasks.py`：自动抓取任务的公共注册表与采集适配器。
- `src/planners/auto_grasp/profiles.py`：6 个 lateral 任务的标定参数；标定值应与 planner 算法分离。
- `src/data_collection/autograb_v21.py`：多 worker episode 采集与重试逻辑。
- `src/data_collection/lerobot_v21.py`：Parquet、视频和 metadata 的 v2.1 写入与发布逻辑。

## 4. 当前公开接口

### Gymnasium 环境

环境注册在 `src/envs/__init__.py`：

- `SpaceUR10e-Cube-v0`
- `SpaceUR10e-Satellite-v0`
- `SpaceUR10e-Satellite2-v0`
- `SpaceUR10e-Satellite3-v0`
- `SpaceUR10e-DebrisAntennaPanel-v0`
- `SpaceUR10e-DebrisTruss-v0`

新增或改名环境时，至少同步检查：环境注册、对应 MJCF、自动抓取 profile、CLI 测试、目标随机范围测试和 README。

### 自动抓取任务

统一入口：

```bash
python scripts/auto_grasp.py --help
python scripts/auto_grasp.py TASK --help
```

当前任务：

- `cube`
- `satellite_handle`
- `satellite_left_antenna_panel`
- `satellite2_left_truss_connection`
- `satellite3_left_antenna_panel`
- `satellite3_upper_rod`
- `debris_antenna_panel`
- `debris_truss`

任务名称和环境映射以 `src/planners/auto_grasp/tasks.py` 为准；lateral 任务的姿态、offset 和容差以 `profiles.py` 为准。增加任务时应让交互式运行和自动采集共用同一注册信息，不要在多个脚本中复制任务表。

## 5. 必须保持的仿真与数据约定

- 动作为 7 维：`[dx, dy, dz, droll, dpitch, dyaw, gripper]`。
- 姿态观测使用四元数时，先确认代码期望的是 `wxyz`，不要默认使用常见库的 `xyzw`。
- 环境控制频率为 20 Hz：MuJoCo timestep `0.0005 s`，每个 `env.step()` 执行 100 个物理步。
- `terminated` 表示确认成功；planner 的瞬时接触或内部完成状态不能替代环境连续成功判定。
- 环境成功需要连续满足条件 10 个控制周期。采集器只把 `terminated and info["is_success"]` 视为确认成功。
- reset 随机性必须由传入 seed 控制。多 worker 采集修改后要检查相同 seed 的可复现性和不同 episode 的 seed 分配。
- 相机 key、shape、dtype、FPS 或动作 shape 的变化属于数据格式变更，必须同步 writer、评估适配、测试和文档。
- 自动采集写入 staging 目录并在成功后发布。不要破坏“失败尝试不覆盖已发布数据集”的语义。
- MuJoCo body、site、joint、camera 和 actuator 名称是 Python 与 MJCF 之间的接口；修改 XML 名称必须全仓搜索消费者。

## 6. 常用命令

安装建议：

```bash
conda install -c conda-forge pinocchio
pip install -r requirements.txt
pip install -e .
```

macOS 上 Pinocchio 容易因 pip 的 `pin`、`cmeel-urdfdom` 和 `cmeel-tinyxml2` ABI 不匹配而导入失败。使用 conda-forge Pinocchio 后，不要再安装 pip `pin`。`requirements.txt` 中 LeRobot 当前也是可选依赖，不要为不涉及 LeRobot 的任务擅自改变依赖策略。

快速检查：

```bash
python -c "import pinocchio as pin; print(pin.__version__)"
python -c "import gymnasium as gym; import envs; print(gym.spec('SpaceUR10e-Cube-v0').id)"
python scripts/auto_grasp.py --help
```

运行与采集：

```bash
python scripts/keyboard/gym_env_keyboard.py --env SpaceUR10e-Cube-v0
python scripts/keyboard/gym_env_keyboard_mac.py --env SpaceUR10e-Satellite-v0
python scripts/auto_grasp.py cube
python scripts/auto_collect_dataset/collect_autograb_lerobot_v21.py cube \
  --output-dir data/cube_lerobot_v21 --episodes 10 --num-envs 4
bash scripts/auto_collect_dataset/collect_dataset.sh
```

Linux 无显示服务器时优先尝试：

```bash
MUJOCO_GL=egl python scripts/auto_grasp.py cube --render_mode none
```

macOS 创建 MuJoCo 图形上下文失败时，交互式命令可能需要使用 `mjpython`。不要把 GUI/显卡失败直接判断为 planner 逻辑失败。

## 7. 修改规则

### Python

- 遵循现有模块边界、类型注解和 dataclass 风格；公开行为变化应补测试。
- `scripts/` 只负责参数解析和编排，可复用业务逻辑放在 `src/`。
- 保持 `env.reset()` 返回 `(obs, info)`、`env.step()` 返回 Gymnasium 五元组。
- 不要仅为“清理”而重写历史兼容代码；先确认它是否仍被入口、测试或外部 LeRobot 插件使用。
- 错误信息应包含任务名、环境 ID、路径或 worker/episode 等可定位上下文。

### Planner 与标定

- 将通用状态机或几何算法放在 planner 模块，将任务特定 offset、旋转模式与容差放在 profile。
- 修改 frame、相对旋转或抓取 offset 时，明确它属于世界坐标、目标 body 坐标、末端坐标还是 pinch site 坐标。
- 不要通过放宽成功条件来掩盖 IK、碰撞、接触或标定问题。
- 超时、重试与成功保持步数会影响数据质量；修改时同时验证成功和失败路径。

### MJCF 与资产

- 尽量保持 include 路径相对且可从场景 XML 正确解析。
- 修改 collision geom、contact、actuator、equality、timestep 或 solver 参数属于高风险改动，应进行实际 MuJoCo smoke test。
- 不要无理由重新导出或格式化大型 STL/OBJ；二进制资产变化需要说明来源和尺度。

### 数据采集

- `data/`、`dataset/`、视频和 episode 文件视为用户产物。除非用户明确要求，不要删除、覆盖、移动或纳入提交。
- 不要手工编辑 Parquet、视频或已发布 metadata 来“修复”数据集；修复 writer 并用临时输出重新验证。
- 测试 writer 时使用临时目录和最小 episode，不要写入现有真实数据集。
- 多进程代码必须考虑 worker 清理、异常传播、确定性排序和部分失败。

### 生成文件

以下内容一般不应手工修改或提交：

- `__pycache__/`、`*.pyc`、`.pytest_cache/`；
- `src/*.egg-info/` 中的生成元数据；
- 用户采集的 Parquet、视频和数据集 metadata；
- 仅由报告脚本可重建的中间文件，除非任务明确要求更新报告产物。

## 8. 验证策略

优先运行与改动直接相关的最小测试，再根据风险扩大范围。测试以标准库 `unittest` 风格编写：

```bash
python -m unittest discover -s test -p 'test_envs_registration.py'
python -m unittest discover -s test -p 'test_auto_grasp_cli.py'
python -m unittest discover -s test -p 'test_auto_grasp_*.py'
python -m unittest discover -s test -p 'test_autograb_collection_v21.py'
python -m unittest discover -s test -p 'test_*.py'
```

建议映射：

| 改动范围 | 最低验证 |
| --- | --- |
| 环境注册、场景路径、初始范围 | `test_envs_registration`、`test_target_init_range` |
| 环境 step、奖励、成功条件 | 相关环境测试 + 一个实际 headless episode |
| IK、base/EE frame | `test_base_pose_consistency`、`test_ik_validation` + Pinocchio 导入检查 |
| 自动抓取 CLI 或任务表 | `test_auto_grasp_cli` + `auto_grasp.py --help` |
| cube/handle/lateral planner | 对应 planner 测试 + 目标任务 smoke test |
| LeRobot v2.1 writer/采集 | `test_autograb_collection_v21` + 临时目录最小采集 |
| 本地/ZMQ 评估 | `test_lerobot_eval_common`、`test_lerobot_zmq_protocol` |
| 报告或视频工具 | 对应测试 + 打开生成图片/视频检查 |

运行完整测试前注意：部分历史测试依赖 GUI、MuJoCo 上下文、Pinocchio、LeRobot 或外部控制器，并可能引用已移除的旧模块。若失败来自缺失可选依赖或图形环境，应准确报告“未运行/环境受限”，不要声称测试通过，也不要为让旧测试变绿而恢复废弃架构。

## 9. 完成标准

交付前确认：

- 改动只覆盖请求范围，没有覆盖用户原有工作；
- 新行为有对应测试，相关测试实际通过；
- 环境 ID、任务名、MJCF 名称、观测 key 和数据 schema 保持同步；
- 未改写或删除用户数据与大型生成产物；
- README 或脚本帮助在公开命令、参数或行为变化时已同步；
- 最终说明列出修改内容、验证命令及任何未验证的 GUI/硬件/可选依赖限制。

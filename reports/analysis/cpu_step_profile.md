# CPU 仿真 env.step 耗时分解

2026-09-08，在本机 Intel Xeon Silver 4316 @ 2.30 GHz 上运行 Cube 自动抓取。MuJoCo 3.12.0、Pinocchio 4.1.0、Gymnasium 1.3.0、NumPy 2.4.6。物理后端为 CPU MuJoCo；RGB 模式使用 EGL/NVIDIA A800 80GB 渲染，不能将 RGB 行理解为纯 CPU 软件渲染。

每种观测模式分别使用 seed 7、8、9，成功 episode 长度为 96、76、113，共 285 个 step。排除初始化、reset 和预热；不打开交互 Viewer，不限速、不 sleep。表中为实际墙钟平均耗时。

| 部分 | state，无相机 ms | RGB，三路相机 ms |
| --- | ---: | ---: |
| 动作处理：更新工具位姿、夹爪目标 | 0.082 | 0.086 |
| Pinocchio IK 求解 | 22.576 | 22.335 |
| IK 输入转换与 control 写入 | 0.028 | 0.030 |
| MuJoCo 100 个物理子步 | 3.576 | 3.586 |
| 状态观测提取与组装 | 0.044 | 0.064 |
| 相机更新、渲染与 RGB 回读 | 0 | 10.705 |
| 奖励诊断与接触检测 | 0.036 | 0.042 |
| 其他：连续成功计数、终止判定、info 组装、Gym 包装等 | 0.023 | 0.021 |
| **env.step 总计** | **26.365** | **36.868** |

- state 模式：IK 占 85.6%，物理步进占 13.6%，其他合计约 0.8%。总耗时 P95 为 28.45 ms。
- RGB 模式：IK 占 60.6%，相机合计占 29.0%，物理步进占 9.7%。总耗时 P95 为 43.52 ms。
- 三路相机分别为 `third_left_camera`、`third_right_camera`、`left_wrist_camera`，均为 640×480 RGB，没有深度。相机部分进一步分为场景更新 0.075 ms、渲染与回读 10.612 ms、循环与图像字典组装等 0.017 ms。这是三路合计，不是单路耗时。
- 接触检测自身平均约 0.005 ms，已包含在奖励诊断行内，不应重复相加。

一次 env.step 推进的仿真时间为 `100 × 0.0005 s = 50 ms`，与实际计算耗时不同。当前原生 CPU 路径每次调用 `mj_step` 一个物理子步，循环 100 次；表中保留这一真实环境路径，没有换成批量 rollout 或更快的特殊基准。

计时通过临时包装实际方法完成，嵌套阶段相减得到独占时间，并校验 570 个样本各部分之和等于总耗时。“其他”是剩余时间，包含未单独埋点的操作与少量计时包装成本，因此不应解读为纯终止判定耗时。各部分 P95 不能直接相加。

这里不包含 `env.step` 外的 planner、策略推理、视频编码、数据集写盘或人为实时等待。观察值提取包括状态及 RGB 数组；相机渲染虽在 GPU 执行，其等待和回读耗时仍计入 step 的墙钟时间。

优化判断：无相机时优先优化 IK；有视觉观测时 IK 和相机都是主要成本。当前单环境物理步进只占约 3.6 ms，即便完全消除，也无法大幅降低整个 step 的耗时。多 world GPU 基准测量的是物理吞吐，不能直接代替这里的端到端 step 耗时。

原始逐步数据与 mean/median/P95：[cpu_step_profile.json](cpu_step_profile.json)。

从仓库根目录，在具备现有 MuJoCo/Pinocchio 依赖的环境中复现：

```bash
python reports/analysis/profile_cpu_step.py --modes state rgb --seeds 7 8 9 \
  --output /tmp/cpu_step_profile.json
```

本次使用 `/tmp/spacerobot-warp-venv/bin/python` 执行。输出文件须不存在。脚本默认设置 `MUJOCO_GL=egl`（如果调用者已设置则保留）；其他机器的 CPU、图形后端、模型和依赖版本会影响结果。

# 优化后 CPU 单步耗时（2026-09-09）

当前代码的 Cube 抓取，Xeon Silver 4316，MuJoCo 3.12.0，Pinocchio 4.1.0。
state 和 rgb 分别运行 seed 7/8/9，各 285 个控制 step，6 个 episode 全部成功。
相机为三路 640×480 RGB，无深度；CPU 物理，EGL 渲染，图形设备使用 GPU。
排除预热、reset、planner/策略推理、数据写盘和 GUI；分项计时存在少量探针开销。

| 阶段 | state 均值(ms) | rgb 均值(ms) |
| --- | ---: | ---: |
| 动作目标更新 | 0.073 | 0.084 |
| IK 求解 | 1.977 | 1.999 |
| IK 坐标转换及写入控制 | 0.024 | 0.028 |
| 100 个 MuJoCo 物理子步 | 3.561 | 3.613 |
| 状态观测组装 | 0.034 | 0.060 |
| 相机场景更新、渲染与回读 | 0.000 | 5.617 |
| 奖励及接触检测 | 0.031 | 0.039 |
| 终止判定、info、Gym 包装等剩余开销 | 0.020 | 0.019 |
| 总计 | 5.721 | 11.458 |

总 step P95：state 6.207 ms，rgb 12.433 ms。

无相机时物理步进占 62.24%，IK 占 34.56%；开启相机后，相机合计约 49.03%，物理步进占 31.53%。
每 step 推进 50 ms 仿真时间，与表中的实际计算耗时不同。
接触检测已包含在奖励项内，不重复求和；所有 570 个样本均检查分项非负及分项之和等于总耗时。
这是 Cube 轨迹上的均值，其他任务可能因 IK 回退、碰撞和接触数量而不同。
不将跨轮次 RGB 耗时差异直接归因于 IK 优化；渲染和系统运行状态也会影响计时。

复现（输出文件必须不存在）：

```bash
MUJOCO_GL=egl /tmp/spacerobot-warp-venv/bin/python reports/analysis/profile_cpu_step.py --modes state rgb --seeds 7 8 9 --output /tmp/cpu_step_new.json
```

原始逐步样本：`cpu_step_profile_optimized.json`。

# IK 第三轮优化：小幅目标直接求解

默认 `Kinematics.ik(..., fast_path=True)` 在 `early_stop=True` 且 `steps >= 8` 时，
对混合位姿残差处于 `(1e-4, 0.02]` 的目标尝试直接求解。最多 40 次迭代，
阻尼保持 0.05，速度范数限制保持 0.2，内部积分步长为 0.2（原插值路径为 0.05）。
这会改变数值搜索路径，不改变 MuJoCo timestep 或环境控制频率。每 10 次检查
最终目标残差，必须下降且达到 1e-4 才接受；非有限数值、残差不降或预算耗尽时，
从原始配置重新运行第二轮的插值求解。回退会多花最多 40 次迭代，困难目标可能变慢。

`fast_path=False` 关闭快速路径，`early_stop=False` 同时关闭两种算法优化。
位置（米）和旋转向量（弧度）仍使用混合范数；没有放宽环境抓取成功条件。

## CPU 实测

Xeon Silver 4316，Pinocchio 4.1.0，MuJoCo 3.12.0，无相机、无 GUI。Cube seed
7/8/9 共 285 个控制 step，全部确认成功，步数仍为 96/76/113。排除预热、reset、
planner 和数据写盘。第二轮历史基准来自 ik_early_stop_profile.json。

| 平均耗时 | 第二轮后 | 第三轮后 |
| --- | ---: | ---: |
| IK | 12.229 ms | 1.960 ms |
| env.step | 15.960 ms | 5.679 ms |

完整 step 比第二轮减少约 64.4%，比最初 26.293 ms 减少约 78.4%。
这是 Cube 小动作轨迹的收益，不能外推到全部任务、RGB 或 GPU 批量环境。

## 8 个任务对照（seed 7）

均开启第二轮提前停止，仅切换 fast_path；成功要求 terminated 且 info.is_success。
两版均成功 7/8，satellite3_upper_rod 两版均因抓取未确认而失败。

| 任务 | 步数：关闭→开启 | IK 总迭代数：关闭→开启 | 开启后成功 |
| --- | --- | --- | --- |
| cube | 96 → 96 | 17610 → 2490 | 是 |
| satellite_handle | 79 → 79 | 13010 → 5190 | 是 |
| satellite_left_antenna_panel | 181 → 181 | 31820 → 26560 | 是 |
| satellite2_left_truss_connection | 117 → 117 | 21180 → 16640 | 是 |
| satellite3_left_antenna_panel | 157 → 157 | 28090 → 23610 | 是 |
| satellite3_upper_rod | 119 → 119 | 19350 → 8190 | 否 |
| debris_antenna_panel | 81 → 81 | 14170 → 10160 | 是 |
| debris_truss | 98 → 101 | 14870 → 9580 | 是 |

此对照发现的轨迹差异是算法优化的结果；单 seed 无法保证大规模采集成功率。
详细聚合数据见 ik_fast_path_profile.json。本轮未验证 RGB、GUI 和 GPU 批量运行。

## 验证命令

本次使用 /tmp/spacerobot-warp-venv/bin/python，从仓库根目录执行：

```bash
PYTHONPATH=src python -m unittest discover -s test -p 'test_ik_optimization.py'
PYTHONPATH=src python -m unittest discover -s test -p 'test_base_pose_consistency.py'
PYTHONPATH=src python -m unittest discover -s test -p 'test_ik_validation.py'
python reports/analysis/profile_cpu_step.py --modes state --seeds 7 8 9 --output /tmp/ik_fast_new.json
```

计时输出文件必须不存在。优化测试 11 项、基座测试 5 项、已有 IK 验证 8 项通过。
历史 test_curated_dataset_statistics 仍因缺少 dataset/ episode 数据报错。
本轮新增小目标残差与 40 次预算、停滞/非有限数值回退一致性、预算耗尽测试；
最后调整非有限数值检查位置后，这 3 项已重新运行通过。

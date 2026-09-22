# CPU IK 第一轮优化

本轮仅优化浮动基座 IK 的数值计算路径：用 `solve(M_bb, M_bm)`
计算并复用动量耦合矩阵，用 `solve(H, g)` 求关节速度，并将重复的正运动学
替换为关节 Jacobian 更新后的 frame placement 更新。基座矩阵求解失败时
回退到原来的伪逆。保留 `compute_matrices` 的返回接口供分析脚本使用。

插值点数、每点迭代数、阻尼、速度限制、精调停止条件和固定基座求解器均保持原值。
插值起点的可变引用和积分前误差语义留待单独处理，避免与本轮性能优化混合。

## 实测

同一机器 Xeon Silver 4316、Pinocchio 4.1.0、MuJoCo 3.12.0，CPU 物理、
state 观测、无 GUI。修改前后分别运行 seed 7/8/9 的 Cube 自动抓取，
每组 285 个控制 step。排除预热、reset、planner 和数据写盘。

| 指标 | 修改前 | 修改后 |
| --- | ---: | ---: |
| IK 平均耗时 | 22.567 ms | 13.243 ms |
| env.step 平均耗时 | 26.293 ms | 16.990 ms |
| env.step P95 | 27.603 ms | 17.969 ms |
| 抓取成功 | 3/3 | 3/3 |
| 各 seed 完成步数 | 96 / 76 / 113 | 96 / 76 / 113 |

IK 平均耗时下降约 41.3%，step 平均耗时下降约 35.4%（约 1.55 倍吞吐）。
这是有限样本的单环境测量，不代表其他任务、相机模式或 GPU 批量环境的收益。
聚合结果保存在 `ik_optimization_profile.json`。

## 验证与复现

以下命令从仓库根目录运行；本次使用 `/tmp/spacerobot-warp-venv/bin/python`。

```bash
PYTHONPATH=src python -m unittest discover -s test -p 'test_ik_optimization.py'
PYTHONPATH=src python -m unittest discover -s test -p 'test_base_pose_consistency.py'
PYTHONPATH=src python -m unittest discover -s test -p 'test_ik_validation.py'
python reports/analysis/profile_cpu_step.py --modes state --seeds 7 8 9 --output /tmp/ik_profile_new.json
```

输出文件必须不存在。新增测试 3 项通过：12 组配置的单步数值一致性及动量约束、
12 组完整求解的一致性、奇异基座矩阵的伪逆回退。单步配置容差 1e-11，完整求解
配置容差 1e-9；包含中立配置、旋转后的浮动基座和未完全收敛的目标。
基座一致性测试 5 项通过；已有 IK 验证 8 项通过、1 项因缺少 `dataset/`
episode 数据而报错（`test_curated_dataset_statistics`），该数据限制此前已存在。
未验证其他七个抓取任务、GUI 和 GPU 批量运行。

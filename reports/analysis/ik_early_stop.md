# IK 第二轮优化：最终目标提前停止

浮动基座 `Kinematics.ik` 默认开启 `early_stop=True`。初始状态以及每组
10 次迭代结束后，用独立的 Pinocchio Data 检查当前配置到最终目标的位姿残差。
位置（米）与世界系旋转向量（弧度）拼接后的范数不超过 `1e-4` 时返回。
该阈值比已有精调停止阈值 `1e-3` 更严格，但仍是混合单位的数值判据。

检查对象是最终目标，不是中间插值点；检查使用独立 Data，避免改变已有插值
起点的可变引用。提前返回的误差对应返回配置。未提前停止的分支保持原算法、
原精调预算及原有误差返回语义。环境连续成功条件和物理控制频率不变。
调用 `ik(..., early_stop=False)` 可关闭新行为，单独比较数值优化后的固定预算版本。

## CPU 计时

Xeon Silver 4316，MuJoCo 3.12.0，Pinocchio 4.1.0，state 观测、无 GUI。
本轮 Cube seed 7/8/9 共 285 个 step，均成功，步数保持 96/76/113。
排除 reset、预热、planner 和数据写盘；上一轮数值来自 `ik_optimization_profile.json`。

| 平均耗时 | 第一轮后 | 本轮后 |
| --- | ---: | ---: |
| IK | 13.243 ms | 12.229 ms |
| env.step | 16.990 ms | 15.960 ms |

IK 平均耗时再下降约 7.7%，完整 step 再下降约 6.1%。P95 step 从 17.969 ms
变为 17.786 ms。收益依赖轨迹中接近目标的比例，不能外推到所有任务。

## 任务对照

8 个任务逐个运行 seed 7，对比提前停止关闭/开启，均使用 CPU state 环境。
成功仅计 `terminated and info['is_success']`。以下是实际 episode 步数；
`satellite3_upper_rod` 两版均未确认成功，失败原因分别为环境截断和抓取未确认。
其步数、失败方式发生变化，不能宣称所有轨迹完全一致。

| 任务 | 关闭：步数/成功 | 开启：步数/成功 | IK 总迭代数：关闭→开启 |
| --- | --- | --- | --- |
| cube | 96 / 是 | 96 / 是 | 19200 → 17610 |
| satellite_handle | 79 / 是 | 79 / 是 | 15800 → 13010 |
| satellite_left_antenna_panel | 181 / 是 | 181 / 是 | 36200 → 31820 |
| satellite2_left_truss_connection | 117 / 是 | 117 / 是 | 23400 → 21180 |
| satellite3_left_antenna_panel | 158 / 是 | 157 / 是 | 31600 → 28090 |
| satellite3_upper_rod | 107 / 否 | 119 / 否 | 21400 → 19350 |
| debris_antenna_panel | 81 / 是 | 81 / 是 | 16200 → 14170 |
| debris_truss | 101 / 是 | 98 / 是 | 20200 → 14870 |

聚合计时与任务记录保存于 `ik_early_stop_profile.json`。单 seed 任务对照只能发现
明显回归，不足以证明大规模采集成功率不变。此次未验证 RGB、GUI 或 GPU 批量运行。

## 测试与复现

本次使用 `/tmp/spacerobot-warp-venv/bin/python`，命令从仓库根目录运行：

```bash
PYTHONPATH=src python -m unittest discover -s test -p 'test_ik_optimization.py'
PYTHONPATH=src python -m unittest discover -s test -p 'test_base_pose_consistency.py'
PYTHONPATH=src python -m unittest discover -s test -p 'test_ik_validation.py'
python reports/analysis/profile_cpu_step.py --modes state --seeds 7 8 9 --output /tmp/ik_early_new.json
```

输出文件必须不存在。IK 优化测试 8 项通过、基座一致性 5 项通过、已有 IK 验证
8 项通过。历史 `test_curated_dataset_statistics` 仍因缺少 `dataset/` episode 数据报错。
新增测试覆盖已对齐目标的零迭代、最终目标及返回配置残差、不可达目标的 250 次预算
与原轨迹一致、检查不扰动插值位姿、非法插值预算。第一轮数值等价测试继续显式使用
`early_stop=False`，避免把算法层面的提前停止与线性代数等价性混为一谈。

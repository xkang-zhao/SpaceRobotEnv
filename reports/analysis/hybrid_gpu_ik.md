# GPU IK 混合后端接入验证（2026-09-09）

## 行为

新增可选 `SpaceUR10eWarpVectorEnv(..., ik_backend="gpu")`。默认仍为 `cpu`。
`shadow` 生成 GPU 候选并比较关节解，但始终应用 CPU 求解结果。
混合后端将 MuJoCo FP32 wxyz 配置转换成 FP64 xyzw，归一化副本后上传。GPU
计算 40 次直接迭代候选；仅初始混合误差 <=0.02 且最终有限残差 <=1e-4 的行可采用。
不符合条件的 world 从原始仿真配置调用原 CPU IK，不从 GPU 中间解继续。
CUDA/求解器异常不冒充正常回退：标记所有 world 必须 reset 并传播异常。
planner、目标更新、控制整理、NumPy 状态返回仍在 CPU，尚未消除主机往返。
物理频率、动作/观测结构、连续成功标准和局部 reset 语义保持不变。

## 32 环境完整抓取对照

A800 80GB PCIe，CPU Xeon Silver 4316，MuJoCo/Warp 依赖沿用前轮。
每个模式运行 Cube seed 7–38 的独立首次尝试。模式顺序执行，不争用 GPU。
完成 world 继续临时 episode；直到全部首次尝试结束。统计包含首次 graph capture、
CPU/GPU IK、planner 和局部 reset，不包含初始创建/reset/最终关闭。无 RGB、GUI 或写盘。

| 模式 | 成功 | 总执行(s) | 平均批量step(ms) | CPU求解(ms/批) | GPU候选(ms/批) | 其余step(ms/批) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu | 32/32 | 23.79 | 155.52 | 63.49 | 0.00 | 92.03 |
| shadow | 32/32 | 24.06 | 157.49 | 61.66 | 3.93 | 91.90 |
| gpu | 32/32 | 15.62 | 96.31 | 0.00 | 3.97 | 92.33 |

混合模式总执行耗时减少约 34.3%，吞吐约为 CPU IK 模式的 1.52 倍。
GPU 候选阶段约 3.97 ms，包含上传、初始残差检查、GPU 求解和回读；其余约 92.33 ms
主要包含物理步进以及控制、观测和同步开销，不能全部解释为 GPU kernel 时间。
共 4448 个 world-step 的候选均被采用，本次实际轨迹没有触发 CPU 回退。
影子模式最大已接受候选关节差为 0.00065752 rad（约 0.038°）。
三个 seed 的终止步数各增加 1：19 为 82→83，22 为 112→113，29 为 72→73；
其他 world 的完成步数保持一致。因此成功率一致不意味着轨迹逐位相同。

## 回归测试

4 项真实 CUDA 混合测试全部通过：强制拒绝/NaN 候选时 CPU 控制与原状态求解完全一致；
影子模式不改变 CPU 控制和物理结果；局部 reset 保留其他 world 和 IK graph；
求解器异常后强制 reset。改动诊断字段后 4 项已重新运行通过。
原有批量环境 7 项 CUDA 测试全部通过，CLI 帮助及 git diff --check 通过。
默认 CPU 路径不增加诊断字段；GPU/shadow 增加按 world 的 ik_gpu_accepted、
ik_cpu_fallback、ik_residual；shadow 另有 ik_shadow_joint_max_abs。
回退统计不包含影子模式主动执行的 CPU 对照。残差是候选的诊断值，不替代环境成功判定。

## 使用与复现

```bash
/tmp/spacerobot-warp-venv/bin/python scripts/auto_grasp_warp_vector.py --num-envs 32 --seed 7 --ik-backend gpu
/tmp/spacerobot-warp-venv/bin/python reports/analysis/profile_warp_grasp.py --worlds 32 --ik-backend gpu --output /tmp/hybrid_new.json
RUN_WARP_TESTS=1 /tmp/spacerobot-warp-venv/bin/python -m unittest discover -s test -p test_hybrid_gpu_ik.py
RUN_WARP_TESTS=1 /tmp/spacerobot-warp-venv/bin/python -m unittest discover -s test -p test_warp_vector_env.py
```

将 ik-backend 改成 cpu/shadow 可复现对应对照，输出路径必须不存在。
原始结果为 hybrid_cpu32.json、hybrid_shadow32.json、hybrid_gpu32.json。
仅验证当前 Cube 的 32 个初始 seed；混合后端尚未测试 128/512 环境、其他任务、RGB、
训练和 LeRobot 数据采集。本轮不自动切换默认后端，也未修改单环境 CPU IK。

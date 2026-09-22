# GPU 多环境性能实测（2026-09-09）

硬件 NVIDIA A800 80GB PCIe，CPU Xeon Silver 4316。MuJoCo / mujoco-warp 3.12.0，
warp-lang 1.17.0。当前默认优化 IK，state 观测，无 RGB、GUI、实时限速或数据写盘。
两组测试顺序执行，避免互相争用 GPU。

## 固定控制物理回放

CPU Cube seed 7 生成 96 个控制步，每步 100 个物理子步。所有 GPU world 使用相同
初始状态与相同控制。预热、编译、graph capture、reset、IK、planner 均在物理计时之外。
物理各阶段显式同步。CPU 对照为一个 CPU 核，不代表多核 CPU 并行吞吐。

| world 数 | GPU 批量物理步进(ms/100子步) | 总物理子步/s | 物理吞吐相对单CPU核 | 进程GPU内存(MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 59.60 | 1,678 | 0.06× | 658 |
| 32 | 69.18 | 46,259 | 1.53× | 726 |
| 128 | 81.11 | 157,813 | 5.22× | 986 |
| 512 | 93.85 | 545,553 | 18.04× | 1946 |

CPU 物理基线：3.308 ms/100子步，30,234 子步/s。

512 world 的同步上传约 0.079 ms、回读约 0.252 ms，含传输批量延迟 94.180 ms。
各规模均有限、无容量溢出，qpos/qvel 相对 CPU 的误差在测试容差 0.005/0.1 内。
容差按分量判断、单位混合，不能解释为所有量的毫米或角度误差。相同 world 也不保证逐位一致。
GPU 内存为采样时进程占用，不是运行峰值。

## 独立任务完整抓取

Cube world w 使用 seed=7+w，各有独立 planner。记录每个 world 的首次尝试；
已经完成的 world 会进行局部 reset 并运行临时 episode，直到所有首次尝试结束。
执行总时间包含 CPU IK、planner、GPU step、首次 graph capture 和局部 reset；
不包含初始环境创建、初始 reset 和最终关闭。完整墙钟时间另外保存在 JSON 中。
下表均为一次批量运行，不是多次重复测量均值。step 列是该运行内各批量 step 的平均。

| 环境数 | 首次成功 | 总执行(s) | 批量step(ms) | CPU IK(ms/批) | step其余(ms/批) | 首次尝试完成数/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1/1 | 7.65 | 79.53 | 2.12 | 77.41 | 0.131 |
| 4 | 4/4 | 10.37 | 86.06 | 7.95 | 78.11 | 0.386 |
| 16 | 16/16 | 14.77 | 116.66 | 30.92 | 85.73 | 1.083 |
| 32 | 32/32 | 23.89 | 157.09 | 64.37 | 92.73 | 1.339 |

1/4/16/32 环境总计 53 次首次抓取均确认成功，连续计数均达到 10。其余 step 时间包含
GPU 物理、控制上传、状态读取、观测/接触/成功检查、Python 开销和首次 graph capture，
不能当作独立测得的 GPU kernel 时间。完整运行的状态和接触随 world 变化，不能直接
用固定回放的物理时间相减来估计同步成本。

结论：大批量物理回放已有明显吞吐收益，但 CPU IK 仍按 world 串行增长；
纯物理的 18 倍不等于完整抓取的 18 倍。当前只验证到 32 world 的完整抓取，
128/512 仅做物理回放。没有测试多核 CPU 采集对照，因此不能宣称 GPU 优于多核采集。

## 复现

以下命令从仓库根目录运行，输出文件必须不存在：

```bash
/tmp/spacerobot-warp-venv/bin/python scripts/benchmark_batch_physics.py --worlds 1 32 128 512 --output /tmp/warp_physics_new.json
/tmp/spacerobot-warp-venv/bin/python reports/analysis/profile_warp_grasp.py --worlds 1 4 16 32 --output /tmp/warp_grasp_new.json
```

原始数据：`warp_batch_current.json` 和 `warp_grasp_current.json`。已检查物理误差/有限性/容量、
53 个首次尝试成功条件，以及批量 step 分项求和一致性。

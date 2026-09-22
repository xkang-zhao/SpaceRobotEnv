# CPU 完整抓取耗时（2026-09-09）

Xeon Silver 4316，CPU MuJoCo 3.12.0、state 观测，无相机、GUI、实时限速或数据写盘。
8 个任务分别运行 seed 7/8/9，24/24 均由环境确认成功。执行时间包含 planner 动作计算、
全部 env.step、planner 状态更新和循环开销；日志格式化后写入内存，不输出到终端。
没有额外预热；每个任务创建一次环境，然后执行 3 次 reset 和抓取。
创建时间不含 Python 进程启动和顶层导入，也不含关闭环境。首次任务创建可能包含首次依赖初始化，
不能直接视为所有任务独立冷启动的比较。reset 包含 planner 初始化。

| 任务 | 执行均值(s) | 最小–最大(s) | reset均值(s) | 创建一次(s) | 仿真时间均值(s) |
| --- | ---: | --- | ---: | ---: | ---: |
| cube | 0.535 | 0.429–0.637 | 0.056 | 5.612 | 4.750 |
| satellite_handle | 0.662 | 0.639–0.687 | 0.055 | 2.513 | 3.950 |
| satellite_left_antenna_panel | 2.453 | 2.328–2.572 | 0.047 | 2.358 | 8.800 |
| satellite2_left_truss_connection | 1.524 | 1.402–1.641 | 0.043 | 2.550 | 5.550 |
| satellite3_left_antenna_panel | 2.124 | 2.019–2.184 | 0.043 | 2.483 | 7.650 |
| satellite3_upper_rod | 1.179 | 1.064–1.406 | 0.044 | 2.315 | 5.083 |
| debris_antenna_panel | 1.000 | 0.988–1.011 | 0.048 | 2.444 | 4.050 |
| debris_truss | 1.118 | 1.042–1.189 | 0.043 | 2.262 | 4.950 |

复用环境的一次抓取总时间 = reset + 执行。第一次使用还需加环境创建时间。
仿真时间是物理世界中经过的时间，不是计算耗时；如果启用 20 Hz 实时限速，实际等待会增加。
采集视频/数据或启用相机的耗时不包含在本报告内。仅 3 个 seed，不表示最坏耗时。

复现命令（输出路径必须不存在）：

```bash
/tmp/spacerobot-warp-venv/bin/python reports/analysis/profile_grasp_episode.py --seeds 7 8 9 --output /tmp/grasp_timing_new.json
```

逐 episode 原始数据：`grasp_episode_timing.json`。已检查 24 次成功、执行阶段求和、
以及仿真时间与每 step 0.05 秒的一致性。

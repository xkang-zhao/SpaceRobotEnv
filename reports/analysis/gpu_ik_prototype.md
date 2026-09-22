# GPU 批量 IK 独立原型验证（2026-09-09）

## 范围与实现

环境默认求解器保持不变。新原型位于 `src/mujoco_robot/gpu_ik_prototype.py`，
一次性从 Pinocchio 导出关节固定变换、转轴、质量、质心和惯性张量。支持当前
`arm.xml` 的自由漂浮基座加六个旋转关节、末端 frame 位于最后一个关节的串联链。
每轮 GPU 计算 FK、末端世界系 Jacobian、基座质量矩阵 M_bb 和耦合块 M_bm，
求解 B=M_bb⁻¹M_bm、J_g=J_m−J_bB，随后阻尼求解关节速度并用 v_b=−Bv_m
保持零动量耦合。基座以 SE(3) 指数映射积分，使用 Pinocchio 的 xyzw 配置约定。
全部数值计算为 FP64。采用 Cholesky 解两个 6×6 线性系统；不同 world 独立。
没有在 CPU 上预计算每轮矩阵，也没有把现有浮动基座模型替换成固定基座。

原型执行最多 40 次直接迭代，dt=0.2、damping=0.05、速度总范数上限 0.2。
每轮最终目标混合残差达到 1e-4 后冻结该 world 的配置。最后独立计算输出残差，
只有有限且达标的输出才标记 accepted。其余行需要在后续集成中送回 CPU 求解。
原型没有实现完整插值回退，也不应直接用于控制。

## 性能

A800 80GB PCIe，Xeon Silver 4316，Warp 1.17.0、Pinocchio 4.1.0。
先从 CPU Cube seed 7 的成功 episode 记录 96 对真实 q/目标；N 个 world 使用
这些固定输入，N>96 时循环复用。不是 N 个独立场景的闭环运行。GPU 对同一输入
重复 3 次，表中给均值；CPU 是相同批输入上的一次串行计时，未做多核并行。
排除编译、模型导出和 CUDA Graph 捕获，计时边界显式同步。上传包含主机数组准备，
回读包含 NumPy 转换和 accepted 检查。首次原型 CUDA 编译另耗约 94 秒，后续命中缓存。

| 批大小 | 当前生产CPU IK(ms) | 同40次算法CPU(ms) | GPU计算(ms) | 上传+回读(ms) | GPU合计(ms) | 达标数 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.58 | 1.84 | 1.86 | 0.38 | 2.24 | 1/1 |
| 32 | 67.17 | 66.76 | 3.21 | 0.48 | 3.69 | 32/32 |
| 128 | 250.34 | 234.32 | 6.61 | 0.67 | 7.28 | 128/128 |
| 512 | 1127.96 | 911.94 | 12.01 | 1.73 | 13.74 | 512/512 |

32 个输入的 GPU 总时间约为 CPU 串行生产路径的 1/18；单输入 GPU 更慢。
批次组成和运行负载会影响倍率，尤其不能将 IK 倍率解释成完整抓取提速倍率。
生产 CPU 每 10 次检查并可能回退到插值；GPU 每轮检查，所以最终配置允许有小差异。
相同算法 CPU 对照使用与 GPU 相同的 40 次预算、逐次达标检查、步长和阻尼。

## 精度与失败路径

验证集 96 组包含不同平移/旋转的浮动基座、中立构型、64 个小目标、8 个已对齐目标、
8 个大目标、8 个接近 π 的旋转目标和 8 个不可达目标。最大差异如下：

| 项目 | 最大误差 |
| --- | ---: |
| mass_abs | 3.411e-13 |
| jacobian_abs | 3.331e-15 |
| pose_abs | 3.719e-15 |
| error_vector_abs | 1.421e-14 |
| single_step_q_abs | 2.109e-15 |
| momentum_relative | 4.514e-16 |
| forty_step_q_abs | 5.159e-09 |
| final_residual_abs | 8.438e-15 |

质量与 Jacobian 是对应分量绝对误差；配置误差混合米、四元数分量和弧度，不转换成单一长度单位。
动量残差是 ||M_bb v_b + M_bm v_m|| 除以两项范数之和；与 CPU 对照一致。
96 个样本中已对齐 8/8、小目标 41/64 达标；其余 47 个未达标并标记需要回退。
大目标、近 π 和不可达目标均没有被错误地判为达标；CPU 同预算也有相同收敛结果。
因此不能因 Cube 轨迹样本全部达标而取消回退。CPU 回退能否解决各困难输入还需另行验证。

真实 Cube 输入中，GPU 的末端位置误差最大约 0.100 mm，旋转误差最大约 3.585e-5 rad。
相对生产 CPU 输出，最大关节差约 1.750e-4 rad（约 0.010°）；区别主要来自提前停止频率。
这些是运动学端点误差，不等于接触状态或闭环抓取成功率。

## 测试与复现

3 项真实 CUDA 测试通过：矩阵/Jacobian/积分/动量对照，逐 world 独立性与四元数符号/重复回放，
输入和预算校验。分析脚本在 4 个规模的所有输出上再次用 Pinocchio 核对残差和同算法配置。
git diff --check 通过。未修改当前环境、planner、成功条件、依赖配置或用户数据。

```bash
RUN_WARP_TESTS=1 /tmp/spacerobot-warp-venv/bin/python -m unittest discover -s test -p test_gpu_ik_prototype.py
/tmp/spacerobot-warp-venv/bin/python reports/analysis/validate_gpu_ik.py --worlds 1 32 128 512 --output /tmp/gpu_ik_new.json
```

原始最终结果：`gpu_ik_prototype_final.json`；首轮试验另保存在 `gpu_ik_prototype_validation.json`。
输出路径必须不存在。GPUChainIK.prepare 会预热并改变临时 q，调用者应随后 upload 原始输入，
再执行返回的 graph。read 应在 graph 执行完成或对最终 q 调用 evaluate 后使用。

建议下一步：先在 32 环境中做 GPU IK 候选结果的影子对照，再接入有明确失败状态和 CPU 回退的
控制链，验证完整抓取成功率、总吞吐与状态传输成本。当前结果支持进入该阶段，尚不支持直接
切换所有环境。硬件/库接口参考 [Warp CUDA Graph API](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.ScopedCapture.html)。

# Satellite3 上杆抓取标定修复

当前失败来自抓取高度对细杆接触的敏感性。任务 profile 的 EE 目标在卫星 body
坐标系中原为 `[-0.678392, 0.003160, 0.218000]`。seed 7 在第 64 步开始闭合，
出现短暂双指接触后失去抓取，最终第 119 步以 `grasp was not confirmed` 结束。

已有测试记录的旧高度是 `0.213932`。仅恢复该高度可以修复 seed 7，但 seed 0–9
中仍有 4、5 两个失败。将 hold_steps 临时从 20 增至 80 只修复 seed 5，seed 4
仍失败，说明延长等待不能解决全部接触问题。

最终将该任务 EE 目标的局部 Z 调整为 `0.210000 m`，相对原配置降低 8 mm。
X/Y、姿态（skip rotation）、位置容差、关闭/保持步数及环境成功判定均未改变。
profile 由 CLI 与数据采集适配器共用，无需分别修改入口。README 和标定测试已同步。
这个值是当前模型和 IK 下的经验标定，不是杆中心位置或 pinch site 的 body offset。

## 验证

CPU MuJoCo 3.12.0、Pinocchio 4.1.0、state 观测、无 GUI，使用当前默认优化 IK。
seed 0–9 全部成功后，额外验证未用于调整高度的 seed 10–19，也全部成功。
总计 20/20，每次均达到 `terminated and info['is_success']` 且 success_counter=10，
完成步数 86–117。详细逐 seed 结果见 `satellite3_upper_rod_fix.json`。

新增实际物理回归测试覆盖 seed 4、5、7，并逐步核对连续成功条件；该测试通过。
24 项自动抓取单元测试全部通过。任务 CLI 帮助与 git diff --check 均通过。

从仓库根目录执行（本次 Python 为 `/tmp/spacerobot-warp-venv/bin/python`）：

```bash
PYTHONPATH=src python -m unittest discover -s test -p 'test_satellite3_upper_rod_smoke.py'
PYTHONPATH=src python -m unittest discover -s test -p 'test_auto_grasp_*.py'
python scripts/auto_grasp.py satellite3_upper_rod --help
```

20 个 seed 不代表所有随机状态均成功。未验证 GUI、Warp 后端和真实 LeRobot 数据采集，
也没有修改 MJCF、用户数据或通用 planner。此前 IK 报告中的失败记录保留为历史结果。

#!/bin/bash
# 启动键盘遥操作来收集多视角和状态数据的脚本

# 防止由于环境变量问题导致的 OMP 错误
export KMP_DUPLICATE_LIB_OK=TRUE

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 使用你默认配置的 conda 环境名称（如果是 lerobot，请确认运行此脚本前已经 activate 或是通过 conda run）
# conda activate lerobot  # 如果需要，取消注释

python "${REPO_ROOT}/scripts/collect_dataset_keyboard_lerobot.py" \
    --env "SpaceUR10e-Cube-v0" \
    --repo_id "local/my_3cam_dataset_keyboard" \
    --num_episodes 2 \
    --task "Grab the cube" \
    --cam1_key "third_left_camera" \
    --cam2_key "third_right_camera" \
    --cam3_key "left_wrist_camera" \
    --enable_rerun

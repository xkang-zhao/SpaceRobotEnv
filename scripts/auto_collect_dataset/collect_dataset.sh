#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ======================== 主要采集参数 ========================
# 每个任务最终保存的 episode 数量。
EPISODES_PER_TASK="${EPISODES_PER_TASK:-100}"
# 每个任务使用的独立 MuJoCo worker 数量。
NUM_ENVS="${NUM_ENVS:-8}"
# 每个任务会保存到 ${OUTPUT_ROOT}/<task>_lerobot_v21。
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/dataset}"
# =============================================================

COLLECTOR="${REPO_ROOT}/scripts/auto_collect_dataset/collect_autograb_lerobot_v21.py"
TASKS=(
    cube
    satellite_handle
    satellite_left_antenna_panel
    satellite2_left_truss_connection
    satellite3_left_antenna_panel
    satellite3_upper_rod
    debris_antenna_panel
    debris_truss
)

mkdir -p "${OUTPUT_ROOT}"

total_tasks=${#TASKS[@]}
batch_started=${SECONDS}
succeeded_tasks=()
failed_tasks=()

echo "============================================================"
echo "[AUTOGRAB] 开始批量采集 LeRobot v2.1 数据集"
echo "[AUTOGRAB] 任务数量: ${total_tasks}"
echo "[AUTOGRAB] 每任务 episodes: ${EPISODES_PER_TASK}"
echo "[AUTOGRAB] 并行环境数: ${NUM_ENVS}"
echo "[AUTOGRAB] 输出根目录: ${OUTPUT_ROOT}"
echo "============================================================"

for task_index in "${!TASKS[@]}"; do
    task="${TASKS[task_index]}"
    display_index=$((task_index + 1))
    task_output="${OUTPUT_ROOT}/${task}_lerobot_v21"
    task_started=${SECONDS}

    echo
    echo "------------------------------------------------------------"
    echo "[AUTOGRAB][${display_index}/${total_tasks}] 开始任务: ${task}"
    echo "[AUTOGRAB][${display_index}/${total_tasks}] 输出目录: ${task_output}"
    echo "[AUTOGRAB][${display_index}/${total_tasks}] 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    if python "${COLLECTOR}" "${task}" \
        --output-dir "${task_output}" \
        --episodes "${EPISODES_PER_TASK}" \
        --num-envs "${NUM_ENVS}" \
        --seed 0 \
        --max-attempts-per-episode 10; then
        task_elapsed=$((SECONDS - task_started))
        succeeded_tasks+=("${task}")
        echo "[AUTOGRAB][${display_index}/${total_tasks}] 成功: ${task}，耗时 ${task_elapsed}s"
    else
        exit_code=$?
        task_elapsed=$((SECONDS - task_started))
        failed_tasks+=("${task}(exit=${exit_code})")
        echo "[AUTOGRAB][${display_index}/${total_tasks}] 失败: ${task}，退出码 ${exit_code}，耗时 ${task_elapsed}s"
        echo "[AUTOGRAB] 将继续执行剩余任务。"
    fi
done

batch_elapsed=$((SECONDS - batch_started))

echo
echo "============================================================"
echo "[AUTOGRAB] 批量采集结束，总耗时 ${batch_elapsed}s"
echo "[AUTOGRAB] 成功任务数: ${#succeeded_tasks[@]}/${total_tasks}"
if ((${#succeeded_tasks[@]} > 0)); then
    printf '[AUTOGRAB] 成功任务:'
    printf ' %s' "${succeeded_tasks[@]}"
    printf '\n'
fi
if ((${#failed_tasks[@]} > 0)); then
    printf '[AUTOGRAB] 失败任务:'
    printf ' %s' "${failed_tasks[@]}"
    printf '\n'
    echo "============================================================"
    exit 1
fi
echo "[AUTOGRAB] 8 个任务全部采集成功。"
echo "============================================================"

# Linux headless:
# MUJOCO_GL=egl bash scripts/auto_collect_dataset/collect_dataset.sh

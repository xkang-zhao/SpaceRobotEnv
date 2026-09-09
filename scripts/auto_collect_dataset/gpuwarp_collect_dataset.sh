#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EPISODES="${EPISODES:-100}"
NUM_ENVS="${NUM_ENVS:-32}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/local}"
SEED="${SEED:-0}"
IK_BACKEND="${IK_BACKEND:-gpu}"
DEVICE="${DEVICE:-cuda:0}"

python "${REPO_ROOT}/scripts/auto_collect_dataset/gpuwarp_collect_autograb_lerobot_v21.py" cube \
  --output-dir "${OUTPUT_ROOT}/cube_gpuwarp_lerobot_v21" \
  --episodes "${EPISODES}" \
  --num-envs "${NUM_ENVS}" \
  --seed "${SEED}" \
  --ik-backend "${IK_BACKEND}" \
  --device "${DEVICE}" \
  --max-attempts-per-episode 10

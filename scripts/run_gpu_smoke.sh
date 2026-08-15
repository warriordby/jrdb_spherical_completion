#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_DIR}/configs/jrdb_vace14b_pro6000.json"
SEQUENCES=(
  cubberly-auditorium-2019-04-22_1
  discovery-walk-2019-02-28_0
  indoor-coupa-cafe-2019-02-06_0
)

bash "${PROJECT_DIR}/scripts/run_offline_vace.sh" --config "${CONFIG}" doctor --require-cuda
for sequence in "${SEQUENCES[@]}"; do
  bash "${PROJECT_DIR}/scripts/run_offline_vace.sh" --config "${CONFIG}" run \
    --backend vace14b --sequences "${sequence}" --limit-frames 81
  bash "${PROJECT_DIR}/scripts/run_offline_vace.sh" --config "${CONFIG}" verify \
    --sequences "${sequence}" --limit-frames 81
  bash "${PROJECT_DIR}/scripts/run_offline_vace.sh" --config "${CONFIG}" quality \
    --sequences "${sequence}" --limit-frames 81
done

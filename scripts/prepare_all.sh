#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_DIR}/configs/jrdb_vace14b_pro6000.json"

bash "${PROJECT_DIR}/scripts/fetch_sources.sh"
bash "${PROJECT_DIR}/scripts/install_cpu_env.sh"
bash "${PROJECT_DIR}/scripts/install_vace_env.sh"
"${PROJECT_DIR}/.venv/bin/python" "${PROJECT_DIR}/scripts/download_resources.py" \
  --config "${CONFIG}" \
  --hf-endpoint "${HF_ENDPOINT:-https://hf-mirror.com}"
"${PROJECT_DIR}/.venv/bin/jrdb-sphere" --config "${CONFIG}" doctor

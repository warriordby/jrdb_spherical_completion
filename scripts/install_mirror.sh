#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point. The production VACE environment is built by
# install_vace_env.sh; this script now installs only the CPU/download CLI.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${PROJECT_DIR}/scripts/install_cpu_env.sh"

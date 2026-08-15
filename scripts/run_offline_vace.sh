#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="/root/autodl-tmp/model_cache_v2/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/root/autodl-tmp/model_cache_v2/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
exec "${PROJECT_DIR}/.venv-vace/bin/jrdb-sphere" "$@"

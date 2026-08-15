#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/model_cache_v2}"
ENV_DIR="${PROJECT_DIR}/.venv-vace"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    python3.10 -m venv "${ENV_DIR}"
  else
    echo "python3.10 is required (install python3.10-venv first)" >&2
    exit 1
  fi
fi

PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"
PIP_CACHE_DIR="${CACHE_ROOT}/pip_cache"
mkdir -p "${PIP_CACHE_DIR}"
PIP_COMMON=(--cache-dir "${PIP_CACHE_DIR}" --timeout 1200 --retries 20)
"${PYTHON}" -m pip install "${PIP_COMMON[@]}" --upgrade pip setuptools wheel ninja

# The PyTorch CUDA index also exposes links for generic dependencies, but those
# mirrors can be much slower than the configured PyPI mirror.  Install the
# small, pinned runtime dependencies first so the CUDA transaction only needs
# the PyTorch/NVIDIA wheels.  This also keeps torchvision from selecting a
# newer Pillow release than the VACE lock file.
"${PIP}" install "${PIP_COMMON[@]}" \
  "filelock==3.19.1" \
  "fsspec==2025.7.0" \
  "Jinja2==3.1.6" \
  "MarkupSafe==3.0.2" \
  "networkx==3.4.2" \
  "numpy==1.26.4" \
  "Pillow==11.3.0" \
  "sympy==1.14.0" \
  "typing_extensions==4.14.1"

torch_ready=0
if "${PIP}" install "${PIP_COMMON[@]}" --index-url "https://download.pytorch.org/whl/cu128" \
    --only-binary=:all: "torch==2.7.1+cu128" "torchvision==0.22.1+cu128"; then
  torch_ready=1
fi
if [[ "${torch_ready}" -ne 1 ]]; then
  echo "Unable to install the pinned CUDA 12.8 PyTorch build" >&2
  exit 1
fi

"${PIP}" install "${PIP_COMMON[@]}" -r "${PROJECT_DIR}/requirements/vace.lock.txt"
"${PIP}" install "${PIP_COMMON[@]}" -e "${CACHE_ROOT}/sources/Wan2.1" --no-deps
"${PIP}" install "${PIP_COMMON[@]}" -e "${PROJECT_DIR}" --no-deps

# PyPI's decord 0.6.0 wheel is correctly named as py3-none but contains an
# obsolete cp36 tag in WHEEL.  The package is ctypes-based and imports on
# Python 3.10; repair the upstream tag so modern pip check does not reject it.
"${PYTHON}" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

wheel = Path(distribution("decord")._path) / "WHEEL"
text = wheel.read_text(encoding="utf-8")
bad = "Tag: cp36-cp36m-manylinux2010_x86_64"
good = "Tag: py3-none-manylinux2010_x86_64"
if bad in text:
    wheel.write_text(text.replace(bad, good), encoding="utf-8")
PY

if ! "${PYTHON}" -c 'import flash_attn' >/dev/null 2>&1; then
  if [[ "$("${PYTHON}" -c 'import torch; print(bool(torch._C._GLIBCXX_USE_CXX11_ABI))')" == "True" ]]; then
    FLASH_ABI="TRUE"
  else
    FLASH_ABI="FALSE"
  fi
  if [[ "${FLASH_ABI}" == "TRUE" ]]; then
    FLASH_FILE="flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    FLASH_SHA256="ce91e246f21d61ad66b1a7555340dbaa28e4aa86edcf00c18f0837422939b529"
    FLASH_DIR="${CACHE_ROOT}/wheels"
    FLASH_PATH="${FLASH_DIR}/${FLASH_FILE}"
    FLASH_INCOMPLETE="${FLASH_PATH}.incomplete"
    FLASH_URL="https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
    mkdir -p "${FLASH_DIR}"
    if [[ -f "${FLASH_PATH}" ]]; then
      if [[ "$(sha256sum "${FLASH_PATH}" | awk '{print $1}')" != "${FLASH_SHA256}" ]]; then
        echo "Cached flash-attn wheel checksum mismatch: ${FLASH_PATH}" >&2
        exit 1
      fi
    else
      flash_downloaded=0
      if command -v aria2c >/dev/null 2>&1; then
        if aria2c --continue=true --allow-overwrite=true --auto-file-renaming=false \
            --file-allocation=none --max-connection-per-server=8 --split=8 \
            --min-split-size=1M --max-tries=8 --retry-wait=3 --timeout=60 \
            --summary-interval=20 --dir="${FLASH_DIR}" \
            --out="${FLASH_FILE}.incomplete" "${FLASH_URL}"; then
          flash_downloaded=1
        fi
      elif curl -L --fail --retry 12 --retry-all-errors --continue-at - \
          --output "${FLASH_INCOMPLETE}" "${FLASH_URL}"; then
        flash_downloaded=1
      fi
      if [[ "${flash_downloaded}" -eq 1 ]] && \
          "${PYTHON}" -m zipfile -t "${FLASH_INCOMPLETE}" >/dev/null && \
          [[ "$(sha256sum "${FLASH_INCOMPLETE}" | awk '{print $1}')" == "${FLASH_SHA256}" ]]; then
        mv "${FLASH_INCOMPLETE}" "${FLASH_PATH}"
      else
        echo "Warning: flash-attn wheel is unavailable; Wan will use PyTorch SDPA." >&2
      fi
    fi
    if [[ -f "${FLASH_PATH}" ]]; then
      "${PIP}" install "${PIP_COMMON[@]}" "${FLASH_PATH}"
    fi
  else
    echo "Warning: no pinned flash-attn wheel for cxx11abiFALSE; Wan will use PyTorch SDPA." >&2
  fi
fi

"${PYTHON}" -m pip check
"${PYTHON}" - <<'PY'
import json
import torch
import torchvision
import transformers
import diffusers
import decord
import importlib.util
print(json.dumps({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "diffusers": diffusers.__version__,
    "decord": decord.__version__,
    "wan_discoverable": importlib.util.find_spec("wan") is not None,
    "flash_attn": importlib.util.find_spec("flash_attn") is not None,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}, indent=2))
PY

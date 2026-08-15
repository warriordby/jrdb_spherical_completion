#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/model_cache_v2}"
SOURCE_ROOT="${CACHE_ROOT}/sources"
mkdir -p "${SOURCE_ROOT}"

checkout_source() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="${SOURCE_ROOT}/${name}"
  local slug="${url#https://github.com/}"
  slug="${slug%.git}"
  local created=0
  if [[ -f "${destination}/.source_commit" ]]; then
    if [[ "$(tr -d '\n' < "${destination}/.source_commit")" != "${commit}" ]]; then
      echo "Source archive commit mismatch in ${destination}" >&2
      return 1
    fi
    return
  fi
  if [[ ! -d "${destination}/.git" ]]; then
    local archive_dir="${CACHE_ROOT}/source_archives"
    local archive="${archive_dir}/${name}-${commit}.tar.gz"
    local staging="${SOURCE_ROOT}/.${name}-${commit}.staging"
    mkdir -p "${archive_dir}" "${staging}"
    curl -L --fail --retry 8 --retry-all-errors --continue-at - \
      --output "${archive}.incomplete" \
      "https://codeload.github.com/${slug}/tar.gz/${commit}"
    mv "${archive}.incomplete" "${archive}"
    tar -xzf "${archive}" --strip-components=1 -C "${staging}"
    printf '%s\n' "${commit}" > "${staging}/.source_commit"
    mv "${staging}" "${destination}"
    return
  fi
  if [[ "${created}" -eq 1 ]]; then
    if [[ "$(git -C "${destination}" rev-parse HEAD)" != "${commit}" ]]; then
      git -C "${destination}" fetch --depth 1 --no-tags origin "${commit}"
      git -C "${destination}" checkout --detach "${commit}"
    fi
    return
  fi
  if ! git -C "${destination}" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "${destination}" fetch --no-tags origin "${commit}"
    git -C "${destination}" checkout --detach "${commit}"
    return
  fi
  # Recover a prior interrupted --no-checkout clone. This branch is only
  # entered when the worktree was never materialized at all.
  if [[ ! -e "${destination}/.gitignore" ]] && \
      [[ -n "$(git -C "${destination}" diff --cached --name-only)" ]]; then
    git -C "${destination}" restore --source=HEAD --staged --worktree .
  fi
  if ! git -C "${destination}" diff --cached --quiet; then
    echo "Refusing to overwrite staged changes in ${destination}" >&2
    return 1
  fi
  if [[ "${name}" != "VACE" && "${name}" != "Wan2.1" ]] && ! git -C "${destination}" diff --quiet; then
    echo "Refusing to overwrite local changes in ${destination}" >&2
    return 1
  fi
  if [[ "${name}" == "VACE" ]]; then
    local unexpected
    unexpected="$(git -C "${destination}" diff --name-only | grep -Ev '^(vace/vace_wan_inference.py|vace/models/wan/wan_vace.py|vace/models/wan/modules/model.py|vace/models/wan/distributed/xdit_context_parallel.py)$' || true)"
    if [[ -n "${unexpected}" ]]; then
      echo "Refusing to overwrite unexpected VACE changes: ${unexpected}" >&2
      return 1
    fi
  fi
  if [[ "${name}" == "Wan2.1" ]]; then
    local unexpected
    unexpected="$(git -C "${destination}" diff --name-only | grep -Ev '^(pyproject.toml|wan/modules/model.py)$' || true)"
    if [[ -n "${unexpected}" ]]; then
      echo "Refusing to overwrite unexpected Wan2.1 changes: ${unexpected}" >&2
      return 1
    fi
  fi
  git -C "${destination}" fetch --no-tags origin "${commit}"
  if [[ "$(git -C "${destination}" rev-parse HEAD 2>/dev/null || true)" != "${commit}" ]]; then
    git -C "${destination}" checkout --detach "${commit}"
  fi
}

checkout_source VACE https://github.com/ali-vilab/VACE.git 48eb44f1c4be87cc65a98bff985a26976841e9f3
checkout_source Wan2.1 https://github.com/Wan-Video/Wan2.1.git 9737cba9c1c3c4d04b33fcad41c111989865d315
checkout_source Seen_to_Scene https://github.com/InSeokJeon/Seen_to_Scene.git 2a9dfc9888e44c7fd00b08af41ef967ae46b6323
checkout_source TrackEval https://github.com/JonathonLuiten/TrackEval.git 12c8791b303e0a0b50f753af204249e622d0281a
checkout_source RAFT https://github.com/princeton-vl/RAFT.git 2888e15a51fa41140771d3f498ed8023cff098d1
checkout_source deep-person-reid https://github.com/KaiyangZhou/deep-person-reid.git f8cd150fdf77e8d9e1ed143b7f308c2c609ded50

VACE_DIR="${SOURCE_ROOT}/VACE"
PATCH_FILE="${PROJECT_DIR}/patches/vace_jrdb_proxy.patch"
if ! grep -q -- "--noise_overlap_in" "${VACE_DIR}/vace/vace_wan_inference.py"; then
  git -C "${VACE_DIR}" apply --check "${PATCH_FILE}"
  git -C "${VACE_DIR}" apply "${PATCH_FILE}"
fi
VACE_MEMORY_PATCH_FILE="${PROJECT_DIR}/patches/vace_memory_offload.patch"
if ! grep -q -- "def move_vae" "${VACE_DIR}/vace/models/wan/wan_vace.py"; then
  git -C "${VACE_DIR}" apply --check "${VACE_MEMORY_PATCH_FILE}"
  git -C "${VACE_DIR}" apply "${VACE_MEMORY_PATCH_FILE}"
fi
VACE_HINT_PATCH_FILE="${PROJECT_DIR}/patches/vace_hint_offload.patch"
if ! grep -q -- "return tuple(all_c)" "${VACE_DIR}/vace/models/wan/modules/model.py"; then
  git -C "${VACE_DIR}" apply --check "${VACE_HINT_PATCH_FILE}"
  git -C "${VACE_DIR}" apply "${VACE_HINT_PATCH_FILE}"
fi

WAN_DIR="${SOURCE_ROOT}/Wan2.1"
WAN_PATCH_FILE="${PROJECT_DIR}/patches/wan_jrdb_runtime.patch"
if ! grep -q -- 'opencv-python-headless>=4.9.0.80' "${WAN_DIR}/pyproject.toml"; then
  if [[ -d "${WAN_DIR}/.git" ]]; then
    git -C "${WAN_DIR}" apply --check "${WAN_PATCH_FILE}"
    git -C "${WAN_DIR}" apply "${WAN_PATCH_FILE}"
  else
    patch --directory="${WAN_DIR}" --strip=1 --forward < "${WAN_PATCH_FILE}"
  fi
fi
WAN_ROPE_PATCH_FILE="${PROJECT_DIR}/patches/wan_rope_memory.patch"
if ! grep -q -- 'chunk_size = 16384' "${WAN_DIR}/wan/modules/model.py"; then
  if grep -q -- 'return torch.stack(output).to(x.dtype)' "${WAN_DIR}/wan/modules/model.py"; then
    WAN_ROPE_PATCH_FILE="${PROJECT_DIR}/patches/wan_rope_chunk_upgrade.patch"
  fi
  if [[ -d "${WAN_DIR}/.git" ]]; then
    git -C "${WAN_DIR}" apply --check "${WAN_ROPE_PATCH_FILE}"
    git -C "${WAN_DIR}" apply "${WAN_ROPE_PATCH_FILE}"
  else
    patch --directory="${WAN_DIR}" --strip=1 --forward < "${WAN_ROPE_PATCH_FILE}"
  fi
fi

for source in VACE Wan2.1 Seen_to_Scene TrackEval RAFT deep-person-reid; do
  if [[ -d "${SOURCE_ROOT}/${source}/.git" ]]; then
    revision="$(git -C "${SOURCE_ROOT}/${source}" rev-parse HEAD)"
  else
    revision="$(tr -d '\n' < "${SOURCE_ROOT}/${source}/.source_commit")"
  fi
  printf '%-20s %s\n' "${source}" "${revision}"
done

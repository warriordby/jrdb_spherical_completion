#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WAN_REPO = "Wan-AI/Wan2.1-VACE-14B"
WAN_REVISION = "539c162b1387eac9dc4c20bd3f74671309e76a4c"
MODELSCOPE_WAN_BASE = "https://www.modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B/resolve/master"
OSNET_REPO = "kaiyangzhou/osnet"
OSNET_REVISION = "a5c5cc037c24235cda3b21085b93ad77c9616224"
OSNET_SIZE = 10_910_553
OSNET_SHA256 = "fe2d63f9157c28a4a8d8ca29bec12d5b2988ac0346d712025789ea9174968e79"
WAN_FILES = {
    ".gitattributes": 2004,
    "LICENSE.txt": 11357,
    "README.md": 40485,
    "Wan2.1_VAE.pth": 507609880,
    "config.json": 325,
    "diffusion_pytorch_model-00001-of-00007.safetensors": 9887603256,
    "diffusion_pytorch_model-00002-of-00007.safetensors": 9839059648,
    "diffusion_pytorch_model-00003-of-00007.safetensors": 9839059744,
    "diffusion_pytorch_model-00004-of-00007.safetensors": 9839059744,
    "diffusion_pytorch_model-00005-of-00007.safetensors": 9839059744,
    "diffusion_pytorch_model-00006-of-00007.safetensors": 7910235256,
    "diffusion_pytorch_model-00007-of-00007.safetensors": 6098227760,
    "diffusion_pytorch_model.safetensors.index.json": 118655,
    "google/umt5-xxl/special_tokens_map.json": 6623,
    "google/umt5-xxl/spiece.model": 4548313,
    "google/umt5-xxl/tokenizer.json": 16837417,
    "google/umt5-xxl/tokenizer_config.json": 61728,
    "models_t5_umt5-xxl-enc-bf16.pth": 11361920418,
}

SOURCE_COMMITS = {
    "VACE": "48eb44f1c4be87cc65a98bff985a26976841e9f3",
    "Wan2.1": "9737cba9c1c3c4d04b33fcad41c111989865d315",
    "Seen_to_Scene": "2a9dfc9888e44c7fd00b08af41ef967ae46b6323",
    "TrackEval": "12c8791b303e0a0b50f753af204249e622d0281a",
    "RAFT": "2888e15a51fa41140771d3f498ed8023cff098d1",
    "deep-person-reid": "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50",
}

DIRECT_FILES = {
    "wheels/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl": {
        "url": "https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
        "size": 256_006_094,
        "sha256": "ce91e246f21d61ad66b1a7555340dbaa28e4aa86edcf00c18f0837422939b529",
        "source": "Dao-AILab flash-attention GitHub release v2.8.3 via ghfast mirror",
    },
    "quality/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth": {
        "url": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        "minimum_size": 100_000_000,
        "source": "torchvision FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT",
    },
    "torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth": {
        "url": "https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth",
        "minimum_size": 10_000_000,
        "source": "torchvision Raft_Large_Weights.DEFAULT",
    },
    "quality/raft-things.pth": {
        "url": "https://gh-proxy.com/https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
        "size": 21_108_000,
        "source": "ProPainter GitHub release v0.1.0 via gh-proxy mirror",
    },
    "quality/recurrent_flow_completion.pth": {
        "url": "https://gh-proxy.com/https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        "size": 20_348_681,
        "source": "ProPainter GitHub release v0.1.0 via gh-proxy mirror",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parents[1] / "configs" / "jrdb_vace14b_pro6000.json"),
    )
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip-wan", action="store_true", help="Only refresh small evaluation assets")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_size(path: Path, spec: dict[str, Any]) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing downloaded file: {path}")
    size = path.stat().st_size
    if "size" in spec and size != int(spec["size"]):
        raise RuntimeError(f"size mismatch for {path}: {size} != {spec['size']}")
    if size < int(spec.get("minimum_size", 1)):
        raise RuntimeError(f"downloaded file is unexpectedly small: {path} ({size} bytes)")
    expected_hash = spec.get("sha256")
    if expected_hash is not None:
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"checksum mismatch for {path}: {actual_hash} != {expected_hash}"
            )


def download_url(url: str, target: Path, spec: dict[str, Any]) -> None:
    if target.is_file():
        try:
            verify_size(target, spec)
            return
        except RuntimeError:
            target.rename(target.with_name(target.name + ".invalid"))
    target.parent.mkdir(parents=True, exist_ok=True)
    incomplete = target.with_name(target.name + ".incomplete")
    command = [
        shutil.which("curl") or "curl",
        "-L",
        "--fail",
        "--retry",
        "12",
        "--retry-all-errors",
        "--retry-delay",
        "3",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(incomplete),
        url,
    ]
    subprocess.run(command, check=True)
    verify_size(incomplete, spec)
    os.replace(incomplete, target)


def download_osnet(target: Path, endpoint: str) -> None:
    if target.is_file() and target.stat().st_size == OSNET_SIZE and sha256(target) == OSNET_SHA256:
        return
    url = (
        f"{endpoint.rstrip('/')}/{OSNET_REPO}/resolve/{OSNET_REVISION}/"
        "osnet_x1_0_imagenet.pth"
    )
    download_url(url, target, {"size": OSNET_SIZE})
    actual_hash = sha256(target)
    if actual_hash != OSNET_SHA256:
        raise RuntimeError(
            f"OSNet checksum mismatch: {actual_hash} != {OSNET_SHA256}"
        )


def verify_sources(cache_root: Path) -> list[dict[str, Any]]:
    records = []
    source_root = cache_root / "sources"
    for name, expected in SOURCE_COMMITS.items():
        path = source_root / name
        if (path / ".git").is_dir():
            actual = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
        elif (path / ".source_commit").is_file():
            actual = (path / ".source_commit").read_text(encoding="ascii").strip()
        else:
            raise RuntimeError(f"source repository is missing: {path}")
        if actual != expected:
            raise RuntimeError(f"source commit mismatch for {name}: {actual} != {expected}")
        records.append(
            {
                "kind": "git",
                "name": name,
                "path": str(path.relative_to(cache_root)),
                "commit": actual,
                "ready": True,
            }
        )
    vace_inference = source_root / "VACE" / "vace" / "vace_wan_inference.py"
    if "--noise_overlap_in" not in vace_inference.read_text(encoding="utf-8"):
        raise RuntimeError("VACE JRDB proxy/noise-overlap patch is not applied")
    vace_model = source_root / "VACE" / "vace" / "models" / "wan" / "wan_vace.py"
    if "def move_vae" not in vace_model.read_text(encoding="utf-8"):
        raise RuntimeError("VACE memory-offload patch is not applied")
    vace_blocks = source_root / "VACE" / "vace" / "models" / "wan" / "modules" / "model.py"
    if "return tuple(all_c)" not in vace_blocks.read_text(encoding="utf-8"):
        raise RuntimeError("VACE hint-offload patch is not applied")
    wan_project = source_root / "Wan2.1" / "pyproject.toml"
    wan_project_text = wan_project.read_text(encoding="utf-8")
    if "opencv-python-headless>=4.9.0.80" not in wan_project_text:
        raise RuntimeError("Wan2.1 JRDB headless-runtime patch is not applied")
    for unused_dependency in ('"dashscope"', '"flash_attn"', '"gradio>=5.0.0"'):
        if unused_dependency in wan_project_text:
            raise RuntimeError(
                f"Wan2.1 unused UI/API dependency remains: {unused_dependency}"
            )
    wan_model = source_root / "Wan2.1" / "wan" / "modules" / "model.py"
    if "chunk_size = 16384" not in wan_model.read_text(encoding="utf-8"):
        raise RuntimeError("Wan2.1 memory-stable RoPE patch is not applied")
    return records


def verify_wan_files(model_dir: Path) -> None:
    for relative, expected_size in WAN_FILES.items():
        path = model_dir / relative
        if not path.is_file() or path.stat().st_size != expected_size:
            actual = path.stat().st_size if path.exists() else None
            raise RuntimeError(
                f"Wan checkpoint mismatch: {relative}, size={actual}, expected={expected_size}"
            )


def model_download(cache_root: Path, endpoint: str, workers: int) -> Path:
    model_dir = cache_root / "models" / "Wan2.1-VACE-14B"
    missing_bytes = sum(
        size
        for relative, size in WAN_FILES.items()
        if not (model_dir / relative).is_file()
        or (model_dir / relative).stat().st_size != size
    )
    free = shutil.disk_usage(cache_root).free
    reserve = 6 * 2**30
    if free < missing_bytes + reserve:
        raise RuntimeError(
            "insufficient space for Wan2.1-VACE-14B: "
            f"missing={missing_bytes / 2**30:.2f} GiB, free={free / 2**30:.2f} GiB, "
            f"required reserve={reserve / 2**30:.2f} GiB"
        )
    large_downloads: list[tuple[str, Path, int]] = []
    for relative, expected_size in WAN_FILES.items():
        target = model_dir / relative
        if target.is_file() and target.stat().st_size == expected_size:
            continue
        encoded = urllib.parse.quote(relative, safe="/")
        # Split large transfers across the official Hugging Face mirror and
        # ModelScope publication to avoid one CDN becoming the bottleneck.
        if expected_size >= 50_000_000:
            url = f"{MODELSCOPE_WAN_BASE}/{encoded}"
        else:
            url = f"{endpoint.rstrip('/')}/{WAN_REPO}/resolve/{WAN_REVISION}/{encoded}"
        if expected_size >= 50_000_000 and shutil.which("aria2c"):
            large_downloads.append((url, target, expected_size))
        else:
            download_url(url, target, {"size": expected_size})
    if large_downloads:
        aria_input = cache_root / "wan_vace_aria2_v2.txt"
        with aria_input.open("w", encoding="utf-8") as handle:
            for url, target, _ in large_downloads:
                target.parent.mkdir(parents=True, exist_ok=True)
                incomplete = target.with_name(target.name + ".incomplete")
                handle.write(url + "\n")
                handle.write(f"  dir={target.parent}\n")
                handle.write(f"  out={incomplete.name}\n")
        command = [
            "aria2c",
            "--continue=true",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            f"--max-concurrent-downloads={max(1, min(workers, 6))}",
            "--max-connection-per-server=4",
            "--split=4",
            "--min-split-size=8M",
            "--max-tries=0",
            "--retry-wait=3",
            "--timeout=60",
            "--summary-interval=20",
            f"--input-file={aria_input}",
        ]
        subprocess.run(command, check=True)
        for _, target, expected_size in large_downloads:
            incomplete = target.with_name(target.name + ".incomplete")
            verify_size(incomplete, {"size": expected_size})
            os.replace(incomplete, target)
    verify_wan_files(model_dir)
    return model_dir


def file_record(cache_root: Path, path: Path, source: str, revision: str | None = None) -> dict[str, Any]:
    return {
        "kind": "file",
        "path": str(path.relative_to(cache_root)),
        "source": source,
        "revision": revision,
        "size": path.stat().st_size,
        "sha256": sha256(path),
        "ready": True,
    }


def main() -> int:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    cache_root = Path(config["cache_root"])
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest or config.get("resources_manifest") or cache_root / "resources_manifest_v2.json")
    os.environ.update(
        {
            "HF_ENDPOINT": args.hf_endpoint,
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
            "TORCH_HOME": str(cache_root / "torch"),
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "resources": [],
        "errors": [],
        "ready": False,
    }
    atomic_json(manifest_path, manifest)
    try:
        manifest["resources"].extend(verify_sources(cache_root))
        if not args.skip_wan:
            model_dir = model_download(cache_root, args.hf_endpoint, args.max_workers)
        else:
            model_dir = cache_root / "models" / "Wan2.1-VACE-14B"
            verify_wan_files(model_dir)

        for relative, spec in DIRECT_FILES.items():
            target = cache_root / relative
            download_url(str(spec["url"]), target, spec)
        osnet = cache_root / "quality" / "osnet_x1_0_imagenet.pth"
        download_osnet(osnet, args.hf_endpoint)

        for relative in WAN_FILES:
            manifest["resources"].append(
                file_record(cache_root, model_dir / relative, WAN_REPO, WAN_REVISION)
            )
        for relative, spec in DIRECT_FILES.items():
            target = cache_root / relative
            verify_size(target, spec)
            manifest["resources"].append(
                file_record(cache_root, target, str(spec["source"]))
            )
        manifest["resources"].append(
            file_record(
                cache_root,
                osnet,
                f"{OSNET_REPO}/osnet_x1_0_imagenet.pth",
                OSNET_REVISION,
            )
        )
        incompletes = sorted(str(path) for path in cache_root.rglob("*.incomplete"))
        if incompletes:
            raise RuntimeError(f"incomplete downloads remain: {incompletes[:10]}")
        manifest["total_bytes"] = sum(
            int(item.get("size", 0)) for item in manifest["resources"]
        )
        manifest["notes"] = [
            "Seen-to-Scene does not publish its trained UNet/latent-refinement checkpoint at the pinned commit.",
            "The prepared integration therefore uses its documented RAFT + ProPainter flow-completion components and VACE latent/noise overlap conditioning.",
            "CUDA was not initialized by this download script.",
        ]
        manifest["ready"] = True
        atomic_json(manifest_path, manifest)
        print(
            f"READY: {len(manifest['resources'])} resources, "
            f"{manifest['total_bytes'] / 2**30:.2f} GiB, manifest={manifest_path}"
        )
        return 0
    except Exception as exc:
        manifest["errors"].append(str(exc))
        manifest["ready"] = False
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

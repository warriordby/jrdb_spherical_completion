from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    atomic_save_png,
    copy_shifted_labels,
    create_mot_view,
    frames,
    read_rgb,
    sequences,
)
from .geometry import (
    BandGeometry,
    output_synthetic_mask,
    proxy_to_output,
    source_to_proxy,
    spherical_quality_stats,
    verify_frame,
)
from .vace_runner import VaceRunner
from .windows import (
    ResumeState,
    atomic_write_json,
    canonical_fingerprint,
    file_sha256,
    make_windows,
    window_manifest,
)


def _selected_sequences(
    config: dict[str, Any], selected: list[str] | None
) -> list[Path]:
    candidates = sequences(Path(config["input_root"]))
    if selected:
        wanted = set(selected)
        unknown = sorted(wanted - {path.name for path in candidates})
        if unknown:
            raise ValueError(f"unknown sequences: {', '.join(unknown)}")
        candidates = [path for path in candidates if path.name in wanted]
    if not candidates:
        raise RuntimeError("no matching sequences")
    return candidates


def config_fingerprint(config: dict[str, Any]) -> str:
    relevant = {
        "geometry": config["geometry"],
        "generation": config["generation"],
        "temporal": config.get("temporal", {}),
        "output": config["output"],
    }
    return canonical_fingerprint(relevant)


def check_full_run_space(config: dict[str, Any]) -> dict[str, float]:
    required = float(config.get("production", {}).get("minimum_free_gib", 150))
    free = shutil.disk_usage(Path(config["output_root"]).parent).free / 2**30
    if free < required:
        raise RuntimeError(
            f"full production requires at least {required:.0f} GiB free; only {free:.2f} GiB is available"
        )
    return {"required_gib": required, "free_gib": free}


def check_production_quality_gate(config: dict[str, Any]) -> None:
    if not config.get("production", {}).get("allow_full_run_only_after_quality_gate", True):
        return
    summary_path = Path(config["output_root"]) / "quality" / "gate_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            f"full production is blocked until the representative quality gate exists: {summary_path}"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("production_approved", False):
        raise RuntimeError(
            "full production is blocked: quality/gate_summary.json does not contain "
            "production_approved=true after geometry, detector/ReID, temporal and TrackEval review"
        )


def prepare_windows(
    config: dict[str, Any],
    selected_sequences: list[str] | None,
    limit_frames: int | None,
    *,
    encode_inputs: bool = True,
) -> dict[str, Any]:
    fingerprint = config_fingerprint(config)
    output_root = Path(config["output_root"])
    work_root = output_root / ".work" / fingerprint
    runner = VaceRunner(config)
    frame_count = int(config["generation"].get("window_frames", 81))
    overlap = int(config["generation"].get("window_overlap", 17))
    result: dict[str, Any] = {"config_fingerprint": fingerprint, "sequences": {}}
    for sequence in _selected_sequences(config, selected_sequences):
        input_frames = frames(sequence)
        if limit_frames is not None:
            input_frames = input_frames[:limit_frames]
        manifest = window_manifest(
            sequence.name,
            input_frames,
            frame_count,
            overlap,
            fingerprint,
            hash_files=False,
        )
        sequence_work = work_root / sequence.name
        atomic_write_json(sequence_work / "windows.json", manifest)
        if encode_inputs:
            for window in make_windows(len(input_frames), frame_count, overlap):
                runner.stage_window(sequence.name, input_frames, window, sequence_work)
        result["sequences"][sequence.name] = {
            "frames": len(input_frames),
            "windows": len(manifest["windows"]),
            "manifest": str(sequence_work / "windows.json"),
            "inputs_encoded": encode_inputs,
        }
    atomic_write_json(work_root / "prepare_report.json", result)
    return result


def _edge_proxy(source: np.ndarray, geometry: BandGeometry) -> np.ndarray:
    proxy = source_to_proxy(
        source,
        geometry.proxy_width,
        geometry.proxy_height,
        geometry.north_latitude,
        geometry.south_latitude,
    )
    start, end = geometry.proxy_observed_rows
    for row in range(start):
        proxy[start - 1 - row] = proxy[min(start + row, end - 1)]
    for row in range(geometry.proxy_height - end):
        proxy[end + row] = proxy[max(start, end - 1 - row)]
    return proxy


def _write_final_frame(
    config: dict[str, Any],
    sequence_name: str,
    input_path: Path,
    proxy: np.ndarray,
) -> tuple[Path, dict[str, float]]:
    geometry = BandGeometry.from_config(config["geometry"])
    output_root = Path(config["output_root"])
    source = read_rgb(input_path)
    completed = proxy_to_output(
        proxy,
        source,
        geometry.output_height,
        geometry.north_latitude,
        geometry.south_latitude,
    )
    output_path = (
        output_root
        / "images"
        / "image_stitched"
        / sequence_name
        / f"{input_path.stem}.png"
    )
    compress = int(config["output"].get("png_compress_level", 4))
    atomic_save_png(output_path, completed, compress_level=compress)
    decoded = read_rgb(output_path)
    top_rows, _ = geometry.output_observed_rows
    errors = verify_frame(decoded, source, top_rows, geometry.output_height)
    if errors:
        raise RuntimeError(f"invalid output {output_path}: {'; '.join(errors)}")
    mask_path = (
        output_root / "synthetic_masks" / sequence_name / f"{input_path.stem}.png"
    )
    atomic_save_png(
        mask_path,
        output_synthetic_mask(
            geometry.output_width,
            geometry.output_height,
            geometry.north_latitude,
            geometry.south_latitude,
        ),
        compress_level=compress,
    )
    return output_path, spherical_quality_stats(decoded, source, top_rows)


def run(
    config: dict[str, Any],
    backend_name: str,
    selected_sequences: list[str] | None,
    limit_frames: int | None,
) -> None:
    if backend_name not in {"vace14b", "edge"}:
        raise ValueError("the frozen production plan supports vace14b; edge is diagnostic only")
    if backend_name == "vace14b" and selected_sequences is None and limit_frames is None:
        check_full_run_space(config)
        check_production_quality_gate(config)

    fingerprint = config_fingerprint(config)
    geometry = BandGeometry.from_config(config["geometry"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    top_rows, _ = geometry.output_observed_rows
    list(copy_shifted_labels(Path(config["input_root"]), output_root, top_rows))
    runner = VaceRunner(config) if backend_name == "vace14b" else None
    if runner is not None:
        readiness = runner.readiness(require_cuda=True)
        if not readiness["ready"]:
            raise RuntimeError(f"VACE GPU runtime is not ready: {json.dumps(readiness)}")

    completed_sequences: list[str] = []
    for sequence in _selected_sequences(config, selected_sequences):
        input_frames = frames(sequence)
        if limit_frames is not None:
            input_frames = input_frames[:limit_frames]
        if not input_frames:
            continue
        frame_count = int(config["generation"].get("window_frames", 81))
        overlap = int(config["generation"].get("window_overlap", 17))
        work_dir = output_root / ".work" / fingerprint / sequence.name
        proxy_dir = work_dir / "proxy_outputs"
        metadata_dir = output_root / "metadata"
        quality_dir = output_root / "quality"
        manifest = window_manifest(
            sequence.name,
            input_frames,
            frame_count,
            overlap,
            fingerprint,
            hash_files=False,
        )
        atomic_write_json(work_dir / "windows.json", manifest)
        state = ResumeState(work_dir / "state.json", sequence.name, fingerprint)
        frame_quality: dict[str, dict[str, float]] = {}

        if backend_name == "edge":
            for input_path in input_frames:
                output_path, stats = _write_final_frame(
                    config, sequence.name, input_path, _edge_proxy(read_rgb(input_path), geometry)
                )
                state.mark_frame(input_path.name, file_sha256(output_path))
                frame_quality[input_path.name] = stats
        else:
            assert runner is not None
            for window in make_windows(len(input_frames), frame_count, overlap):
                if state.window_complete(window.name):
                    committed_inputs = input_frames[
                        window.start + window.commit_from : window.stop
                    ]
                    committed_outputs = [
                        output_root
                        / "images"
                        / "image_stitched"
                        / sequence.name
                        / f"{path.stem}.png"
                        for path in committed_inputs
                    ]
                    committed_proxies = [
                        proxy_dir / f"{path.stem}.png" for path in committed_inputs
                    ]
                    if all(path.is_file() for path in committed_outputs + committed_proxies):
                        continue
                state.mark_window(window.name, "running")
                details = runner.generate_window(
                    sequence.name, input_frames, window, work_dir, proxy_dir
                )
                for name in details["committed"]:
                    stem = Path(name).stem
                    input_path = next(path for path in input_frames if path.stem == stem)
                    output_path, stats = _write_final_frame(
                        config, sequence.name, input_path, read_rgb(proxy_dir / name)
                    )
                    state.mark_frame(input_path.name, file_sha256(output_path))
                    frame_quality[input_path.name] = stats
                state.mark_window(window.name, "complete", **details)

        image_paths = [
            output_root / "images" / "image_stitched" / sequence.name / f"{path.stem}.png"
            for path in input_frames
        ]
        if config["output"].get("mot_view", True):
            create_mot_view(
                output_root,
                sequence.name,
                image_paths,
                int(config["output"].get("fps", 15)),
                geometry.output_observed_rows,
            )
        input_hashes = {path.name: file_sha256(path) for path in input_frames}
        metadata = {
            "schema_version": 2,
            "sequence": sequence.name,
            "source_root": str(sequence),
            "input_hashes": input_hashes,
            "backend": backend_name,
            "model": config["generation"].get("model"),
            "model_revision": config["generation"].get("model_revision"),
            "vace_commit": config["generation"].get("vace_commit"),
            "proxy_size": [geometry.proxy_width, geometry.proxy_height],
            "observed_latitudes": [geometry.south_latitude, geometry.north_latitude],
            "seed": int(config["generation"].get("seed", 3407)),
            "window_frames": frame_count,
            "window_overlap": overlap,
            "prompt": config["generation"].get("prompt"),
            "config_fingerprint": fingerprint,
            "synthetic_regions_have_no_ground_truth": True,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "package_version": "0.2.0",
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(metadata_dir / f"{sequence.name}.json", metadata)
        atomic_write_json(
            quality_dir / f"{sequence.name}.json",
            {
                "schema_version": 2,
                "sequence": sequence.name,
                "geometry_verified": True,
                "frame_metrics": frame_quality,
                "mot_gate_status": "pending_quality_command",
            },
        )
        completed_sequences.append(sequence.name)

    atomic_write_json(
        output_root / "manifest.json",
        {
            "schema_version": 2,
            "name": "JRDB-Spherical-VFOV180-VACE",
            "source": str(Path(config["input_root"])),
            "image_size": [geometry.output_width, geometry.output_height],
            "observed_band_y": list(geometry.output_observed_rows),
            "proxy_size": [geometry.proxy_width, geometry.proxy_height],
            "backend": backend_name,
            "config_fingerprint": fingerprint,
            "sequences": completed_sequences,
            "synthetic_regions_have_no_ground_truth": True,
        },
    )


def verify_dataset(
    config: dict[str, Any],
    selected_sequences: list[str] | None = None,
    limit_frames: int | None = None,
) -> tuple[int, list[str], dict[str, Any]]:
    geometry = BandGeometry.from_config(config["geometry"])
    input_root = Path(config["input_root"])
    output_root = Path(config["output_root"])
    top_rows, _ = geometry.output_observed_rows
    errors: list[str] = []
    count = 0
    stats: dict[str, Any] = {}
    for sequence in _selected_sequences(config, selected_sequences):
        input_frames = frames(sequence)
        if limit_frames is not None:
            input_frames = input_frames[:limit_frames]
        sequence_stats = []
        output_dir = output_root / "images" / "image_stitched" / sequence.name
        actual_names = {path.name for path in output_dir.glob("*.png")} if output_dir.exists() else set()
        expected_names = {f"{path.stem}.png" for path in input_frames}
        if limit_frames is None and actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            if missing:
                errors.append(f"{sequence.name}: missing output names {missing[:10]}")
            if extra:
                errors.append(f"{sequence.name}: unexpected output names {extra[:10]}")
        for input_path in input_frames:
            output_path = output_dir / f"{input_path.stem}.png"
            mask_path = output_root / "synthetic_masks" / sequence.name / f"{input_path.stem}.png"
            if not output_path.exists():
                errors.append(f"missing {output_path}")
                continue
            source = read_rgb(input_path)
            output = read_rgb(output_path)
            frame_errors = verify_frame(
                output,
                source,
                top_rows,
                geometry.output_height,
                seam_ratio_limit=float(config.get("quality", {}).get("seam_ratio_limit", 2.0)),
            )
            errors.extend(f"{output_path}: {error}" for error in frame_errors)
            expected_mask = output_synthetic_mask(
                geometry.output_width,
                geometry.output_height,
                geometry.north_latitude,
                geometry.south_latitude,
            )
            if not mask_path.exists():
                errors.append(f"missing {mask_path}")
            else:
                mask = read_rgb(mask_path)[:, :, 0]
                if not np.array_equal(mask, expected_mask):
                    errors.append(f"{mask_path}: synthetic mask content is invalid")
            sequence_stats.append(spherical_quality_stats(output, source, top_rows))
            count += 1
        stats[sequence.name] = sequence_stats
    return count, errors, stats

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import cache_env, load_config
from .data import frames, sequences


def _apply_cache(config: dict[str, Any]) -> None:
    for key, value in cache_env(config).items():
        os.environ.setdefault(key, value)


def command_inspect(config: dict[str, Any]) -> int:
    input_root = Path(config["input_root"])
    found = sequences(input_root)
    counts = {sequence.name: len(frames(sequence)) for sequence in found}
    sample = next((frame for sequence in found for frame in frames(sequence)), None)
    payload: dict[str, object] = {
        "input_root": str(input_root),
        "sequences": len(found),
        "frames": sum(counts.values()),
        "per_sequence": counts,
        "free_gib": round(shutil.disk_usage(input_root).free / 2**30, 2),
    }
    if sample:
        from PIL import Image

        with Image.open(sample) as image:
            payload["sample"] = {
                "path": str(sample),
                "size": list(image.size),
                "mode": image.mode,
            }
    print(json.dumps(payload, indent=2))
    return 0


def _resource_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config.get("resources_manifest", Path(config["cache_root"]) / "resources_manifest_v2.json"))
    if not path.is_file():
        return {"path": str(path), "ready": False, "error": "manifest missing"}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return {
            "path": str(path),
            "ready": bool(value.get("ready")),
            "resource_count": len(value.get("resources", [])),
            "total_bytes": value.get("total_bytes", 0),
            "errors": value.get("errors", []),
        }
    except Exception as exc:
        return {"path": str(path), "ready": False, "error": str(exc)}


def command_doctor(config: dict[str, Any], require_cuda: bool) -> int:
    from .quality import quality_resources
    from .vace_runner import VaceRunner

    runner = VaceRunner(config)
    payload = {
        "vace": runner.readiness(require_cuda=require_cuda),
        "resource_manifest": _resource_manifest(config),
        "quality_resources": quality_resources(config),
        "disk": {
            "free_gib": round(shutil.disk_usage(Path(config["output_root"]).parent).free / 2**30, 2),
            "production_minimum_gib": float(
                config.get("production", {}).get("minimum_free_gib", 150)
            ),
        },
    }
    payload["ready"] = (
        payload["vace"]["ready"]
        and payload["resource_manifest"]["ready"]
        and all(item["ready"] for item in payload["quality_resources"].values())
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["ready"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jrdb-sphere")
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect")
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--require-cuda", action="store_true")

    stage = subparsers.add_parser("stage")
    stage.add_argument("--sequences", nargs="*")
    stage.add_argument("--limit-frames", type=int)
    stage.add_argument("--manifest-only", action="store_true")
    stage.add_argument("--output-root")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--backend", choices=["vace14b", "edge"], default=None)
    run_parser.add_argument("--sequences", nargs="*")
    run_parser.add_argument("--limit-frames", type=int)
    run_parser.add_argument("--output-root")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--sequences", nargs="*")
    verify_parser.add_argument("--limit-frames", type=int)
    verify_parser.add_argument("--output-root")

    quality_parser = subparsers.add_parser("quality")
    quality_parser.add_argument("--sequences", nargs="*")
    quality_parser.add_argument("--limit-frames", type=int)
    quality_parser.add_argument("--skip-models", action="store_true")
    quality_parser.add_argument("--output-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    _apply_cache(config)
    if getattr(args, "output_root", None):
        config["output_root"] = str(Path(args.output_root).expanduser().resolve())
    if args.command == "inspect":
        return command_inspect(config)
    if args.command == "doctor":
        return command_doctor(config, args.require_cuda)
    if args.command == "stage":
        from .pipeline import prepare_windows

        report = prepare_windows(
            config,
            args.sequences,
            args.limit_frames,
            encode_inputs=not args.manifest_only,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "run":
        from .pipeline import run

        run(
            config,
            args.backend or config["generation"].get("backend", "vace14b"),
            args.sequences,
            args.limit_frames,
        )
        return 0
    if args.command == "verify":
        from .pipeline import verify_dataset

        count, errors, stats = verify_dataset(
            config, args.sequences, args.limit_frames
        )
        print(
            json.dumps(
                {
                    "verified_frames": count,
                    "errors": errors[:100],
                    "error_count": len(errors),
                    "statistics": stats,
                },
                indent=2,
            )
        )
        return 1 if errors else 0
    if args.command == "quality":
        from .quality import quality_gate

        result = quality_gate(
            config,
            args.sequences,
            args.limit_frames,
            run_models=not args.skip_models,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

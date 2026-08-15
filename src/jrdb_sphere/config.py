from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .geometry import BandGeometry


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"input_root", "output_root", "cache_root", "geometry", "generation", "output"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"config is missing required fields: {', '.join(missing)}")
    geometry = BandGeometry.from_config(config["geometry"])
    if geometry.output_width != geometry.input_width:
        raise ValueError("final output width must equal the JRDB input width")
    if geometry.proxy_width != geometry.proxy_height * 2:
        raise ValueError("proxy canvas must be a standard 2:1 ERP")
    start, end = geometry.output_observed_rows
    if end - start != geometry.input_height:
        raise ValueError(
            "output latitude sampling must exactly fit the observed source rows"
        )
    frame_count = int(config["generation"].get("window_frames", 81))
    overlap = int(config["generation"].get("window_overlap", 17))
    if (frame_count - 1) % 4:
        raise ValueError("generation.window_frames must be 4n+1")
    if overlap < 0 or overlap >= frame_count:
        raise ValueError("generation.window_overlap must be smaller than window_frames")


def cache_env(config: dict[str, Any]) -> dict[str, str]:
    root = Path(config["cache_root"]).resolve()
    return {
        "HF_HOME": str(root / "huggingface"),
        "HF_HUB_CACHE": str(root / "huggingface" / "hub"),
        "TORCH_HOME": str(root / "torch"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BandGeometry:
    input_width: int = 3760
    input_height: int = 480
    output_width: int = 3760
    output_height: int = 720
    proxy_width: int = 1440
    proxy_height: int = 720
    north_latitude: float = 60.0
    south_latitude: float = -60.0

    @classmethod
    def from_config(cls, value: dict) -> "BandGeometry":
        return cls(
            input_width=int(value["input_width"]),
            input_height=int(value["input_height"]),
            output_width=int(value.get("output_width", value["input_width"])),
            output_height=int(value["output_height"]),
            proxy_width=int(value.get("proxy_width", 1440)),
            proxy_height=int(value.get("proxy_height", 720)),
            north_latitude=float(value.get("north_latitude", 60.0)),
            south_latitude=float(value.get("south_latitude", -60.0)),
        )

    @property
    def output_observed_rows(self) -> tuple[int, int]:
        return observed_row_bounds(
            self.output_height, self.north_latitude, self.south_latitude
        )

    @property
    def proxy_observed_rows(self) -> tuple[int, int]:
        return observed_row_bounds(
            self.proxy_height, self.north_latitude, self.south_latitude
        )


def pixel_center_latitudes(height: int) -> np.ndarray:
    """ERP latitude for each pixel centre, north to south."""
    return 90.0 - (np.arange(height, dtype=np.float64) + 0.5) * (180.0 / height)


def pixel_center_longitudes(width: int) -> np.ndarray:
    """ERP longitude for each pixel centre in [-180, 180)."""
    return (np.arange(width, dtype=np.float64) + 0.5) * (360.0 / width) - 180.0


def observed_row_bounds(
    height: int, north_latitude: float = 60.0, south_latitude: float = -60.0
) -> tuple[int, int]:
    latitudes = pixel_center_latitudes(height)
    rows = np.flatnonzero(
        (latitudes <= north_latitude + 1e-9)
        & (latitudes >= south_latitude - 1e-9)
    )
    if not len(rows):
        raise ValueError("observed latitude band does not cover a pixel row")
    return int(rows[0]), int(rows[-1] + 1)


def _periodic_x_map(source_width: int, target_width: int, target_x_offset: int = 0) -> np.ndarray:
    x = np.arange(target_width, dtype=np.float32) + float(target_x_offset)
    return np.mod((x + 0.5) * source_width / target_width - 0.5, source_width).astype(
        np.float32
    )


def _remap_rgb(source: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def source_to_proxy(
    source: np.ndarray,
    proxy_width: int = 1440,
    proxy_height: int = 720,
    north_latitude: float = 60.0,
    south_latitude: float = -60.0,
    fill_value: int = 127,
) -> np.ndarray:
    """Map a complete-longitude latitude band onto a standard 2:1 ERP canvas.

    Longitude and latitude are mapped using pixel-centre coordinates. Unknown
    latitudes remain neutral gray, as expected by VACE's masked-video input.
    """
    source = np.asarray(source, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"expected HxWx3 source, got {source.shape}")
    result = np.full((proxy_height, proxy_width, 3), fill_value, dtype=np.uint8)
    latitudes = pixel_center_latitudes(proxy_height)
    known = (latitudes <= north_latitude + 1e-9) & (
        latitudes >= south_latitude - 1e-9
    )
    y_rows = np.flatnonzero(known)
    if not len(y_rows):
        return result
    span = north_latitude - south_latitude
    source_y = (
        (north_latitude - latitudes[y_rows]) * source.shape[0] / span - 0.5
    ).astype(np.float32)
    source_y = np.clip(source_y, 0, source.shape[0] - 1)
    source_x = _periodic_x_map(source.shape[1], proxy_width)
    map_x, map_y = np.meshgrid(source_x, source_y)
    result[y_rows] = _remap_rgb(source, map_x, map_y)
    return result


def proxy_to_output(
    proxy: np.ndarray,
    source: np.ndarray,
    output_height: int = 720,
    north_latitude: float = 60.0,
    south_latitude: float = -60.0,
) -> np.ndarray:
    """Inverse-map only synthetic latitudes and copy the source band verbatim."""
    proxy = np.asarray(proxy, dtype=np.uint8)
    source = np.asarray(source, dtype=np.uint8)
    output_width = source.shape[1]
    out_latitudes = pixel_center_latitudes(output_height)
    proxy_y = ((90.0 - out_latitudes) * proxy.shape[0] / 180.0 - 0.5).astype(
        np.float32
    )
    proxy_y = np.clip(proxy_y, 0, proxy.shape[0] - 1)
    proxy_x = _periodic_x_map(proxy.shape[1], output_width)
    map_x, map_y = np.meshgrid(proxy_x, proxy_y)
    result = _remap_rgb(proxy, map_x, map_y)

    start, end = observed_row_bounds(
        output_height, north_latitude, south_latitude
    )
    if end - start != source.shape[0]:
        raise ValueError(
            "output sampling does not provide one row per observed source row: "
            f"output band={end - start}, source={source.shape[0]}"
        )
    result[start:end] = source
    return result


def proxy_generation_mask(
    width: int,
    height: int,
    north_latitude: float = 60.0,
    south_latitude: float = -60.0,
) -> np.ndarray:
    """VACE mask: white is generated, black is retained."""
    mask = np.full((height, width), 255, dtype=np.uint8)
    start, end = observed_row_bounds(height, north_latitude, south_latitude)
    mask[start:end] = 0
    return mask


def output_synthetic_mask(
    width: int,
    height: int,
    north_latitude: float = 60.0,
    south_latitude: float = -60.0,
) -> np.ndarray:
    return proxy_generation_mask(width, height, north_latitude, south_latitude)


def generation_mask(
    width: int, input_height: int, output_height: int, top_rows: int
) -> Image.Image:
    """Backward-compatible fixed-row mask used by legacy baselines."""
    mask = np.full((output_height, width), 255, dtype=np.uint8)
    mask[top_rows : top_rows + input_height] = 0
    return Image.fromarray(mask, mode="L")


def embed_source(source: Image.Image, output_height: int, top_rows: int) -> np.ndarray:
    rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    canvas = np.empty((output_height, rgb.shape[1], 3), dtype=np.uint8)
    canvas[:top_rows] = rgb[0:1]
    canvas[top_rows : top_rows + rgb.shape[0]] = rgb
    canvas[top_rows + rgb.shape[0] :] = rgb[-1:]
    return canvas


def periodic_take(array: np.ndarray, start: int, width: int) -> np.ndarray:
    indices = np.arange(start, start + width) % array.shape[1]
    return array[:, indices]


def roll_longitude(array: np.ndarray, pixels: int) -> np.ndarray:
    return np.roll(array, shift=int(pixels), axis=1)


def seam_repair_blend(
    primary: np.ndarray,
    rolled_back: np.ndarray,
    synthetic_mask: np.ndarray,
) -> np.ndarray:
    """Blend two W/2-separated VACE passes only inside generated latitudes."""
    if primary.shape != rolled_back.shape:
        raise ValueError("seam-repair inputs must have identical shapes")
    width = primary.shape[1]
    x = np.arange(width, dtype=np.float32)
    primary_weight = np.sin(np.pi * x / width) ** 2
    weight = primary_weight[None, :, None]
    blended = np.clip(
        primary.astype(np.float32) * weight
        + rolled_back.astype(np.float32) * (1.0 - weight),
        0,
        255,
    ).astype(np.uint8)
    result = primary.copy()
    selected = synthetic_mask.astype(bool)
    result[selected] = blended[selected]
    return result


def cosine_weights(width: int, overlap: int) -> np.ndarray:
    weights = np.ones(width, dtype=np.float32)
    if overlap <= 0:
        return weights
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, overlap, dtype=np.float32))
    weights[:overlap] = ramp
    weights[-overlap:] = ramp[::-1]
    return np.maximum(weights, 1e-4)


def tile_starts(panorama_width: int, tile_width: int, overlap: int) -> list[int]:
    if tile_width >= panorama_width:
        return [0]
    step = tile_width - overlap
    count = int(np.ceil(panorama_width / step))
    return [round(i * panorama_width / count) for i in range(count)]


def enforce_contract(generated: np.ndarray, source: np.ndarray, top_rows: int) -> np.ndarray:
    """Restore the immutable observed band without altering synthetic pixels."""
    result = np.asarray(generated, dtype=np.uint8).copy()
    result[top_rows : top_rows + source.shape[0]] = source
    return result


def spherical_quality_stats(
    output: np.ndarray, source: np.ndarray, top_rows: int
) -> dict[str, float]:
    synthetic_rows = np.r_[np.arange(top_rows), np.arange(top_rows + source.shape[0], output.shape[0])]
    synthetic = output[synthetic_rows].astype(np.float32)
    seam = np.abs(synthetic[:, 0] - synthetic[:, -1]).mean(axis=1)
    adjacent = np.abs(synthetic[:, 1:] - synthetic[:, :-1]).mean(axis=2).reshape(-1)
    return {
        "synthetic_seam_mae": float(seam.mean()) if seam.size else 0.0,
        "synthetic_seam_p95": float(np.percentile(seam, 95)) if seam.size else 0.0,
        "adjacent_gradient_p95": float(np.percentile(adjacent, 95)) if adjacent.size else 0.0,
        "north_pole_row_std": float(output[0].astype(np.float32).std()),
        "south_pole_row_std": float(output[-1].astype(np.float32).std()),
    }


def verify_frame(
    output: np.ndarray,
    source: np.ndarray,
    top_rows: int,
    output_height: int,
    *,
    seam_ratio_limit: float | None = None,
) -> list[str]:
    errors: list[str] = []
    expected = (output_height, source.shape[1], 3)
    if output.shape != expected:
        return [f"shape is {output.shape}, expected {expected}"]
    center = output[top_rows : top_rows + source.shape[0]]
    if not np.array_equal(center, source):
        delta = np.abs(center.astype(np.int16) - source.astype(np.int16))
        errors.append(f"observed band changed (max error {int(delta.max())})")
    if seam_ratio_limit is not None:
        stats = spherical_quality_stats(output, source, top_rows)
        baseline = max(stats["adjacent_gradient_p95"], 1.0)
        if stats["synthetic_seam_p95"] > baseline * seam_ratio_limit:
            errors.append(
                "synthetic ERP seam exceeds local-gradient limit "
                f"({stats['synthetic_seam_p95']:.3f} > {baseline * seam_ratio_limit:.3f})"
            )
    return errors

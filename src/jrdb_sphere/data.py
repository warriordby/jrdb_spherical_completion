from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def stitched_root(input_root: Path) -> Path:
    candidates = [
        input_root / "images" / "image_stitched",
        input_root / "test_images" / "images" / "image_stitched",
        input_root / "train_images" / "images" / "image_stitched",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"cannot find images/image_stitched below {input_root}")


def sequences(input_root: Path) -> list[Path]:
    return sorted(path for path in stitched_root(input_root).iterdir() if path.is_dir())


def frames(sequence: Path) -> list[Path]:
    return sorted(path for path in sequence.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def atomic_save_png(path: Path, image: np.ndarray, compress_level: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}.png")
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(
        temporary, format="PNG", compress_level=compress_level
    )
    os.replace(temporary, path)


def write_seqinfo(
    sequence_dir: Path,
    name: str,
    length: int,
    width: int,
    height: int,
    fps: int,
) -> None:
    sequence_dir.mkdir(parents=True, exist_ok=True)
    target = sequence_dir / "seqinfo.ini"
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="ascii") as handle:
        handle.write(
            "[Sequence]\n"
            f"name={name}\n"
            "imDir=img1\n"
            f"frameRate={fps}\n"
            f"seqLength={length}\n"
            f"imWidth={width}\n"
            f"imHeight={height}\n"
            "imExt=.png\n"
        )
    os.replace(temporary, target)


def create_mot_view(
    output_root: Path,
    sequence_name: str,
    image_paths: list[Path],
    fps: int,
    observed_rows: tuple[int, int] | None = None,
) -> None:
    if not image_paths:
        return
    mot_dir = output_root / "mot" / sequence_name
    img1 = mot_dir / "img1"
    img1.mkdir(parents=True, exist_ok=True)
    expected: set[str] = set()
    for image_path in image_paths:
        target = img1 / image_path.name
        expected.add(target.name)
        relative = os.path.relpath(image_path, start=target.parent)
        if target.is_symlink() and os.readlink(target) == relative:
            continue
        target.unlink(missing_ok=True)
        target.symlink_to(relative)
    for existing in img1.iterdir():
        if existing.name not in expected and existing.is_symlink():
            existing.unlink()
    with Image.open(image_paths[0]) as sample:
        write_seqinfo(
            mot_dir,
            sequence_name,
            len(image_paths),
            sample.width,
            sample.height,
            fps,
        )
        if observed_rows is not None:
            top, bottom = observed_rows
            ignore_path = mot_dir / "synthetic_ignore.txt"
            temporary = ignore_path.with_name(ignore_path.name + f".tmp-{os.getpid()}")
            with temporary.open("w", encoding="ascii") as handle:
                for frame_index in range(1, len(image_paths) + 1):
                    # MOTChallenge-like rows kept separate from gt.txt. Consumers
                    # must explicitly load these as ignore regions.
                    handle.write(
                        f"{frame_index},-1,0,0,{sample.width},{top},0,-1,-1,-1\n"
                    )
                    handle.write(
                        f"{frame_index},-1,0,{bottom},{sample.width},{sample.height - bottom},0,-1,-1,-1\n"
                    )
            os.replace(temporary, ignore_path)


def shift_json_y(value: object, offset: int) -> object:
    """Shift common JRDB/COCO 2D fields without touching 3D data."""
    if isinstance(value, list):
        return [shift_json_y(item, offset) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: shift_json_y(item, offset) for key, item in value.items()}
    if isinstance(value.get("bbox"), list) and len(value["bbox"]) >= 4:
        result["bbox"] = list(value["bbox"])
        result["bbox"][1] += offset
    if isinstance(value.get("box"), list) and len(value["box"]) >= 4:
        result["box"] = list(value["box"])
        result["box"][1] += offset
    keypoints = value.get("keypoints")
    if isinstance(keypoints, list) and len(keypoints) % 3 == 0:
        shifted = list(keypoints)
        for index in range(0, len(shifted), 3):
            if shifted[index + 2] != 0:
                shifted[index + 1] += offset
        result["keypoints"] = shifted
    segmentation = value.get("segmentation")
    if isinstance(segmentation, list):
        shifted_segments = []
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) % 2:
                shifted_segments.append(polygon)
                continue
            shifted = list(polygon)
            for index in range(1, len(shifted), 2):
                shifted[index] += offset
            shifted_segments.append(shifted)
        result["segmentation"] = shifted_segments
    return result


def copy_shifted_labels(input_root: Path, output_root: Path, offset: int) -> Iterator[Path]:
    """Copy labels for the observed band only; synthetic bands remain unlabelled."""
    for path in input_root.rglob("*.json"):
        if "labels_2d" not in path.parts:
            continue
        relative = path.relative_to(input_root)
        target = output_root / "observed_labels_only" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(shift_json_y(data, offset), handle, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, target)
        yield target

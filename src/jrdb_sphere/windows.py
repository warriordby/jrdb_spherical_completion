from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Window:
    index: int
    start: int
    stop: int
    valid_frames: int
    frame_count: int
    overlap_with_previous: int

    @property
    def name(self) -> str:
        return f"window-{self.index:05d}-{self.start:06d}-{self.stop - 1:06d}"

    @property
    def padded_indices(self) -> list[int]:
        indices = list(range(self.start, self.stop))
        if not indices:
            return []
        indices.extend([indices[-1]] * (self.frame_count - len(indices)))
        return indices

    @property
    def commit_from(self) -> int:
        return self.overlap_with_previous


def make_windows(length: int, frame_count: int = 81, overlap: int = 17) -> list[Window]:
    if length < 0:
        raise ValueError("length must be non-negative")
    if frame_count <= 0 or (frame_count - 1) % 4:
        raise ValueError("VACE frame_count must be 4n+1")
    if overlap < 0 or overlap >= frame_count:
        raise ValueError("overlap must be in [0, frame_count)")
    if length == 0:
        return []
    step = frame_count - overlap
    result: list[Window] = []
    for index, start in enumerate(range(0, length, step)):
        stop = min(start + frame_count, length)
        previous_overlap = 0 if index == 0 else min(overlap, stop - start)
        result.append(
            Window(
                index=index,
                start=start,
                stop=stop,
                valid_frames=stop - start,
                frame_count=frame_count,
                overlap_with_previous=previous_overlap,
            )
        )
        if stop == length:
            break
    return result


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_inventory(paths: Iterable[Path], hash_files: bool = False) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        stat = path.stat()
        item: dict[str, Any] = {
            "name": path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if hash_files:
            item["sha256"] = file_sha256(path)
        result.append(item)
    return result


def window_manifest(
    sequence: str,
    frame_paths: list[Path],
    frame_count: int,
    overlap: int,
    config_fingerprint: str,
    *,
    hash_files: bool = False,
) -> dict[str, Any]:
    inventory = frame_inventory(frame_paths, hash_files=hash_files)
    windows = make_windows(len(frame_paths), frame_count, overlap)
    return {
        "schema_version": 2,
        "sequence": sequence,
        "frame_count": len(frame_paths),
        "input_inventory": inventory,
        "input_inventory_fingerprint": canonical_fingerprint(inventory),
        "config_fingerprint": config_fingerprint,
        "window_size": frame_count,
        "window_overlap": overlap,
        "windows": [
            {
                **asdict(window),
                "name": window.name,
                "padded_indices": window.padded_indices,
                "frame_names": [frame_paths[i].name for i in window.padded_indices],
            }
            for window in windows
        ],
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ResumeState:
    def __init__(self, path: Path, sequence: str, fingerprint: str) -> None:
        self.path = path
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                self.value = json.load(handle)
            if self.value.get("config_fingerprint") != fingerprint:
                raise RuntimeError(
                    f"resume fingerprint mismatch for {sequence}: "
                    f"{self.value.get('config_fingerprint')} != {fingerprint}"
                )
        else:
            self.value = {
                "schema_version": 2,
                "sequence": sequence,
                "config_fingerprint": fingerprint,
                "windows": {},
                "frames": {},
            }

    def window_complete(self, name: str) -> bool:
        return self.value["windows"].get(name, {}).get("status") == "complete"

    def mark_window(self, name: str, status: str, **details: Any) -> None:
        self.value["windows"][name] = {"status": status, **details}
        atomic_write_json(self.path, self.value)

    def mark_frame(self, name: str, output_sha256: str) -> None:
        self.value["frames"][name] = {
            "status": "complete",
            "output_sha256": output_sha256,
        }
        atomic_write_json(self.path, self.value)

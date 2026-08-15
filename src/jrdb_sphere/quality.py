from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .data import frames, read_rgb, sequences
from .geometry import BandGeometry, spherical_quality_stats, verify_frame
from .windows import atomic_write_json


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=2)
    first_area = np.maximum(first[:, 2] - first[:, 0], 0) * np.maximum(
        first[:, 3] - first[:, 1], 0
    )
    second_area = np.maximum(second[:, 2] - second[:, 0], 0) * np.maximum(
        second[:, 3] - second[:, 1], 0
    )
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-6)


def greedy_iou_matches(
    first: np.ndarray, second: np.ndarray, threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    ious = box_iou(first, second)
    matches: list[tuple[int, int, float]] = []
    while ious.size:
        flat = int(np.argmax(ious))
        score = float(ious.flat[flat])
        if score < threshold:
            break
        left, right = np.unravel_index(flat, ious.shape)
        matches.append((int(left), int(right), score))
        ious[left, :] = -1
        ious[:, right] = -1
    return matches


class FixedPersonDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        quality = config.get("quality", {})
        self.torch = torch
        self.score_threshold = float(quality.get("person_score_threshold", 0.7))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None
        )
        checkpoint = Path(quality["detector_checkpoint"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
        tensor = self.transforms(tensor).to(self.device)
        with torch.inference_mode():
            output = self.model([tensor])[0]
        keep = (output["labels"] == 1) & (output["scores"] >= self.score_threshold)
        return {
            "boxes": output["boxes"][keep].detach().cpu().numpy(),
            "scores": output["scores"][keep].detach().cpu().numpy(),
        }


class FixedReID:
    def __init__(self, config: dict[str, Any]) -> None:
        import torch

        quality = config.get("quality", {})
        source = Path(quality["reid_source"])
        checkpoint = Path(quality["reid_checkpoint"])
        sys.path.insert(0, str(source))
        from torchreid.models import build_model
        from torchreid.utils import load_pretrained_weights

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            name=str(quality.get("reid_model", "osnet_x1_0")),
            num_classes=1,
            loss="softmax",
            pretrained=False,
            use_gpu=torch.cuda.is_available(),
        )
        load_pretrained_weights(self.model, str(checkpoint))
        self.model.eval().to(self.device)

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        import cv2

        crops = []
        for box in boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(image[y1:y2, x1:x2], (128, 256), interpolation=cv2.INTER_LINEAR)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )
            crops.append(crop.transpose(2, 0, 1))
        if not crops:
            return np.empty((0, 512), dtype=np.float32)
        tensor = self.torch.from_numpy(np.stack(crops)).to(self.device)
        with self.torch.inference_mode():
            features = self.model(tensor)
            features = self.torch.nn.functional.normalize(features, dim=1)
        return features.detach().cpu().numpy()


def _synthetic_detections(boxes: np.ndarray, top: int, bottom: int) -> np.ndarray:
    if not len(boxes):
        return boxes
    center_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
    return boxes[(center_y < top) | (center_y >= bottom)]


def _paired_reid_distances(
    reid: FixedReID,
    source: np.ndarray,
    completed: np.ndarray,
    source_boxes: np.ndarray,
    top: int,
) -> list[float]:
    if not len(source_boxes):
        return []
    output_boxes = source_boxes.copy()
    output_boxes[:, [1, 3]] += top
    source_features = reid(source, source_boxes)
    output_features = reid(completed, output_boxes)
    length = min(len(source_features), len(output_features))
    if not length:
        return []
    return (1.0 - (source_features[:length] * output_features[:length]).sum(axis=1)).tolist()


def quality_gate(
    config: dict[str, Any],
    selected_sequences: list[str] | None = None,
    limit_frames: int | None = None,
    *,
    run_models: bool = True,
) -> dict[str, Any]:
    geometry = BandGeometry.from_config(config["geometry"])
    top, bottom = geometry.output_observed_rows
    output_root = Path(config["output_root"])
    detector = FixedPersonDetector(config) if run_models else None
    reid = FixedReID(config) if run_models else None
    selected = set(selected_sequences or [])
    reports: dict[str, Any] = {}
    hard_failures = 0
    for sequence in sequences(Path(config["input_root"])):
        if selected and sequence.name not in selected:
            continue
        input_frames = frames(sequence)
        if limit_frames is not None:
            input_frames = input_frames[:limit_frames]
        new_people = 0
        reid_distances: list[float] = []
        geometry_errors: list[str] = []
        frame_stats: dict[str, Any] = {}
        for input_path in input_frames:
            output_path = (
                output_root
                / "images"
                / "image_stitched"
                / sequence.name
                / f"{input_path.stem}.png"
            )
            if not output_path.exists():
                geometry_errors.append(f"missing {output_path}")
                continue
            source = read_rgb(input_path)
            completed = read_rgb(output_path)
            errors = verify_frame(completed, source, top, geometry.output_height)
            geometry_errors.extend(f"{input_path.name}: {item}" for item in errors)
            stats = spherical_quality_stats(completed, source, top)
            if detector is not None and reid is not None:
                source_detection = detector(source)
                completed_detection = detector(completed)
                synthetic = _synthetic_detections(completed_detection["boxes"], top, bottom)
                new_people += len(synthetic)
                reid_distances.extend(
                    _paired_reid_distances(
                        reid, source, completed, source_detection["boxes"], top
                    )
                )
                stats["source_person_detections"] = int(len(source_detection["boxes"]))
                stats["synthetic_person_detections"] = int(len(synthetic))
            frame_stats[input_path.name] = stats
        reid_p95 = float(np.percentile(reid_distances, 95)) if reid_distances else 0.0
        passed = not geometry_errors and new_people == 0 and reid_p95 <= float(
            config.get("quality", {}).get("reid_cosine_p95_limit", 0.05)
        )
        if not passed:
            hard_failures += 1
        report = {
            "schema_version": 2,
            "sequence": sequence.name,
            "passed": passed,
            "geometry_errors": geometry_errors,
            "synthetic_person_detections": new_people,
            "reid_cosine_distance_p95": reid_p95,
            "frame_metrics": frame_stats,
            "detector_reid_executed": run_models,
            "trackeval_status": (
                "pending_ground_truth"
                if not config.get("quality", {}).get("mot_ground_truth")
                else "run_via_scripts/run_trackeval.sh"
            ),
            "synthetic_detections_are_ignore_only": True,
        }
        reports[sequence.name] = report
        atomic_write_json(output_root / "quality" / f"{sequence.name}.json", report)
    structural_passed = hard_failures == 0
    result = {
        "passed": structural_passed,
        "structural_passed": structural_passed,
        "production_approved": False,
        "production_approval_note": (
            "Set to true only after the representative temporal metrics and TrackEval "
            "HOTA/IDF1/MOTA gates have been reviewed."
        ),
        "failure_count": hard_failures,
        "sequences": reports,
    }
    atomic_write_json(output_root / "quality" / "gate_summary.json", result)
    return result


def quality_resources(config: dict[str, Any]) -> dict[str, Any]:
    quality = config.get("quality", {})
    paths = {
        "detector_checkpoint": Path(quality.get("detector_checkpoint", "")),
        "trackeval_source": Path(quality.get("trackeval_source", "")),
        "reid_source": Path(quality.get("reid_source", "")),
        "reid_checkpoint": Path(quality.get("reid_checkpoint", "")),
        "raft_checkpoint": Path(quality.get("raft_checkpoint", "")),
        "flow_completion_checkpoint": Path(
            quality.get("flow_completion_checkpoint", "")
        ),
    }
    return {
        key: {"path": str(path), "ready": bool(str(path)) and path.exists()}
        for key, path in paths.items()
    }

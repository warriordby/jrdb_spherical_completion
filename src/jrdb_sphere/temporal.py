from __future__ import annotations

import numpy as np


def latent_overlap_frames(frame_overlap: int, temporal_stride: int = 4) -> int:
    if frame_overlap <= 0:
        return 0
    return (frame_overlap - 1) // temporal_stride + 1


def _structural_feature(image: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    feature = cv2.resize(gray, (64, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    feature -= feature.mean()
    norm = np.linalg.norm(feature)
    if norm > 1e-6:
        feature /= norm
    return feature.reshape(-1)


def select_reference_indices(images: list[np.ndarray], count: int = 3) -> list[int]:
    """Select representative frames by structural coverage, not recency alone."""
    if not images or count <= 0:
        return []
    features = np.stack([_structural_feature(image) for image in images])
    count = min(count, len(images))
    mean = features.mean(axis=0)
    selected = [int(np.argmax(features @ mean))]
    while len(selected) < count:
        similarity = features @ features[selected].T
        nearest_similarity = similarity.max(axis=1)
        nearest_similarity[selected] = np.inf
        selected.append(int(np.argmin(nearest_similarity)))
    return sorted(selected)


class RAFTWarper:
    def __init__(self, cache_root: str) -> None:
        import os

        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        if not torch.cuda.is_available():
            raise RuntimeError("RAFT propagation requires CUDA")
        os.environ.setdefault("TORCH_HOME", cache_root + "/torch")
        self.torch = torch
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=False).eval().to("cuda")

    def flow(self, target: np.ndarray, source: np.ndarray, circular_pad: int) -> np.ndarray:
        torch = self.torch
        target_pad = np.pad(target, ((0, 0), (circular_pad, circular_pad), (0, 0)), mode="wrap")
        source_pad = np.pad(source, ((0, 0), (circular_pad, circular_pad), (0, 0)), mode="wrap")
        first = torch.from_numpy(target_pad).permute(2, 0, 1).unsqueeze(0).to("cuda")
        second = torch.from_numpy(source_pad).permute(2, 0, 1).unsqueeze(0).to("cuda")
        first, second = self.weights.transforms()(first, second)
        with torch.inference_mode():
            result = self.model(first, second)[-1][0]
        flow = result[:, :, circular_pad:-circular_pad].permute(1, 2, 0).float().cpu().numpy()
        return flow

    @staticmethod
    def warp(completed_source: np.ndarray, flow: np.ndarray, top_rows: int, input_height: int) -> np.ndarray:
        import cv2

        output_height, width = completed_source.shape[:2]
        full_flow = np.empty((output_height, width, 2), dtype=np.float32)
        full_flow[top_rows : top_rows + input_height] = flow
        full_flow[:top_rows] = flow[0:1]
        full_flow[top_rows + input_height :] = flow[-1:]
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(output_height, dtype=np.float32))
        map_x = np.mod(grid_x + full_flow[:, :, 0], width).astype(np.float32)
        map_y = np.clip(grid_y + full_flow[:, :, 1], 0, output_height - 1).astype(np.float32)
        return cv2.remap(completed_source, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def blend_warps(previous: np.ndarray, following: np.ndarray, fraction: float) -> np.ndarray:
    return np.clip(previous.astype(np.float32) * (1.0 - fraction) + following.astype(np.float32) * fraction, 0, 255).astype(np.uint8)

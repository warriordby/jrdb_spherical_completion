from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .data import atomic_save_png, read_rgb
from .geometry import (
    BandGeometry,
    proxy_generation_mask,
    roll_longitude,
    seam_repair_blend,
    source_to_proxy,
)
from .temporal import latent_overlap_frames, select_reference_indices
from .windows import Window, atomic_write_json, file_sha256


def sequence_seed(base_seed: int, sequence_name: str) -> int:
    suffix = int(hashlib.sha256(sequence_name.encode("utf-8")).hexdigest()[:8], 16)
    return (int(base_seed) + suffix) % (2**31 - 1)


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _encode_png_sequence(directory: Path, output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp-{os.getpid()}.mp4")
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(directory / "%05d.png"),
        "-c:v",
        "libx264rgb",
        "-crf",
        "0",
        "-preset",
        "fast",
        "-pix_fmt",
        "rgb24",
        str(temporary),
    ]
    _run(command)
    os.replace(temporary, output)


def _decode_video(video: Path, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.png"):
        stale.unlink()
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vsync",
        "0",
        str(directory / "%05d.png"),
    ]
    _run(command)
    return sorted(directory.glob("*.png"))


class VaceRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.geometry = BandGeometry.from_config(config["geometry"])
        self.generation = config["generation"]
        self.output = config["output"]
        self.cache_root = Path(config["cache_root"])
        self.source_root = Path(self.generation["vace_source"])
        self.checkpoint_dir = Path(self.generation["checkpoint_dir"])
        self.python = Path(self.generation["vace_python"])
        self.inference_script = self.source_root / "vace" / "vace_wan_inference.py"

    def offline_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "HF_HOME": str(self.cache_root / "huggingface"),
                "HF_HUB_CACHE": str(self.cache_root / "huggingface" / "hub"),
                "TORCH_HOME": str(self.cache_root / "torch"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "CUDA_MODULE_LOADING": "LAZY",
                "OMP_NUM_THREADS": "8",
            }
        )
        return env

    def readiness(self, require_cuda: bool = False) -> dict[str, Any]:
        required_checkpoint_files = [
            "config.json",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "diffusion_pytorch_model.safetensors.index.json",
        ]
        checkpoint_status = {
            name: (self.checkpoint_dir / name).is_file() for name in required_checkpoint_files
        }
        shards = sorted(self.checkpoint_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
        patch_status = False
        memory_patch_status = False
        if self.inference_script.exists():
            patch_status = "--noise_overlap_in" in self.inference_script.read_text(encoding="utf-8")
            model_source = self.source_root / "vace" / "models" / "wan" / "wan_vace.py"
            block_source = self.source_root / "vace" / "models" / "wan" / "modules" / "model.py"
            memory_patch_status = (
                model_source.is_file()
                and "def move_vae" in model_source.read_text(encoding="utf-8")
                and block_source.is_file()
                and "return tuple(all_c)" in block_source.read_text(encoding="utf-8")
            )
        status: dict[str, Any] = {
            "vace_python": self.python.is_file(),
            "vace_source": self.source_root.is_dir(),
            "vace_inference": self.inference_script.is_file(),
            "jrdb_proxy_patch": patch_status,
            "memory_offload_patch": memory_patch_status,
            "checkpoint_files": checkpoint_status,
            "checkpoint_shards": len(shards),
            "ffmpeg": bool(shutil.which("ffmpeg")),
        }
        if self.python.is_file():
            probe = subprocess.run(
                [
                    str(self.python),
                    "-c",
                    "import json,torch; print(json.dumps({'cuda':torch.cuda.is_available(),'torch':torch.__version__}))",
                ],
                capture_output=True,
                text=True,
                env=self.offline_env(),
            )
            try:
                status["runtime"] = json.loads(probe.stdout.strip())
            except Exception:
                status["runtime"] = {"error": (probe.stderr or probe.stdout).strip()}
        status["offline_ready"] = (
            status["vace_python"]
            and status["vace_inference"]
            and status["jrdb_proxy_patch"]
            and status["memory_offload_patch"]
            and all(checkpoint_status.values())
            and len(shards) == 7
            and status["ffmpeg"]
        )
        status["gpu_ready"] = status["offline_ready"] and bool(
            status.get("runtime", {}).get("cuda")
        )
        status["ready"] = status["gpu_ready"] if require_cuda else status["offline_ready"]
        return status

    def stage_window(
        self,
        sequence_name: str,
        frame_paths: list[Path],
        window: Window,
        work_dir: Path,
        previous_proxy_dir: Path | None = None,
    ) -> dict[str, Any]:
        stage_dir = work_dir / window.name / "inputs"
        primary_frames = stage_dir / "primary_frames"
        primary_masks = stage_dir / "primary_masks"
        rolled_frames = stage_dir / "rolled_frames"
        rolled_masks = stage_dir / "rolled_masks"
        primary_refs = stage_dir / "primary_refs"
        rolled_refs = stage_dir / "rolled_refs"
        for directory in (
            primary_frames,
            primary_masks,
            rolled_frames,
            rolled_masks,
            primary_refs,
            rolled_refs,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        mask = proxy_generation_mask(
            self.geometry.proxy_width,
            self.geometry.proxy_height,
            self.geometry.north_latitude,
            self.geometry.south_latitude,
        )
        roll_pixels = self.geometry.proxy_width // 2
        staged = []
        overlap_reference_candidates: list[tuple[Path, np.ndarray]] = []
        for local_index, global_index in enumerate(window.padded_indices):
            source_path = frame_paths[global_index]
            source = read_rgb(source_path)
            retained_overlap = (
                previous_proxy_dir is not None
                and local_index < window.overlap_with_previous
                and (previous_proxy_dir / f"{source_path.stem}.png").is_file()
            )
            if retained_overlap:
                proxy = read_rgb(previous_proxy_dir / f"{source_path.stem}.png")
                frame_mask = np.zeros_like(mask)
                overlap_reference_candidates.append((source_path, proxy))
            else:
                proxy = source_to_proxy(
                    source,
                    self.geometry.proxy_width,
                    self.geometry.proxy_height,
                    self.geometry.north_latitude,
                    self.geometry.south_latitude,
                )
                frame_mask = mask
            name = f"{local_index:05d}.png"
            atomic_save_png(primary_frames / name, proxy, compress_level=1)
            atomic_save_png(primary_masks / name, frame_mask, compress_level=1)
            atomic_save_png(
                rolled_frames / name,
                roll_longitude(proxy, roll_pixels),
                compress_level=1,
            )
            atomic_save_png(
                rolled_masks / name,
                roll_longitude(frame_mask, roll_pixels),
                compress_level=1,
            )
            staged.append(
                {
                    "local_index": local_index,
                    "global_index": global_index,
                    "frame": source_path.name,
                    "retained_overlap": retained_overlap,
                }
            )

        reference_paths: list[Path] = []
        rolled_reference_paths: list[Path] = []
        if overlap_reference_candidates:
            selected_refs = select_reference_indices(
                [item[1] for item in overlap_reference_candidates],
                count=int(self.generation.get("reference_frames", 3)),
            )
            for output_index, candidate_index in enumerate(selected_refs):
                _, reference = overlap_reference_candidates[candidate_index]
                name = f"reference-{output_index:02d}.png"
                primary_path = primary_refs / name
                rolled_path = rolled_refs / name
                atomic_save_png(primary_path, reference, compress_level=1)
                atomic_save_png(
                    rolled_path,
                    roll_longitude(reference, roll_pixels),
                    compress_level=1,
                )
                reference_paths.append(primary_path)
                rolled_reference_paths.append(rolled_path)

        fps = int(self.output.get("fps", 15))
        paths = {
            "primary_video": stage_dir / "src_video.mp4",
            "primary_mask": stage_dir / "src_mask.mp4",
            "rolled_video": stage_dir / "src_video_rolled.mp4",
            "rolled_mask": stage_dir / "src_mask_rolled.mp4",
            "primary_refs": reference_paths,
            "rolled_refs": rolled_reference_paths,
        }
        _encode_png_sequence(primary_frames, paths["primary_video"], fps)
        _encode_png_sequence(primary_masks, paths["primary_mask"], fps)
        _encode_png_sequence(rolled_frames, paths["rolled_video"], fps)
        _encode_png_sequence(rolled_masks, paths["rolled_mask"], fps)
        stage_manifest = {
            "schema_version": 2,
            "sequence": sequence_name,
            "window": window.name,
            "proxy_size": [self.geometry.proxy_width, self.geometry.proxy_height],
            "fps": fps,
            "frames": staged,
            "files": {
                key: {"path": str(path), "sha256": file_sha256(path)}
                for key, path in paths.items()
                if isinstance(path, Path)
            },
            "reference_images": [str(path) for path in reference_paths],
        }
        atomic_write_json(stage_dir / "stage_manifest.json", stage_manifest)
        return paths

    def _invoke_pass(
        self,
        source_video: Path,
        source_mask: Path,
        output_video: Path,
        noise_output: Path,
        seed: int,
        previous_noise: Path | None,
        reference_images: list[Path] | None = None,
    ) -> None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.python),
            str(self.inference_script),
            "--model_name",
            "vace-14B",
            "--size",
            "720p",
            "--width",
            str(self.geometry.proxy_width),
            "--height",
            str(self.geometry.proxy_height),
            "--frame_num",
            str(int(self.generation.get("window_frames", 81))),
            "--ckpt_dir",
            str(self.checkpoint_dir),
            "--src_video",
            str(source_video),
            "--src_mask",
            str(source_mask),
            "--prompt",
            str(self.generation["prompt"]),
            "--base_seed",
            str(seed),
            "--sample_solver",
            str(self.generation.get("sample_solver", "unipc")),
            "--sample_steps",
            str(int(self.generation.get("sample_steps", 50))),
            "--sample_shift",
            str(float(self.generation.get("sample_shift", 16.0))),
            "--sample_guide_scale",
            str(float(self.generation.get("guidance_scale", 5.0))),
            "--offload_model",
            str(bool(self.generation.get("offload_model", False))).lower(),
            "--save_file",
            str(output_video),
            "--noise_out",
            str(noise_output),
        ]
        if self.generation.get("t5_cpu", False):
            command.append("--t5_cpu")
        if reference_images:
            command.extend(
                ["--src_ref_images", ",".join(str(path) for path in reference_images)]
            )
        if previous_noise is not None:
            latent_overlap = latent_overlap_frames(
                int(self.generation.get("window_overlap", 17))
            )
            command.extend(
                [
                    "--noise_overlap_in",
                    str(previous_noise),
                    "--noise_overlap_latent_frames",
                    str(latent_overlap),
                ]
            )
        _run(command, cwd=self.source_root, env=self.offline_env())

    def generate_window(
        self,
        sequence_name: str,
        frame_paths: list[Path],
        window: Window,
        work_dir: Path,
        proxy_output_dir: Path,
    ) -> dict[str, Any]:
        previous_proxy_dir = proxy_output_dir if window.index > 0 else None
        staged = self.stage_window(
            sequence_name, frame_paths, window, work_dir, previous_proxy_dir
        )
        window_dir = work_dir / window.name
        primary_out = window_dir / "primary.mp4"
        primary_noise = window_dir / "primary_noise.pt"
        previous_window_dir = work_dir / f"window-{window.index - 1:05d}-placeholder"
        previous_primary_noise: Path | None = None
        previous_rolled_noise: Path | None = None
        if window.index > 0:
            candidates = sorted(work_dir.glob(f"window-{window.index - 1:05d}-*"))
            if candidates:
                previous_window_dir = candidates[0]
                if (previous_window_dir / "primary_noise.pt").is_file():
                    previous_primary_noise = previous_window_dir / "primary_noise.pt"
                if (previous_window_dir / "rolled_noise.pt").is_file():
                    previous_rolled_noise = previous_window_dir / "rolled_noise.pt"

        base_seed = sequence_seed(int(self.generation.get("seed", 3407)), sequence_name)
        window_seed = base_seed + window.index * 1009
        self._invoke_pass(
            staged["primary_video"],
            staged["primary_mask"],
            primary_out,
            primary_noise,
            window_seed,
            previous_primary_noise,
            staged["primary_refs"],
        )
        primary_frames = _decode_video(primary_out, window_dir / "primary_decoded")

        rolled_frames: list[Path] | None = None
        if bool(self.generation.get("seam_repair", True)):
            rolled_out = window_dir / "rolled.mp4"
            rolled_noise = window_dir / "rolled_noise.pt"
            self._invoke_pass(
                staged["rolled_video"],
                staged["rolled_mask"],
                rolled_out,
                rolled_noise,
                window_seed + 7919,
                previous_rolled_noise,
                staged["rolled_refs"],
            )
            rolled_frames = _decode_video(rolled_out, window_dir / "rolled_decoded")

        required = int(self.generation.get("window_frames", 81))
        if len(primary_frames) != required or (
            rolled_frames is not None and len(rolled_frames) != required
        ):
            raise RuntimeError(
                f"VACE returned wrong frame count for {window.name}: "
                f"primary={len(primary_frames)}, rolled={len(rolled_frames or [])}, expected={required}"
            )
        proxy_output_dir.mkdir(parents=True, exist_ok=True)
        mask = proxy_generation_mask(
            self.geometry.proxy_width,
            self.geometry.proxy_height,
            self.geometry.north_latitude,
            self.geometry.south_latitude,
        )
        mask3 = np.repeat(mask[:, :, None], 3, axis=2)
        roll_pixels = self.geometry.proxy_width // 2
        committed: list[str] = []
        for local_index in range(window.commit_from, window.valid_frames):
            global_index = window.start + local_index
            source_path = frame_paths[global_index]
            primary = read_rgb(primary_frames[local_index])
            if rolled_frames is not None:
                rolled_back = roll_longitude(read_rgb(rolled_frames[local_index]), -roll_pixels)
                generated = seam_repair_blend(primary, rolled_back, mask3)
            else:
                generated = primary
            source_proxy = source_to_proxy(
                read_rgb(source_path),
                self.geometry.proxy_width,
                self.geometry.proxy_height,
                self.geometry.north_latitude,
                self.geometry.south_latitude,
            )
            generated[mask == 0] = source_proxy[mask == 0]
            target = proxy_output_dir / f"{source_path.stem}.png"
            atomic_save_png(
                target,
                generated,
                compress_level=int(self.output.get("png_compress_level", 4)),
            )
            committed.append(target.name)
        return {
            "window": window.name,
            "seed": window_seed,
            "committed": committed,
            "primary_noise_sha256": file_sha256(primary_noise),
            "rolled_noise_sha256": (
                file_sha256(window_dir / "rolled_noise.pt")
                if (window_dir / "rolled_noise.pt").is_file()
                else None
            ),
        }

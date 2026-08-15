from __future__ import annotations

import hashlib
import base64
import io
import json
import os
import subprocess
import sys
import time
import tomllib
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .geometry import cosine_weights, embed_source, generation_mask, periodic_take, tile_starts


class Backend:
    def generate(
        self, source: Image.Image, frame_seed: int, reference: np.ndarray | None = None
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class EdgeBackend(Backend):
    output_height: int
    top_rows: int

    def generate(
        self, source: Image.Image, frame_seed: int, reference: np.ndarray | None = None
    ) -> np.ndarray:
        del frame_seed, reference
        canvas = embed_source(source, self.output_height, self.top_rows)
        src = np.asarray(source.convert("RGB"), dtype=np.uint8)
        for row in range(self.top_rows):
            canvas[self.top_rows - 1 - row] = src[min(row, src.shape[0] - 1)]
            canvas[self.top_rows + src.shape[0] + row] = src[max(0, src.shape[0] - 1 - row)]
        return canvas


class SDXLIPBackend(Backend):
    def __init__(self, config: dict) -> None:
        import torch
        from diffusers import AutoPipelineForInpainting, DPMSolverMultistepScheduler

        if not torch.cuda.is_available():
            raise RuntimeError("sdxl_ip requires CUDA; use --backend edge only for CPU tests")
        generation = config["generation"]
        cache_dir = config["cache_root"] + "/huggingface/hub"
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            generation["model"],
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            local_files_only=True,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        self.pipe.load_ip_adapter(
            generation["ip_adapter"],
            subfolder=generation["ip_adapter_subfolder"],
            weight_name=generation["ip_adapter_weight"],
            image_encoder_folder="models/image_encoder",
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self.pipe.set_ip_adapter_scale(float(generation["ip_adapter_scale"]))
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=False)
        self.config = config

    def generate(
        self, source: Image.Image, frame_seed: int, reference: np.ndarray | None = None
    ) -> np.ndarray:
        import torch

        del reference

        geometry = self.config["geometry"]
        generation = self.config["generation"]
        canvas = embed_source(source, geometry["output_height"], geometry["top_rows"])
        mask = np.asarray(
            generation_mask(
                source.width,
                geometry["input_height"],
                geometry["output_height"],
                geometry["top_rows"],
            )
        )
        tile_width = int(generation["tile_width"])
        overlap = int(generation["tile_overlap"])
        starts = tile_starts(source.width, tile_width, overlap)
        accumulator = np.zeros_like(canvas, dtype=np.float32)
        weight_sum = np.zeros((canvas.shape[0], canvas.shape[1], 1), dtype=np.float32)
        tile_weight = cosine_weights(tile_width, overlap)[None, :, None]

        for tile_index, start in enumerate(starts):
            image_tile = periodic_take(canvas, start, tile_width)
            mask_tile = periodic_take(mask[:, :, None], start, tile_width)[:, :, 0]
            condition = periodic_take(np.asarray(source.convert("RGB")), start, tile_width)
            generator = torch.Generator(device="cuda").manual_seed(frame_seed + tile_index)
            result = self.pipe(
                prompt=generation["prompt"],
                negative_prompt=generation["negative_prompt"],
                image=Image.fromarray(image_tile),
                mask_image=Image.fromarray(mask_tile),
                ip_adapter_image=Image.fromarray(condition),
                width=tile_width,
                height=geometry["output_height"],
                num_inference_steps=int(generation["steps"]),
                guidance_scale=float(generation["guidance_scale"]),
                strength=float(generation["strength"]),
                generator=generator,
            ).images[0]
            generated = np.asarray(result.convert("RGB"), dtype=np.float32)
            indices = np.arange(start, start + tile_width) % source.width
            accumulator[:, indices] += generated * tile_weight
            weight_sum[:, indices] += tile_weight
        return np.clip(accumulator / np.maximum(weight_sum, 1e-6), 0, 255).astype(np.uint8)


class OpenAIChatImage2Backend(Backend):
    """OpenAI-compatible image edit backend.

    The central observed band is sent as opaque pixels and the synthetic bands
    as transparent pixels in the mask. This keeps the source contract enforced
    locally even if a provider returns a slightly different crop.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        generation = config["generation"]
        self.model = generation.get("model", "OpenAI-chat-Image2")
        provider_url, provider_key = self._codex_provider(generation) if generation.get("use_codex_provider") else (None, None)
        self.base_url = os.environ.get("OPENAI_BASE_URL", provider_url or generation.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY", provider_key or "")
        if not self.api_key:
            raise RuntimeError("openai_chat_image2 requires OPENAI_API_KEY")
        self.timeout = int(generation.get("api_timeout", 180))
        self.retries = int(generation.get("api_retries", 4))
        self.cache_dir = config["cache_root"] + "/openai_image2"
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _codex_provider(generation: dict) -> tuple[str | None, str | None]:
        path = os.path.expanduser(generation.get("codex_config", "/root/.codex/config.toml"))
        with open(path, "rb") as handle:
            config = tomllib.load(handle)
        provider_name = generation.get("codex_provider") or config.get("model_provider")
        provider = config.get("model_providers", {}).get(provider_name, {})
        auth = provider.get("auth", {})
        command = auth.get("command")
        key = None
        if command:
            result = subprocess.run([command, *auth.get("args", [])], check=True, capture_output=True, text=True)
            key = result.stdout.strip()
        return provider.get("base_url"), key

    @staticmethod
    def _png_bytes(image: Image.Image) -> bytes:
        handle = io.BytesIO()
        image.save(handle, format="PNG", optimize=False)
        return handle.getvalue()

    def _endpoint(self, path: str) -> str:
        prefix = "" if self.base_url.endswith("/v1") else "/v1"
        return self.base_url + prefix + "/" + path.lstrip("/")

    def _edit(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
        reference: Image.Image | None = None,
    ) -> Image.Image:
        # requests is intentionally avoided so this backend works in the lean
        # offline environment used for geometry tests.
        generation = self.config["generation"]
        api_mode = generation.get("api_mode", "images_edits")
        if api_mode == "chat_completions":
            return self._chat_edit(image, mask, prompt, seed, reference)
        if api_mode == "json_images_edits":
            return self._json_edit(image, mask, prompt, seed, reference)
        boundary = "----jrdb-sphere-" + hashlib.sha256(f"{seed}:{prompt}".encode()).hexdigest()[:24]
        fields = {
            "model": self.model,
            "prompt": prompt,
            "size": f"{image.width}x{image.height}",
            "response_format": "b64_json",
            "n": "1",
        }
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        for name, filename, payload in (("image", "image.png", self._png_bytes(image)), ("mask", "mask.png", self._png_bytes(mask))):
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n".encode() + payload + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            self.base_url + "/images/edits", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
        )
        last_error = "unknown error"
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                item = payload.get("data", [{}])[0]
                if item.get("b64_json"):
                    return Image.open(io.BytesIO(base64.b64decode(item["b64_json"]))).convert("RGB")
                if item.get("url"):
                    with urllib.request.urlopen(item["url"], timeout=self.timeout) as response:
                        return Image.open(io.BytesIO(response.read())).convert("RGB")
                last_error = f"provider returned no image: {payload}"
            except Exception as exc:  # retry transient provider/network errors
                last_error = str(exc)
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(f"OpenAI-chat-Image2 request failed after {self.retries} attempts: {last_error}")

    def _json_edit(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
        reference: Image.Image | None = None,
    ) -> Image.Image:
        image_data = "data:image/png;base64," + base64.b64encode(self._png_bytes(image)).decode()
        mask_data = "data:image/png;base64," + base64.b64encode(self._png_bytes(mask)).decode()
        generation = self.config["generation"]
        images = [{"image_url": image_data}, {"image_url": mask_data}]
        if reference is not None:
            images.append(
                {
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(self._png_bytes(reference)).decode()
                }
            )
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "size": f"{image.width}x{image.height}",
            "response_format": "b64_json",
            "n": 1,
            "seed": seed,
        }).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint("images/edits"), data=payload, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"},
        )
        last_error = "unknown error"
        for attempt in range(self.retries):
            try:
                print(
                    f"API gpt-image edit attempt {attempt + 1}/{self.retries} "
                    f"size={image.width}x{image.height}",
                    file=sys.stderr,
                    flush=True,
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                result = self._find_image(response_payload)
                if result is not None:
                    return result
                last_error = f"provider returned no decodable image: {response_payload}"
            except urllib.error.HTTPError as exc:
                raw = exc.read(8000).decode("utf-8", "replace")
                try:
                    payload_error = json.loads(raw)
                    message = payload_error.get("error", {}).get("message", raw)
                except Exception:
                    message = raw
                last_error = f"HTTP {exc.code}: {message}"
                print(last_error, file=sys.stderr, flush=True)
                if exc.code < 500 and exc.code not in {408, 409, 429}:
                    break
            except Exception as exc:
                last_error = str(exc)
                print(f"API request error: {last_error}", file=sys.stderr, flush=True)
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(f"gpt-image JSON edit request failed after {self.retries} attempts: {last_error}")

    @staticmethod
    def _find_image(value: object) -> Image.Image | None:
        """Accept common OpenAI-compatible image response shapes."""
        if isinstance(value, dict):
            for key in ("b64_json", "base64", "image_base64"):
                if isinstance(value.get(key), str):
                    try:
                        return Image.open(io.BytesIO(base64.b64decode(value[key]))).convert("RGB")
                    except Exception:
                        pass
            for key in ("url", "image_url"):
                candidate = value.get(key)
                if isinstance(candidate, dict):
                    candidate = candidate.get("url")
                if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
                    try:
                        with urllib.request.urlopen(candidate, timeout=180) as response:
                            return Image.open(io.BytesIO(response.read())).convert("RGB")
                    except Exception:
                        pass
            for child in value.values():
                found = OpenAIChatImage2Backend._find_image(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = OpenAIChatImage2Backend._find_image(child)
                if found is not None:
                    return found
        elif isinstance(value, str) and value.startswith("data:image/"):
            try:
                return Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[1]))).convert("RGB")
            except Exception:
                pass
        return None

    def _chat_edit(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        seed: int,
        reference: Image.Image | None = None,
    ) -> Image.Image:
        generation = self.config["generation"]
        image_data = base64.b64encode(self._png_bytes(image)).decode()
        mask_data = base64.b64encode(self._png_bytes(mask)).decode()
        content = [
            {"type": "text", "text": prompt + " Return only the completed PNG image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_data}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + mask_data}},
        ]
        if reference is not None:
            reference_data = base64.b64encode(self._png_bytes(reference)).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + reference_data}}
            )
        payload = json.dumps({
            "model": self.model,
            "temperature": 0,
            "seed": seed,
            "messages": [{"role": "user", "content": content}],
        }).encode("utf-8")
        request = urllib.request.Request(self.base_url + "/chat/completions", data=payload, method="POST",
                                         headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"})
        last_error = "unknown error"
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = self._find_image(json.loads(response.read().decode("utf-8")))
                if result is not None:
                    return result
                last_error = "chat response did not contain a decodable image"
            except Exception as exc:
                last_error = str(exc)
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(f"OpenAI-chat-Image2 chat request failed after {self.retries} attempts: {last_error}")

    def generate(
        self, source: Image.Image, frame_seed: int, reference: np.ndarray | None = None
    ) -> np.ndarray:
        geometry = self.config["geometry"]
        generation = self.config["generation"]
        canvas = embed_source(source, geometry["output_height"], geometry["top_rows"])
        mask = np.asarray(generation_mask(source.width, geometry["input_height"], geometry["output_height"], geometry["top_rows"]))
        tile_width = int(generation.get("tile_width", 1024))
        overlap = int(generation.get("tile_overlap", 256))
        starts = tile_starts(source.width, tile_width, overlap)
        accumulator = np.zeros_like(canvas, dtype=np.float32)
        weight_sum = np.zeros((canvas.shape[0], canvas.shape[1], 1), dtype=np.float32)
        tile_weight = cosine_weights(tile_width, overlap)[None, :, None]
        request_size = generation.get("api_size")
        per_tile_seed = bool(generation.get("per_tile_seed", False))
        for tile_index, start in enumerate(starts):
            print(
                f"OpenAI image tile {tile_index + 1}/{len(starts)} start={start}",
                file=sys.stderr,
                flush=True,
            )
            image_tile = periodic_take(canvas, start, tile_width)
            mask_tile = periodic_take(mask[:, :, None], start, tile_width)[:, :, 0]
            # OpenAI image edits convention: transparent = editable, opaque = keep.
            alpha = (255 - mask_tile).astype(np.uint8)
            mask_rgba = np.zeros((mask_tile.shape[0], mask_tile.shape[1], 4), dtype=np.uint8)
            mask_rgba[:, :, 3] = alpha
            tile_seed = frame_seed + tile_index if per_tile_seed else frame_seed
            reference_tile = periodic_take(reference, start, tile_width) if reference is not None else None
            key_parts = [
                image_tile.tobytes(),
                mask_rgba.tobytes(),
                reference_tile.tobytes() if reference_tile is not None else b"",
                json.dumps(
                    {
                        "model": self.model,
                        "prompt": generation["prompt"],
                        "api_mode": generation.get("api_mode"),
                        "api_size": request_size,
                        "seed": tile_seed,
                    },
                    sort_keys=True,
                ).encode(),
            ]
            tile_key = hashlib.sha256(b"".join(key_parts)).hexdigest()
            cache_path = os.path.join(self.cache_dir, tile_key + ".png")
            if os.path.isfile(cache_path):
                result = Image.open(cache_path).convert("RGB")
            else:
                request_image = Image.fromarray(image_tile)
                request_mask = Image.fromarray(mask_rgba, "RGBA")
                request_reference = Image.fromarray(reference_tile) if reference_tile is not None else None
                if request_size:
                    req_width, req_height = (int(part) for part in str(request_size).lower().split("x", 1))
                    if req_width < tile_width or req_height < geometry["output_height"]:
                        raise ValueError("api_size must be at least tile_width x output_height")
                    if (req_width, req_height) != request_image.size:
                        padded = Image.new("RGB", (req_width, req_height))
                        padded.paste(request_image, (0, 0))
                        padded.paste(request_image.crop((0, request_image.height - 1, request_image.width, request_image.height)).resize((req_width, req_height - request_image.height)), (0, request_image.height))
                        padded_mask = Image.new("RGBA", (req_width, req_height), (0, 0, 0, 255))
                        padded_mask.paste(request_mask, (0, 0))
                        request_image, request_mask = padded, padded_mask
                        if request_reference is not None:
                            padded_reference = Image.new("RGB", (req_width, req_height))
                            padded_reference.paste(request_reference, (0, 0))
                            padded_reference.paste(
                                request_reference.crop(
                                    (0, request_reference.height - 1, request_reference.width, request_reference.height)
                                ).resize((req_width, req_height - request_reference.height)),
                                (0, request_reference.height),
                            )
                            request_reference = padded_reference
                result = self._edit(
                    request_image,
                    request_mask,
                    generation["prompt"],
                    tile_seed,
                    request_reference,
                )
                result = result.crop((0, 0, tile_width, geometry["output_height"]))
                result.save(cache_path, format="PNG")
            generated = np.asarray(result.resize((tile_width, geometry["output_height"]), Image.Resampling.LANCZOS), dtype=np.float32)
            indices = np.arange(start, start + tile_width) % source.width
            accumulator[:, indices] += generated * tile_weight
            weight_sum[:, indices] += tile_weight
        return np.clip(accumulator / np.maximum(weight_sum, 1e-6), 0, 255).astype(np.uint8)


def build_backend(name: str, config: dict) -> Backend:
    geometry = config["geometry"]
    if name == "edge":
        return EdgeBackend(geometry["output_height"], geometry["top_rows"])
    if name == "sdxl_ip":
        return SDXLIPBackend(config)
    if name == "openai_chat_image2":
        return OpenAIChatImage2Backend(config)
    raise ValueError(f"unknown backend: {name}")


def config_fingerprint(config: dict, backend: str) -> str:
    relevant = {"geometry": config["geometry"], "generation": config["generation"], "backend": backend}
    return hashlib.sha256(repr(relevant).encode("utf-8")).hexdigest()[:16]

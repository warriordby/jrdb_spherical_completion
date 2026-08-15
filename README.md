# JRDB Spherical Completion

English | [简体中文](README.zh-CN.md)

This project implements the frozen production design for completing a JRDB stitched `3760×480` observation band into a `3760×720` panorama:

```text
JRDB 3760×480
  → spherical resampling to a standard 2:1 ERP proxy at 1440×720
  → Wan2.1 VACE-14B masked video-to-video generation over 81 frames
  → 17-frame overlap with condition locking and latent-noise reuse
  → a second pass rolled by W/2, used only to repair synthetic-region seams
  → inverse spherical mapping to 3760×720
  → direct restoration of the observed center band y=[120,600)
```

See [`docs/JRDB_MOT_SPHERICAL_COMPLETION_PLAN_2026-08.md`](docs/JRDB_MOT_SPHERICAL_COMPLETION_PLAN_2026-08.md) for the frozen technical plan. The older SDXL and single-frame API configurations are retained only for historical comparison and are not production entry points.

## 81-frame before/after preview

![JRDB 81-frame before/after comparison](assets/jrdb_81_frame_before_after.gif)

[▶ Play or download the MP4 version](assets/jrdb_81_frame_before_after.mp4)

- Sequence: `cubberly-auditorium-2019-04-22_1`
- Length: 81 frames at 15 FPS (5.4 seconds)
- Top: the original `3760×480` JRDB observation band centered on an equal-size canvas
- Bottom: the completed `3760×720` spherical result
- The observed center band is restored directly from JPEG-decoded source pixels; generated content exists only in the top and bottom synthetic regions

## Implemented invariants

- The observed center band is copied directly from the JPEG-decoded input. PNG round-trip verification requires `max_abs_error=0`.
- Proxy canvases use pixel-center latitude/longitude mapping; higher-resolution proxies such as `2048×1024` do not reuse a fixed 120-row offset.
- Horizontal sampling is periodic. VACE runs both the original longitude and a `W/2`-rolled pass, and blends only the top and bottom synthetic regions.
- Every sequence has deterministic 81/17 window manifests, a configuration fingerprint, per-window state, and per-frame completion hashes.
- Adjacent windows reuse the final five VAE latent-noise frames from the previous window and lock the 17 overlapping RGB frames as non-editable VACE conditions. No frame-wise RGB cross-fade is used.
- A synthetic mask is written for every frame. Synthetic regions are never copied into ground truth and must be treated as ignore regions.
- The output layout contains `images/image_stitched`, `mot/<seq>/img1`, `synthetic_masks`, `metadata`, `quality`, and a root `manifest.json`.
- An unrestricted full run checks for at least 150 GiB of free space before starting.

## Data and default configuration

- Input: `/root/autodl-tmp/JRDB2019-MOT/images/image_stitched`
- Dataset: 27 sequences and 27,661 frames
- Primary configuration: `configs/jrdb_vace14b_pro6000.json`
- Resource cache: `/root/autodl-tmp/model_cache_v2`
- Output: `/root/autodl-tmp/datasets/JRDB-Spherical-VFOV180-VACE`
- VACE environment: `.venv-vace` using Python 3.10 and PyTorch CUDA 12.8. The 84 GB single-GPU configuration uses same-precision CPU offload to avoid overlapping T5, DiT, and VAE memory peaks.
- CPU/download environment: `.venv`
- Fully resolved environment snapshot: `requirements/vace-resolved-cu128.lock.txt`

## Preparation without a GPU

Run the complete preparation workflow:

```bash
cd /root/autodl-tmp/jrdb_spherical_completion
bash scripts/prepare_all.sh
```

The script:

1. Fetches pinned commits of VACE, Wan2.1, Seen-to-Scene, RAFT, TrackEval, and deep-person-reid. ProPainter uses the pinned flow-completion release weights without downloading unrelated demo assets.
2. Applies the exact `1440×720` sizing, cross-window noise reuse, and 84 GB same-precision memory patches to the pinned VACE/Wan sources.
3. Builds isolated `.venv-vace` and CPU utility environments, pins CUDA 12.8 PyTorch and the FlashAttention wheel, and retains PyTorch SDPA as a fallback.
4. Downloads the pinned `Wan-AI/Wan2.1-VACE-14B` revision plus RAFT, ProPainter flow-completion, Faster R-CNN, and OSNet weights.
5. Computes SHA-256 for every resource and sets `ready=true` in `resources_manifest_v2.json` only when every file is complete and no `.incomplete` marker remains.

The downloader does not initialize CUDA and supports repeatable, resumable downloads:

```bash
.venv/bin/python scripts/download_resources.py \
  --config configs/jrdb_vace14b_pro6000.json \
  --hf-endpoint https://hf-mirror.com
```

Generate only the window manifest without encoding videos:

```bash
.venv/bin/jrdb-sphere --config configs/jrdb_vace14b_pro6000.json stage \
  --sequences cubberly-auditorium-2019-04-22_1 \
  --limit-frames 81 --manifest-only
```

Remove `--manifest-only` to also encode the primary and `W/2`-rolled VACE source/mask videos.

## CPU validation

```bash
.venv/bin/pytest -q
.venv/bin/jrdb-sphere --config configs/jrdb_vace14b_pro6000.json inspect
```

Run a three-frame geometry diagnostic without loading the generation model:

```bash
.venv/bin/jrdb-sphere --config configs/jrdb_vace14b_pro6000.json run \
  --backend edge --sequences cubberly-auditorium-2019-04-22_1 --limit-frames 3
.venv/bin/jrdb-sphere --config configs/jrdb_vace14b_pro6000.json verify \
  --sequences cubberly-auditorium-2019-04-22_1 --limit-frames 3
```

## GPU execution

Do not start with the complete dataset. First run three representative 81-frame sequences:

```bash
cd /root/autodl-tmp/jrdb_spherical_completion
bash scripts/run_gpu_smoke.sh
```

The script processes:

- `cubberly-auditorium-2019-04-22_1`
- `discovery-walk-2019-02-28_0`
- `indoor-coupa-cafe-2019-02-06_0`

Each sequence runs VACE generation, geometry verification, and detector/ReID quality checks. Expand to 243 frames per category only after all three sequences pass and their output has been manually reviewed.

Run one 81-frame window directly:

```bash
bash scripts/run_offline_vace.sh --config configs/jrdb_vace14b_pro6000.json run \
  --backend vace14b \
  --sequences cubberly-auditorium-2019-04-22_1 \
  --limit-frames 81
```

The 84 GB single-GPU configuration must retain `offload_model=true`. At stage boundaries, the runtime unloads T5, VAE, input conditions, and generated VACE hints, and chunks temporary FP64 RoPE tensors. Model weights, resolution, 81-frame length, 50 sampling steps, guidance, and BF16 autocast remain unchanged, and no quantization is used. `run_offline_vace.sh` configures runtime settings such as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

On an RTX 6000D, the same window completed three consecutive sampling steps at approximately 83,230 MiB allocated with about 1,804 MiB free. Each step took approximately 157 seconds. Primary and rolled passes each use 50 steps, for an estimated total of about 4.4 hours per window. Interrupted windows remain in the `running` state; rerunning the command regenerates them with the same seed and noise rather than registering partial output as complete.

## Quality gates

Structural verification:

```bash
.venv/bin/jrdb-sphere --config configs/jrdb_vace14b_pro6000.json verify \
  --sequences cubberly-auditorium-2019-04-22_1 --limit-frames 81
```

Detector/ReID verification:

```bash
bash scripts/run_offline_vace.sh --config configs/jrdb_vace14b_pro6000.json quality \
  --sequences cubberly-auditorium-2019-04-22_1 --limit-frames 81
```

The current workspace contains JRDB test images without MOT ground truth. Center-pixel, seam, mask, synthetic-region person detection, and paired ReID checks can be completed, but HOTA/IDF1/MOTA must remain pending until matching ground truth is supplied to the pinned TrackEval revision. Top and bottom generated content must never be represented as real ground truth.

## Known upstream limitation

The pinned Seen-to-Scene repository describes RAFT, ProPainter flow completion, and a self-trained UNet/latent-refinement model, but does not publish its `checkpoint-100000`. This project therefore prepares its public RAFT and flow-completion components and ports reference conditioning, overlap locking, and latent-noise propagation into the VACE wrapper. The resource manifest records the missing checkpoint explicitly rather than fabricating an official asset.

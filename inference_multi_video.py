"""Per-camera reconstruction-video inference for NeoVerse.

Unlike ``inference_multi_new.py`` (which evaluates novel-view synthesis by
re-rendering a held-out middle frame), this script renders **the same frames it
was given** from each camera's own predicted pose at its own timestamp. Per
scene we feed the reconstructor 100 non-overlapping chunks of 4 contiguous
timestamps × 4 cameras (16 context views per chunk, time-major interleaved):

    [(t_{4k}, c0), (t_{4k}, c1), (t_{4k}, c2), (t_{4k}, c3),
     (t_{4k+1}, c0), ..., (t_{4k+3}, c3)]                    -> 16 context views

For each chunk, the rasterizer renders all 16 slots from their predicted
poses/intrinsics at their input timestamps. Frames are saved as PNGs (under
each camera's ``frames/`` subdir, using the GT filename for downstream metric
diffs) and assembled into one MP4 per camera per scene at the chosen FPS.

Output layout::

    <output_path>/<take_name>/<cam>.mp4
    <output_path>/<take_name>/<cam>/frames/<original_frame_filename>.png

Example::

    python inference_multi_video.py \\
        --scenes_txt /path/to/scenes.txt \\
        --scenes_root /path/to/scenes_root \\
        --output_path outputs/multi_video/exp_a \\
        --reconstructor_path models/NeoVerse/reconstructor.ckpt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from diffsynth.data.video import save_video
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth.utils.auxiliary import center_crop, homo_matrix_inverse
from diffsynth.utils.multiview import load_frames_from_dir


NUM_CAMERAS = 4
FRAMES_PER_BATCH = 4
BATCHES_PER_SCENE = 100
_IMG_EXTS = {".png", ".jpg", ".jpeg"}


def discover_camera_dirs(scene_path: Path) -> List[str]:
    cams = []
    for entry in sorted(scene_path.iterdir()):
        if not entry.is_dir():
            continue
        if any(p.suffix.lower() in _IMG_EXTS for p in entry.iterdir()):
            cams.append(entry.name)
    if not cams:
        raise FileNotFoundError(
            f"No camera subdirectories with image files found under {scene_path}"
        )
    return cams


def _load_and_resize(path: str, width: int, height: int, resize_mode: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if resize_mode == "resize":
        return img.resize((width, height), resample=Image.LANCZOS)
    return center_crop(img, (width, height))


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return F.to_tensor(img)


def _frame_chw_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    return (frame_chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()


def _save_rendered(rendered_chw: torch.Tensor, path: Path) -> None:
    arr = _frame_chw_to_uint8_hwc(rendered_chw)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _resolve_cameras_for_scene(
    scene_path: Path, requested: Optional[List[str]]
) -> Optional[List[str]]:
    available = discover_camera_dirs(scene_path)

    if requested is None:
        if len(available) < NUM_CAMERAS:
            print(f"  [SKIP] Only {len(available)} non-empty cameras, need {NUM_CAMERAS}")
            return None
        return available[:NUM_CAMERAS]

    available_set = set(available)
    chosen: List[str] = []
    substitutions: List[Tuple[str, str]] = []
    fallback_pool = [c for c in available if c not in requested]
    fallback_iter = iter(fallback_pool)

    for cam in requested:
        if cam in available_set:
            chosen.append(cam)
            continue
        replacement = next(
            (c for c in fallback_iter if c not in chosen),
            None,
        )
        if replacement is None:
            print(
                f"  [SKIP] '{cam}' is missing/empty and no available camera "
                f"left to substitute (available={available})"
            )
            return None
        chosen.append(replacement)
        substitutions.append((cam, replacement))

    for orig, sub in substitutions:
        print(f"  [INFO] '{orig}' missing/empty -> substituting '{sub}'")
    return chosen


def _intersect_frames(
    cameras: List[str],
    frames_by_cam_paths: Dict[str, List[str]],
) -> List[Dict[str, str]]:
    name_to_path: Dict[str, Dict[str, str]] = {}
    cam_basenames: Dict[str, set] = {}
    for cam in cameras:
        cam_basenames[cam] = set()
        for full in frames_by_cam_paths[cam]:
            name = Path(full).name
            cam_basenames[cam].add(name)
            name_to_path.setdefault(name, {})[cam] = full

    common = sorted(set.intersection(*(cam_basenames[c] for c in cameras)))

    for cam in cameras:
        extra = sorted(cam_basenames[cam] - set(common))
        if extra:
            preview = ", ".join(extra[:10])
            more = f" ... (+{len(extra) - 10} more)" if len(extra) > 10 else ""
            print(
                f"  [WARN] Camera '{cam}' has {len(extra)} frame(s) missing from "
                f"other cameras; dropping: [{preview}{more}]"
            )

    return [{cam: name_to_path[name][cam] for cam in cameras} for name in common]


def _png_path(output_dir: Path, take_name: str, cam: str, frame_path: str) -> Path:
    return output_dir / take_name / cam / "frames" / Path(frame_path).name


def _mp4_path(output_dir: Path, take_name: str, cam: str) -> Path:
    return output_dir / take_name / f"{cam}.mp4"


@torch.no_grad()
def render_scene(
    pipe,
    scene_path: Path,
    output_dir: Path,
    height: int,
    width: int,
    resize_mode: str,
    motion_frame_stride: int,
    fps: int,
    max_batches: Optional[int],
    cameras_arg: Optional[List[str]],
) -> bool:
    take_name = scene_path.name
    print(f"\n=== Scene {take_name} ===")

    if not scene_path.is_dir():
        print(f"  [SKIP] Scene directory not found: {scene_path}")
        return False

    cameras = _resolve_cameras_for_scene(scene_path, cameras_arg)
    if cameras is None:
        return False
    if len(cameras) != NUM_CAMERAS:
        print(f"  [SKIP] Expected {NUM_CAMERAS} cameras, got {len(cameras)}")
        return False
    print(f"  Cameras: {cameras}")

    frames_by_cam_paths: Dict[str, List[str]] = {}
    for cam in cameras:
        frames_by_cam_paths[cam] = load_frames_from_dir(str(scene_path / cam))

    intersection = _intersect_frames(cameras, frames_by_cam_paths)
    num_frames = len(intersection)
    if num_frames < FRAMES_PER_BATCH:
        print(
            f"  [SKIP] Only {num_frames} frame(s) common to all cameras; need "
            f"≥ {FRAMES_PER_BATCH}"
        )
        return False
    print(f"  {num_frames} frames common to all cameras")

    frames_by_cam: Dict[str, List[str]] = {
        cam: [row[cam] for row in intersection] for cam in cameras
    }

    target_batches = BATCHES_PER_SCENE if max_batches is None else min(BATCHES_PER_SCENE, max_batches)
    available_batches = num_frames // FRAMES_PER_BATCH
    n_batches = min(target_batches, available_batches)
    if n_batches == 0:
        print("  [SKIP] Not enough frames for a single batch")
        return False
    print(f"  Running {n_batches} batches × {FRAMES_PER_BATCH} frames = "
          f"{n_batches * FRAMES_PER_BATCH} frames per camera")

    chunks: List[List[int]] = [
        list(range(k * FRAMES_PER_BATCH, (k + 1) * FRAMES_PER_BATCH))
        for k in range(n_batches)
    ]

    for cam in cameras:
        (output_dir / take_name / cam / "frames").mkdir(parents=True, exist_ok=True)

    mp4_paths = {cam: _mp4_path(output_dir, take_name, cam) for cam in cameras}
    all_mp4s_exist = all(p.exists() for p in mp4_paths.values())
    all_pngs_exist = all(
        _png_path(output_dir, take_name, cam, frames_by_cam[cam][t]).exists()
        for cam in cameras
        for chunk in chunks
        for t in chunk
    )
    if all_mp4s_exist and all_pngs_exist:
        print("  [SKIP] All MP4s and PNGs already exist for this scene")
        return True

    def _chunk_done(chunk: List[int]) -> bool:
        return all(
            _png_path(output_dir, take_name, cam, frames_by_cam[cam][t]).exists()
            for cam in cameras
            for t in chunk
        )

    device = pipe.device
    per_cam_frames: Dict[str, List[np.ndarray]] = {cam: [] for cam in cameras}
    rendered_count = 0
    skipped_chunks = 0

    for b_idx, chunk in enumerate(chunks):
        if _chunk_done(chunk):
            for cam in cameras:
                for t in chunk:
                    img = Image.open(
                        _png_path(output_dir, take_name, cam, frames_by_cam[cam][t])
                    ).convert("RGB")
                    per_cam_frames[cam].append(np.array(img))
            skipped_chunks += 1
            continue

        ctx_pairs: List[Tuple[int, str]] = []
        for t in chunk:
            for cam in cameras:
                ctx_pairs.append((t, cam))

        imgs_list = [
            _to_tensor(_load_and_resize(frames_by_cam[cam][t], width, height, resize_mode))
            for (t, cam) in ctx_pairs
        ]
        ctx_imgs = torch.stack(imgs_list, dim=0).unsqueeze(0).to(device)

        timestamps = torch.tensor(
            [2 * t for (t, _) in ctx_pairs], dtype=torch.int64, device=device
        )

        views = {
            "img": ctx_imgs,
            "is_target": torch.zeros((1, len(ctx_pairs)), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, len(ctx_pairs)), dtype=torch.bool, device=device),
            "timestamp": timestamps.unsqueeze(0),
        }

        if pipe.vram_management_enabled:
            pipe.reconstructor.to(device)

        t0 = time.time()
        with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
            predictions = pipe.reconstructor(
                views,
                is_inference=True,
                use_motion=True,
                motion_frame_stride=motion_frame_stride,
            )
        encode_time = time.time() - t0

        if pipe.vram_management_enabled:
            pipe.reconstructor.cpu()
            torch.cuda.empty_cache()

        pred_c2w = predictions["rendered_extrinsics"][0]  # [16, 4, 4]
        pred_K = predictions["rendered_intrinsics"][0]    # [16, 3, 3]
        splats = predictions["splats"][0]
        render_w2c = homo_matrix_inverse(pred_c2w)

        t0 = time.time()
        rendered_rgb, _, _ = pipe.reconstructor.gs_renderer.rasterizer.forward(
            render_splats=[splats],
            render_viewmats=[render_w2c],
            render_Ks=[pred_K],
            render_timestamps=[timestamps],
            sh_degree=0,
            width=width,
            height=height,
        )
        render_time = time.time() - t0

        rendered = rendered_rgb[0]  # [16, H, W, 3]

        for t_local, t in enumerate(chunk):
            for c_idx, cam in enumerate(cameras):
                slot = t_local * NUM_CAMERAS + c_idx
                frame_chw = rendered[slot].permute(2, 0, 1)
                png_path = _png_path(output_dir, take_name, cam, frames_by_cam[cam][t])
                _save_rendered(frame_chw, png_path)
                per_cam_frames[cam].append(_frame_chw_to_uint8_hwc(frame_chw))
                rendered_count += 1

        print(
            f"  Batch {b_idx + 1}/{n_batches} chunk={chunk} "
            f"enc={encode_time:.2f}s render={render_time:.2f}s"
        )
        torch.cuda.empty_cache()

    for cam in cameras:
        save_video(per_cam_frames[cam], str(mp4_paths[cam]), fps=fps)
        print(f"  Wrote {mp4_paths[cam]} ({len(per_cam_frames[cam])} frames @ {fps} fps)")

    print(
        f"  Saved {rendered_count} rendered frames "
        f"({skipped_chunks} chunk(s) skipped via existing PNGs) under "
        f"{output_dir / take_name}"
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-camera reconstruction-video inference for NeoVerse"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes_txt", type=Path,
                     help="Text file with one scene folder name per line.")
    src.add_argument("--input_path", type=Path,
                     help="Single scene directory.")
    parser.add_argument("--scenes_root", type=Path, default=None,
                        help="Root directory containing scene folders (required with --scenes_txt).")
    parser.add_argument("--output_path", type=Path, default=Path("outputs/multi_video"),
                        help="Output directory; per-camera MP4 lands at "
                             "<output_path>/<take_name>/<cam>.mp4 and per-frame "
                             "PNGs at <output_path>/<take_name>/<cam>/frames/<filename>.")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help=f"Camera folder names to use (must be exactly "
                             f"{NUM_CAMERAS}). If omitted, auto-discover per scene.")
    parser.add_argument("--model_path", default="models",
                        help="Local model directory.")
    parser.add_argument("--reconstructor_path",
                        default="models/NeoVerse/reconstructor.ckpt",
                        help="Path to reconstructor checkpoint.")
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--resize_mode", choices=["center_crop", "resize"],
                        default="center_crop")
    parser.add_argument("--fps", type=int, default=24,
                        help="Output MP4 frame rate.")
    parser.add_argument("--max_batches", type=int, default=None,
                        help=f"Cap batches per scene (default {BATCHES_PER_SCENE}).")
    parser.add_argument("--no_skip_frames_in_motion_branch", action="store_true",
                        help=f"Use motion_frame_stride=1 (ablation; default uses "
                             f"stride={NUM_CAMERAS}).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_vram", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cameras is not None and len(args.cameras) != NUM_CAMERAS:
        raise ValueError(
            f"--cameras must list exactly {NUM_CAMERAS} folder names, got {len(args.cameras)}: "
            f"{args.cameras}"
        )

    if args.scenes_txt is not None:
        if args.scenes_root is None:
            raise ValueError("--scenes_root is required with --scenes_txt")
        with open(args.scenes_txt) as f:
            names = [line.strip() for line in f if line.strip()]
        scenes = [(args.scenes_root / n, n) for n in names]
    else:
        scenes = [(args.input_path, args.input_path.name)]

    args.output_path.mkdir(parents=True, exist_ok=True)

    motion_frame_stride = 1 if args.no_skip_frames_in_motion_branch else NUM_CAMERAS
    print(f"motion_frame_stride = {motion_frame_stride}")
    print(f"cameras (script-wide) = {args.cameras}")
    print(f"fps = {args.fps}")

    print(f"Loading model from {args.model_path}...")
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        lora_path=None,
        lora_alpha=1.0,
        device="cuda",
        torch_dtype=torch.bfloat16,
        enable_vram_management=args.low_vram,
    )
    print("Model loaded. Diffusion modules will NOT be invoked.")

    succeeded = 0
    for scene_path, take_name in scenes:
        try:
            ok = render_scene(
                pipe=pipe,
                scene_path=scene_path,
                output_dir=args.output_path,
                height=args.height,
                width=args.width,
                resize_mode=args.resize_mode,
                motion_frame_stride=motion_frame_stride,
                fps=args.fps,
                max_batches=args.max_batches,
                cameras_arg=args.cameras,
            )
            if ok:
                succeeded += 1
        except Exception as e:
            print(f"[WARN] Scene {take_name} failed: {e}")
            continue

    print(f"\nFinished: rendered {succeeded}/{len(scenes)} scenes -> {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

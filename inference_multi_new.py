"""Interleaved-multi-camera NVS evaluation for NeoVerse.

For every non-overlapping 5-frame window with center ``i``, we feed all four
cameras at the four context timestamps ``[i-2, i-1, i+1, i+2]`` to the
reconstructor in time-major interleaved order:

    [(t_{i-2}, c0), (t_{i-2}, c1), (t_{i-2}, c2), (t_{i-2}, c3),
     (t_{i-1}, c0), ..., (t_{i+2}, c3)]                       -> 16 context views

The motion branch pairs same-camera consecutive timestamps via
``motion_frame_stride = num_cameras = 4`` (`--no_skip_frames_in_motion_branch`
disables this for ablation). The held-out timestamp ``t_i`` is rendered for
each of the 4 cameras using each camera's predicted pose at the t_{i-1}
batch slot. Per-camera and overall PSNR / SSIM / LPIPS are reported.

The script never invokes the diffusion stack; only ``pipe.reconstructor`` and
``pipe.reconstructor.gs_renderer.rasterizer`` run.

Example:

    python inference_multi_new.py \
        --scenes_txt /path/to/scenes.txt \
        --scenes_root /path/to/scenes_root \
        --output_path outputs/multi_new_eval/exp_a \
        --reconstructor_path models/NeoVerse/reconstructor.ckpt \
        --height 336 --width 560 --resize_mode center_crop
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth.utils.auxiliary import center_crop, homo_matrix_inverse
from diffsynth.utils.metrics import compute_lpips, compute_psnr, compute_ssim
from diffsynth.utils.multiview import load_frames_from_dir


_IMG_EXTS = {".png", ".jpg", ".jpeg"}


def discover_camera_dirs(scene_path: Path) -> List[str]:
    """Find camera subdirectories of ``scene_path`` purely by listing folders
    that contain image files. Camera-extrinsics metadata is not required —
    this script renders only with predicted poses.
    """
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


NUM_CAMERAS = 4
FRAMES_PER_WINDOW = 5
TARGET_INDEX_IN_WINDOW = 2  # the held-out middle frame
CTX_LOCAL_INDICES = (0, 1, 3, 4)  # frame offsets within a 5-frame window used as context


def _load_and_resize(path: str, width: int, height: int, resize_mode: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if resize_mode == "resize":
        return img.resize((width, height), resample=Image.LANCZOS)
    return center_crop(img, (width, height))


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return F.to_tensor(img)  # [3, H, W] in [0, 1]


def _save_comparison(rendered: torch.Tensor, gt: torch.Tensor, path: Path) -> None:
    rend_np = rendered.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    gt_np = gt.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    side = np.concatenate([rend_np, gt_np], axis=1)
    Image.fromarray((side * 255).clip(0, 255).astype(np.uint8)).save(path)


@torch.no_grad()
def eval_scene(
    pipe,
    scene_path: Path,
    output_dir: Path,
    height: int,
    width: int,
    resize_mode: str,
    motion_frame_stride: int,
    max_windows: Optional[int],
    save_comparison_images: bool,
    max_saved_images: int,
) -> Optional[Dict]:
    take_name = scene_path.name
    print(f"\n=== Scene {take_name} ===")

    cameras = discover_camera_dirs(scene_path)
    if len(cameras) < NUM_CAMERAS:
        print(f"  [SKIP] Found only {len(cameras)} cameras, need {NUM_CAMERAS}")
        return None
    cameras = cameras[:NUM_CAMERAS]
    print(f"  Cameras: {cameras}")

    frames_by_cam: Dict[str, List[str]] = {}
    for cam in cameras:
        frames_by_cam[cam] = load_frames_from_dir(str(scene_path / cam))
    num_frames = len(frames_by_cam[cameras[0]])
    for cam in cameras:
        if len(frames_by_cam[cam]) != num_frames:
            raise ValueError(
                f"Camera {cam} has {len(frames_by_cam[cam])} frames, expected {num_frames}"
            )
    print(f"  {num_frames} frames per camera")

    # Non-overlapping 5-frame windows: starts 0, 5, 10, ... ; centers 2, 7, 12, ...
    centers = list(range(TARGET_INDEX_IN_WINDOW, num_frames - (FRAMES_PER_WINDOW - TARGET_INDEX_IN_WINDOW - 1), FRAMES_PER_WINDOW))
    if max_windows is not None:
        centers = centers[:max_windows]
    if not centers:
        print("  [SKIP] Not enough frames for a single window")
        return None

    device = pipe.device

    per_cam_psnr: Dict[str, List[float]] = {c: [] for c in cameras}
    per_cam_ssim: Dict[str, List[float]] = {c: [] for c in cameras}
    per_cam_lpips: Dict[str, List[float]] = {c: [] for c in cameras}
    per_cam_frame: Dict[str, List[int]] = {c: [] for c in cameras}
    saved_count = 0

    for w_idx, center in enumerate(centers):
        window_starts = [center - 2, center - 1, center + 1, center + 2]
        target_t = center
        # Interleaved time-major ordering: [(t0,c0..c3), (t1,c0..c3), (t3,c0..c3), (t4,c0..c3)]
        ctx_pairs: List[Tuple[int, str]] = []
        for t in window_starts:
            for cam in cameras:
                ctx_pairs.append((t, cam))

        # Build [1, 16, 3, H, W]
        imgs_list = [
            _to_tensor(_load_and_resize(frames_by_cam[cam][t], width, height, resize_mode))
            for (t, cam) in ctx_pairs
        ]
        ctx_imgs = torch.stack(imgs_list, dim=0).unsqueeze(0).to(device)

        # Timestamps spaced by 2 (matches inference_multiview.py convention)
        timestamps = torch.tensor([2 * t for (t, _) in ctx_pairs], dtype=torch.int64, device=device)

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
        splats = predictions["splats"][0]                 # list[len=16] of Gaussians

        # For target camera c, reuse the predicted pose at the t_{i-1} slot
        # (batch position 4 + c — the closest pre-target context for that camera).
        slot_t_minus_1 = NUM_CAMERAS * 1  # = 4
        render_c2w = torch.stack(
            [pred_c2w[slot_t_minus_1 + c] for c in range(NUM_CAMERAS)], dim=0
        )
        render_K = torch.stack(
            [pred_K[slot_t_minus_1 + c] for c in range(NUM_CAMERAS)], dim=0
        )
        render_w2c = homo_matrix_inverse(render_c2w)

        render_timestamps = torch.full(
            (NUM_CAMERAS,), 2 * target_t, dtype=torch.int64, device=device
        )

        t0 = time.time()
        rendered_rgb, _, _ = pipe.reconstructor.gs_renderer.rasterizer.forward(
            render_splats=[splats],
            render_viewmats=[render_w2c],
            render_Ks=[render_K],
            render_timestamps=[render_timestamps],
            sh_degree=0,
            width=width,
            height=height,
        )
        render_time = time.time() - t0

        # rendered_rgb: [1, 4, H, W, 3] in [0, 1]
        rendered = rendered_rgb[0].permute(0, 3, 1, 2)  # [4, 3, H, W]

        # Load GT for the target timestamp per camera
        gt_imgs_list = [
            _to_tensor(_load_and_resize(frames_by_cam[cam][target_t], width, height, resize_mode))
            for cam in cameras
        ]
        gt_imgs = torch.stack(gt_imgs_list, dim=0).to(device)  # [4, 3, H, W]

        # Per-camera metrics
        psnr = compute_psnr(gt_imgs, rendered)
        ssim = compute_ssim(gt_imgs, rendered)
        lpips_val = compute_lpips(gt_imgs, rendered)

        for c_idx, cam in enumerate(cameras):
            per_cam_psnr[cam].append(psnr[c_idx].item())
            per_cam_ssim[cam].append(ssim[c_idx].item())
            per_cam_lpips[cam].append(lpips_val[c_idx].item())
            per_cam_frame[cam].append(target_t)

        avg_psnr = float(psnr.mean().item())
        print(
            f"  Window {w_idx + 1}/{len(centers)} center={target_t} "
            f"enc={encode_time:.2f}s render={render_time:.2f}s "
            f"PSNR={avg_psnr:.2f}"
        )

        if save_comparison_images and (max_saved_images <= 0 or saved_count < max_saved_images):
            imgs_dir = output_dir / take_name / "images"
            imgs_dir.mkdir(parents=True, exist_ok=True)
            for c_idx, cam in enumerate(cameras):
                if max_saved_images > 0 and saved_count >= max_saved_images:
                    break
                _save_comparison(
                    rendered[c_idx],
                    gt_imgs[c_idx],
                    imgs_dir / f"{cam}_t{target_t:04d}_rendered_vs_gt.png",
                )
                saved_count += 1

        torch.cuda.empty_cache()

    # Aggregate per camera
    summary: Dict[str, dict] = {}
    for cam in cameras:
        if not per_cam_psnr[cam]:
            continue
        summary[cam] = {
            "avg_psnr": float(np.mean(per_cam_psnr[cam])),
            "avg_ssim": float(np.mean(per_cam_ssim[cam])),
            "avg_lpips": float(np.mean(per_cam_lpips[cam])),
            "num_frames": len(per_cam_psnr[cam]),
        }

        csv_path = output_dir / f"{take_name}_{cam}_frame_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "psnr", "ssim", "lpips"])
            for fi, p, s, l in zip(
                per_cam_frame[cam],
                per_cam_psnr[cam],
                per_cam_ssim[cam],
                per_cam_lpips[cam],
            ):
                w.writerow([fi, f"{p:.6f}", f"{s:.6f}", f"{l:.6f}"])

    if not summary:
        print(f"  [WARN] No metrics computed for {take_name}")
        return None

    overall = {
        "avg_psnr": float(np.mean([m["avg_psnr"] for m in summary.values()])),
        "avg_ssim": float(np.mean([m["avg_ssim"] for m in summary.values()])),
        "avg_lpips": float(np.mean([m["avg_lpips"] for m in summary.values()])),
    }
    summary["overall"] = overall

    with open(output_dir / f"{take_name}_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  [{take_name}] overall PSNR={overall['avg_psnr']:.2f} "
        f"SSIM={overall['avg_ssim']:.4f} LPIPS={overall['avg_lpips']:.4f}"
    )
    return overall


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interleaved-multi-camera NVS evaluation for NeoVerse"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes_txt", type=Path,
                     help="Text file with one scene folder name per line.")
    src.add_argument("--input_path", type=Path,
                     help="Single scene directory.")
    parser.add_argument("--scenes_root", type=Path, default=None,
                        help="Root directory containing scene folders (required with --scenes_txt).")
    parser.add_argument("--output_path", type=Path, default=Path("outputs/multi_new_eval"),
                        help="Output directory for metrics + optional comparison images.")
    parser.add_argument("--model_path", default="models",
                        help="Local model directory.")
    parser.add_argument("--reconstructor_path",
                        default="models/NeoVerse/reconstructor.ckpt",
                        help="Path to reconstructor checkpoint.")
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--resize_mode", choices=["center_crop", "resize"],
                        default="center_crop")
    parser.add_argument("--max_windows", type=int, default=None,
                        help="Cap windows per scene (debug).")
    parser.add_argument("--no_skip_frames_in_motion_branch", action="store_true",
                        help="Use motion_frame_stride=1 (ablation; default uses stride=4).")
    parser.add_argument("--save_comparison_images", action="store_true",
                        help="Save side-by-side rendered/GT PNGs.")
    parser.add_argument("--max_saved_images", type=int, default=0,
                        help="Cap on saved comparison images per scene (0 = unlimited).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_vram", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    per_scene_overall: Dict[str, Dict] = {}
    for scene_path, take_name in scenes:
        try:
            overall = eval_scene(
                pipe=pipe,
                scene_path=scene_path,
                output_dir=args.output_path,
                height=args.height,
                width=args.width,
                resize_mode=args.resize_mode,
                motion_frame_stride=motion_frame_stride,
                max_windows=args.max_windows,
                save_comparison_images=args.save_comparison_images,
                max_saved_images=args.max_saved_images,
            )
            if overall is not None:
                per_scene_overall[take_name] = overall
        except Exception as e:
            print(f"[WARN] Scene {take_name} failed: {e}")
            continue

    if len(scenes) > 1 and per_scene_overall:
        agg = {
            "scenes": per_scene_overall,
            "macro_avg": {
                "avg_psnr": float(np.mean([m["avg_psnr"] for m in per_scene_overall.values()])),
                "avg_ssim": float(np.mean([m["avg_ssim"] for m in per_scene_overall.values()])),
                "avg_lpips": float(np.mean([m["avg_lpips"] for m in per_scene_overall.values()])),
                "num_scenes": len(per_scene_overall),
            },
        }
        with open(args.output_path / "aggregate_summary.json", "w") as f:
            json.dump(agg, f, indent=2)
        print(
            f"\n=== Aggregate over {len(per_scene_overall)} scenes ===\n"
            f"PSNR={agg['macro_avg']['avg_psnr']:.2f} "
            f"SSIM={agg['macro_avg']['avg_ssim']:.4f} "
            f"LPIPS={agg['macro_avg']['avg_lpips']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

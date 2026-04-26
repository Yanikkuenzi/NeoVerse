"""Interleaved-multi-camera NVS rendering for NeoVerse.

The sliding 5-frame window advances by 1 frame, so every target index ``i`` in
``[2, n-3]`` (where ``n`` is the number of frames common to all cameras) is
rendered. For each center ``i``, we feed all four cameras at the four context
timestamps ``[i-2, i-1, i+1, i+2]`` to the reconstructor in time-major
interleaved order:

    [(t_{i-2}, c0), (t_{i-2}, c1), (t_{i-2}, c2), (t_{i-2}, c3),
     (t_{i-1}, c0), ..., (t_{i+2}, c3)]                       -> 16 context views

The motion branch pairs same-camera consecutive timestamps via
``motion_frame_stride = num_cameras = 4`` (``--no_skip_frames_in_motion_branch``
disables this for ablation). The held-out timestamp ``t_i`` is rendered for
each of the 4 cameras using each camera's predicted pose at the t_{i-1}
batch slot. Rendered frames are written to disk as PNGs; metrics are
computed by a separate downstream script that diffs the output tree against
the GT scene tree.

The script never invokes the diffusion stack; only ``pipe.reconstructor`` and
``pipe.reconstructor.gs_renderer.rasterizer`` run.

Output layout (matches the inference_multiview.py convention so a metrics
script can map output -> GT by identical filename):

    <output_path>/<take_name>/<cam>/<original_frame_filename>

Example:

    python inference_multi_new.py \
        --scenes_txt /path/to/scenes.txt \
        --scenes_root /path/to/scenes_root \
        --output_path outputs/multi_new_eval/exp_a \
        --cameras camera_0000 camera_0001 camera_0002 camera_0003 \
        --reconstructor_path models/NeoVerse/reconstructor.ckpt \
        --height 336 --width 560 --resize_mode center_crop
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

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth.utils.auxiliary import center_crop, homo_matrix_inverse
from diffsynth.utils.multiview import load_frames_from_dir


NUM_CAMERAS = 4
FRAMES_PER_WINDOW = 5
TARGET_INDEX_IN_WINDOW = 2  # the held-out middle frame
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


def _load_and_resize(path: str, width: int, height: int, resize_mode: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if resize_mode == "resize":
        return img.resize((width, height), resample=Image.LANCZOS)
    return center_crop(img, (width, height))


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return F.to_tensor(img)  # [3, H, W] in [0, 1]


def _save_rendered(rendered_chw: torch.Tensor, path: Path) -> None:
    arr = (rendered_chw.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def _resolve_cameras_for_scene(
    scene_path: Path, requested: Optional[List[str]]
) -> Optional[List[str]]:
    """Return the camera-folder names to use for ``scene_path``.

    If ``requested`` is given, keep each requested camera that has frames; for
    each missing-or-empty one, substitute the next available (non-empty) camera
    in sorted order that isn't already in the chosen set. If the scene can't
    fill ``NUM_CAMERAS`` slots even after substitution, returns None.

    If ``requested`` is None, auto-discover non-empty cameras and take the first
    ``NUM_CAMERAS``.
    """
    available = discover_camera_dirs(scene_path)  # already filters out empty dirs

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
    """Compute the per-frame intersection (by basename) across cameras.

    Returns a list of dicts ``{cam: full_path}`` in sorted-filename order.
    Prints a warning per camera that contributed extra (dropped) frames.
    """
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


@torch.no_grad()
def render_scene(
    pipe,
    scene_path: Path,
    output_dir: Path,
    height: int,
    width: int,
    resize_mode: str,
    motion_frame_stride: int,
    max_windows: Optional[int],
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
    if num_frames < FRAMES_PER_WINDOW:
        print(
            f"  [SKIP] Only {num_frames} frame(s) common to all cameras; need "
            f"≥ {FRAMES_PER_WINDOW}"
        )
        return False
    print(f"  {num_frames} frames common to all cameras")

    # Per-camera ordered path list that is index-aligned with the intersection.
    frames_by_cam: Dict[str, List[str]] = {
        cam: [row[cam] for row in intersection] for cam in cameras
    }

    centers = list(range(
        TARGET_INDEX_IN_WINDOW,
        num_frames - (FRAMES_PER_WINDOW - TARGET_INDEX_IN_WINDOW - 1),
        1,
    ))
    if max_windows is not None:
        centers = centers[:max_windows]
    if not centers:
        print("  [SKIP] Not enough frames for a single window")
        return False

    # Output dirs (one per camera under the scene).
    for cam in cameras:
        (output_dir / take_name / cam).mkdir(parents=True, exist_ok=True)

    device = pipe.device
    rendered_count = 0
    skipped_existing = 0

    for w_idx, center in enumerate(centers):
        window_starts = [center - 2, center - 1, center + 1, center + 2]
        target_t = center

        # Resume support: if all 4 target PNGs for this window already exist on
        # disk, skip the reconstruction+render entirely.
        out_paths = [
            output_dir / take_name / cam / Path(frames_by_cam[cam][target_t]).name
            for cam in cameras
        ]
        if all(p.exists() for p in out_paths):
            skipped_existing += 1
            print(
                f"  Window {w_idx + 1}/{len(centers)} center={target_t} "
                f"[SKIP] outputs already exist"
            )
            continue

        # Time-major interleaved ordering: [(t0,c0..c3), (t1,c0..c3), (t3,c0..c3), (t4,c0..c3)]
        ctx_pairs: List[Tuple[int, str]] = []
        for t in window_starts:
            for cam in cameras:
                ctx_pairs.append((t, cam))

        imgs_list = [
            _to_tensor(_load_and_resize(frames_by_cam[cam][t], width, height, resize_mode))
            for (t, cam) in ctx_pairs
        ]
        ctx_imgs = torch.stack(imgs_list, dim=0).unsqueeze(0).to(device)

        # Timestamps spaced by 2 (matches inference_multiview.py convention).
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
        splats = predictions["splats"][0]

        # Reuse each camera's predicted pose at the t_{i-1} slot
        # (batch position 4 + c — closest pre-target context for that camera).
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

        # Save under the GT filename for that target frame so a downstream
        # metrics script can map output -> GT trivially.
        for c_idx, out_path in enumerate(out_paths):
            _save_rendered(rendered[c_idx], out_path)
            rendered_count += 1

        print(
            f"  Window {w_idx + 1}/{len(centers)} center={target_t} "
            f"enc={encode_time:.2f}s render={render_time:.2f}s"
        )
        torch.cuda.empty_cache()

    print(
        f"  Saved {rendered_count} rendered frames "
        f"({skipped_existing} window(s) skipped as already rendered) under "
        f"{output_dir / take_name}"
    )
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interleaved-multi-camera NVS rendering for NeoVerse"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--scenes_txt", type=Path,
                     help="Text file with one scene folder name per line.")
    src.add_argument("--input_path", type=Path,
                     help="Single scene directory.")
    parser.add_argument("--scenes_root", type=Path, default=None,
                        help="Root directory containing scene folders (required with --scenes_txt).")
    parser.add_argument("--output_path", type=Path, default=Path("outputs/multi_new"),
                        help="Output directory; rendered PNGs land at "
                             "<output_path>/<take_name>/<cam>/<filename>.")
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
    parser.add_argument("--max_windows", type=int, default=None,
                        help="Cap windows per scene (debug).")
    parser.add_argument("--no_skip_frames_in_motion_branch", action="store_true",
                        help="Use motion_frame_stride=1 (ablation; default uses stride=4).")
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
                max_windows=args.max_windows,
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

import torch
import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth.utils.auxiliary import homo_matrix_inverse, center_crop
from diffsynth.utils.multiview import (
    get_camera_extrinsics,
    transform_gaussians_to_world,
    load_frames_from_dir,
)


def _load_and_resize(path, width, height, resize_mode):
    img = Image.open(path).convert("RGB")
    if resize_mode == "resize":
        return img.resize((width, height), resample=Image.LANCZOS)
    else:
        return center_crop(img, (width, height))


@torch.no_grad()
def multiview_eval(pipe, cameras, input_path, output_path, height, width,
                   batch_size, resize_mode):
    if batch_size % 2 == 0:
        batch_size += 1
    assert batch_size >= 3, "batch_size must be >= 3 (need at least 2 context frames)"

    device = pipe.device
    input_path = Path(input_path)

    # Load ALL frame paths per camera, split into even (context) / odd (target)
    camera_context_paths = {}
    camera_target_names = {}
    total_frames = None
    for cam_name in cameras:
        all_paths = load_frames_from_dir(str(input_path / cam_name))
        if total_frames is None:
            total_frames = len(all_paths)
        elif len(all_paths) != total_frames:
            raise ValueError(
                f"{cam_name} has {len(all_paths)} frames but {cameras[0]} has "
                f"{total_frames}. All cameras must have the same number of frames."
            )
        camera_context_paths[cam_name] = [all_paths[i] for i in range(0, len(all_paths), 2)]
        camera_target_names[cam_name] = [Path(all_paths[i]).name for i in range(1, len(all_paths), 2)]

    S = len(camera_context_paths[cameras[0]])  # total context frames per camera
    total_targets = S - 1
    print(f"  {total_frames} total frames per camera -> {S} context, {total_targets} targets")

    # Preload GT extrinsics
    camera_c2w = {}
    for cam_name in cameras:
        c2w, _ = get_camera_extrinsics(input_path, cam_name)
        camera_c2w[cam_name] = c2w.to(device)

    # Create output dirs
    for cam_name in cameras:
        os.makedirs(os.path.join(output_path, cam_name), exist_ok=True)

    # Sliding window loop
    stride = batch_size - 1
    start = 0
    while start < S:
        end = min(start + batch_size, S)
        W = end - start
        num_window_targets = W - 1

        if num_window_targets == 0:
            break

        print(f"  Window [{start}:{end}) — {W} context frames, {num_window_targets} targets")

        # --- Phase 1: Reconstruct all cameras for this window ---
        window_gaussians = []
        window_intrinsics = {}

        for cam_name in cameras:
            window_paths = camera_context_paths[cam_name][start:end]
            images = [_load_and_resize(p, width, height, resize_mode) for p in window_paths]

            img_tensor = torch.stack(
                [F.to_tensor(img)[None] for img in images], dim=1
            ).to(device)  # [1, W, 3, H, W]

            # Global timestamps: context frame i -> timestamp 2*i
            timestamps = torch.arange(2 * start, 2 * end, 2,
                                      dtype=torch.int64, device=device)

            views = {
                "img": img_tensor,
                "is_target": torch.zeros((1, W), dtype=torch.bool, device=device),
                "is_static": torch.zeros((1, W), dtype=torch.bool, device=device),
                "timestamp": timestamps.unsqueeze(0),
            }

            if pipe.vram_management_enabled:
                pipe.reconstructor.to(device)

            with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
                predictions = pipe.reconstructor(views, is_inference=True, use_motion=True)

            if pipe.vram_management_enabled:
                pipe.reconstructor.cpu()
                torch.cuda.empty_cache()

            window_intrinsics[cam_name] = predictions["rendered_intrinsics"][0]  # [W, 3, 3]

            gt_c2w = camera_c2w[cam_name]
            for gs in predictions["splats"][0]:
                window_gaussians.append(transform_gaussians_to_world(gs, gt_c2w))

        # --- Phase 2: Render from each camera at this window's target timestamps ---
        # Context timestamps: [2*start, 2*start+2, ..., 2*(end-1)]
        # Target timestamps:  [2*start+1, 2*start+3, ..., 2*(end-1)-1]
        eval_timestamps = torch.arange(2 * start + 1, 2 * (end - 1), 2,
                                       dtype=torch.int64, device=device)
        assert len(eval_timestamps) == num_window_targets

        for cam_name in cameras:
            render_w2c = homo_matrix_inverse(camera_c2w[cam_name].unsqueeze(0))
            render_w2c = render_w2c.expand(num_window_targets, -1, -1)

            # Target i sits between context i and i+1; use context i's intrinsics
            render_K = window_intrinsics[cam_name][:num_window_targets]

            eval_rgb, _, _ = pipe.reconstructor.gs_renderer.rasterizer.forward(
                render_splats=[window_gaussians],
                render_viewmats=[render_w2c],
                render_Ks=[render_K],
                render_timestamps=[eval_timestamps],
                sh_degree=0, width=width, height=height,
            )

            # Save — target index in the global list is (start + i)
            cam_out_dir = os.path.join(output_path, cam_name)
            for i in range(num_window_targets):
                name = camera_target_names[cam_name][start + i]
                frame = (eval_rgb[0, i].clamp(0, 1) * 255).byte().cpu().numpy()
                Image.fromarray(frame).save(os.path.join(cam_out_dir, name))

        print(f"    Rendered {num_window_targets} targets from {len(cameras)} cameras")

        # Free memory
        del window_gaussians, window_intrinsics
        torch.cuda.empty_cache()

        start += stride

    print(f"  Saved {total_targets} target frames per camera to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeoVerse Multi-View Evaluation",
    )
    parser.add_argument("--input_path", type=Path, required=True,
                        help="Base directory with camera_*/ subdirs and models.json")
    parser.add_argument("--cameras", nargs="+", required=True,
                        help="Camera names to use (must match names in models.json)")
    parser.add_argument("--output_path", default="outputs/multiview_eval",
                        help="Output directory for rendered frames (default: outputs/multiview_eval)")
    parser.add_argument("--model_path", default="models",
                        help="Model directory path (default: models)")
    parser.add_argument("--reconstructor_path",
                        default="models/NeoVerse/reconstructor.ckpt",
                        help="Path to reconstructor checkpoint")
    parser.add_argument("--height", type=int, default=336,
                        help="Render height (default: 336)")
    parser.add_argument("--width", type=int, default=560,
                        help="Render width (default: 560)")
    parser.add_argument("--resize_mode", choices=["center_crop", "resize"],
                        default="center_crop",
                        help="Image resize mode (default: center_crop)")
    parser.add_argument("--batch_size", type=int, default=41,
                        help="Max context frames per reconstructor call, must be odd (default: 41)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode with model offloading")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model (only reconstructor needed)
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
    print("Model loaded!")

    print(f"Running multi-view evaluation with {len(args.cameras)} cameras...")
    multiview_eval(
        pipe=pipe,
        cameras=args.cameras,
        input_path=args.input_path,
        output_path=args.output_path,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        resize_mode=args.resize_mode,
    )

    print(f"Done! Output saved to: {args.output_path}")
    return 0


if __name__ == "__main__":
    exit(main())

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


@torch.no_grad()
def multiview_eval(pipe, cameras, input_path, output_path, height, width,
                   num_frames, resize_mode):
    device = pipe.device
    input_path = Path(input_path)

    all_gaussians = []
    pred_intrinsics_per_camera = {}
    target_names_per_camera = {}
    num_context = None

    # Phase 1: Reconstruct each camera independently, collect Gaussians
    for cam_name in cameras:
        print(f"  Reconstructing {cam_name}...")

        gt_c2w, _ = get_camera_extrinsics(input_path, cam_name)
        gt_c2w = gt_c2w.to(device)

        # Load all frames, pick even as context, odd as targets
        cam_dir = str(input_path / cam_name)
        all_frame_paths = load_frames_from_dir(cam_dir)
        selected = all_frame_paths[:2 * num_frames]
        if len(selected) < 2 * num_frames:
            print(f"    Warning: found {len(selected)} images, need {2 * num_frames}")

        context_paths = [selected[i] for i in range(0, len(selected), 2)]
        target_names = [Path(selected[i]).name for i in range(1, len(selected), 2)]
        target_names_per_camera[cam_name] = target_names

        S = len(context_paths)
        if num_context is None:
            num_context = S
        elif S != num_context:
            raise ValueError(
                f"{cam_name} has {S} context frames but expected {num_context}. "
                "All cameras must have the same number of frames."
            )

        # Load and resize context frames
        resolution = (width, height)
        if resize_mode == "resize":
            images = [Image.open(p).convert("RGB").resize(resolution, resample=Image.LANCZOS)
                      for p in context_paths]
        else:
            images = [center_crop(Image.open(p).convert("RGB"), resolution)
                      for p in context_paths]

        # Build views dict with even timestamps (same as monocular eval)
        img_tensor = torch.stack(
            [F.to_tensor(img)[None] for img in images], dim=1
        ).to(device)  # [1, S, 3, H, W]

        views = {
            "img": img_tensor,
            "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, S), dtype=torch.bool, device=device),
            "timestamp": torch.arange(0, 2 * S, 2, dtype=torch.int64, device=device).unsqueeze(0),
        }

        # Run reconstructor with motion prediction
        if pipe.vram_management_enabled:
            pipe.reconstructor.to(device)

        with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
            predictions = pipe.reconstructor(views, is_inference=True, use_motion=True)

        if pipe.vram_management_enabled:
            pipe.reconstructor.cpu()
            torch.cuda.empty_cache()

        # Store predicted intrinsics for rendering later
        pred_intrinsics_per_camera[cam_name] = predictions["rendered_intrinsics"][0]  # [S, 3, 3]

        # Transform Gaussians to world frame and collect
        for gs in predictions["splats"][0]:
            all_gaussians.append(transform_gaussians_to_world(gs, gt_c2w))

    # Phase 2: Render from each camera's viewpoint at odd timestamps
    num_targets = num_context - 1
    eval_timestamps = torch.arange(1, 2 * num_context - 1, 2, dtype=torch.int64, device=device)

    for cam_name in cameras:
        print(f"  Rendering from {cam_name}...")

        gt_c2w, _ = get_camera_extrinsics(input_path, cam_name)
        gt_c2w = gt_c2w.to(device)

        render_w2c = homo_matrix_inverse(gt_c2w.unsqueeze(0))  # [1, 4, 4]
        render_w2c = render_w2c.expand(num_targets, -1, -1)

        # Use predicted intrinsics (first S entries, take first num_targets)
        render_K = pred_intrinsics_per_camera[cam_name][:num_targets]

        eval_rgb, _, _ = pipe.reconstructor.gs_renderer.rasterizer.forward(
            render_splats=[all_gaussians],
            render_viewmats=[render_w2c],
            render_Ks=[render_K],
            render_timestamps=[eval_timestamps],
            sh_degree=0, width=width, height=height,
        )

        # Save rendered frames
        cam_out_dir = os.path.join(output_path, cam_name)
        os.makedirs(cam_out_dir, exist_ok=True)
        target_names = target_names_per_camera[cam_name]
        for i, name in enumerate(target_names[:num_targets]):
            frame = (eval_rgb[0, i].clamp(0, 1) * 255).byte().cpu().numpy()
            Image.fromarray(frame).save(os.path.join(cam_out_dir, name))

        print(f"    Saved {num_targets} frames to {cam_out_dir}")


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
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of context frames per view (default: 81)")
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
        num_frames=args.num_frames,
        resize_mode=args.resize_mode,
    )

    print(f"Done! Output saved to: {args.output_path}")
    return 0


if __name__ == "__main__":
    exit(main())

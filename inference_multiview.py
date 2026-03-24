import torch
import os
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import homo_matrix_inverse, load_video
from diffsynth.utils.multiview import (
    transform_gaussians_to_world,
    load_multiview_config,
    load_frames_from_dir,
)


@torch.no_grad()
def multiview_render(pipe, config, height, width, num_frames):
    device = pipe.device

    views_config = config["views"]
    render_vp = config["render_viewpoint"]

    all_gaussians = []
    num_timestamps = None

    for i, view_spec in enumerate(views_config):
        print(f"  Processing view {i}: {view_spec['image_dir']}")

        # Load frames from directory in sorted order
        frame_paths = load_frames_from_dir(view_spec["image_dir"])
        images = load_video(frame_paths, num_frames,
                            resolution=(width, height), resize_mode="resize")
        S = len(images)

        if num_timestamps is None:
            num_timestamps = S
        elif S != num_timestamps:
            raise ValueError(
                f"View {i} has {S} frames but view 0 has {num_timestamps}. "
                "All views must have the same number of frames."
            )

        # Build views dict for reconstructor
        img_tensor = torch.stack(
            [F.to_tensor(img)[None] for img in images], dim=1
        ).to(device)  # [1, S, 3, H, W]

        views = {
            "img": img_tensor,
            "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, S), dtype=torch.bool, device=device),
            "timestamp": torch.arange(0, S, dtype=torch.int64, device=device).unsqueeze(0),
        }

        # Run reconstructor
        if pipe.vram_management_enabled:
            pipe.reconstructor.to(device)

        with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
            predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

        if pipe.vram_management_enabled:
            pipe.reconstructor.cpu()
            torch.cuda.empty_cache()

        # Parse GT c2w for this view
        gt_c2w = torch.tensor(
            view_spec["c2w"], dtype=torch.float32, device=device
        )

        # Transform each frame's Gaussians to world frame and collect
        for gs in predictions["splats"][0]:
            all_gaussians.append(transform_gaussians_to_world(gs, gt_c2w))

    # Parse render viewpoint
    render_c2w = torch.tensor(
        render_vp["c2w"], dtype=torch.float32, device=device
    )
    render_K = torch.tensor(
        render_vp["intrinsics"], dtype=torch.float32, device=device
    )

    # Compute w2c and expand for all timestamps
    render_w2c = homo_matrix_inverse(render_c2w.unsqueeze(0))  # [1, 4, 4]
    render_w2c = render_w2c.expand(num_timestamps, -1, -1)      # [S, 4, 4]
    render_Ks = render_K.unsqueeze(0).expand(num_timestamps, -1, -1)  # [S, 3, 3]
    timestamps = torch.arange(0, num_timestamps, dtype=torch.int64, device=device)

    # Render all timestamps
    print(f"  Rendering {num_timestamps} frames from merged Gaussians...")
    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        render_splats=[all_gaussians],
        render_viewmats=[render_w2c],
        render_Ks=[render_Ks],
        render_timestamps=[timestamps],
        sh_degree=0, width=width, height=height,
    )
    # target_rgb: [1, S, H, W, 3]

    # Convert rendered frames to PIL images for save_video
    frames = []
    for t in range(num_timestamps):
        frame = target_rgb[0, t].clamp(0, 1).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frames.append(Image.fromarray(frame))

    return frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeoVerse Multi-View Gaussian Merge & Render",
    )
    parser.add_argument("--config", required=True,
                        help="Path to multi-view JSON config file")
    parser.add_argument("--output_path", default="outputs/multiview_render.mp4",
                        help="Output video path (default: outputs/multiview_render.mp4)")
    parser.add_argument("--model_path", default="models",
                        help="Model directory path (default: models)")
    parser.add_argument("--reconstructor_path",
                        default="models/NeoVerse/reconstructor.ckpt",
                        help="Path to reconstructor checkpoint")
    parser.add_argument("--height", type=int, default=336,
                        help="Render height (default: 336)")
    parser.add_argument("--width", type=int, default=560,
                        help="Render width (default: 560)")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to sample per view (default: 81)")
    parser.add_argument("--fps", type=int, default=16,
                        help="Output video FPS (default: 16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode with model offloading")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    print(f"Loading config from {args.config}...")
    config = load_multiview_config(args.config)
    print(f"  {len(config['views'])} views configured")

    # Load model (only reconstructor is needed, no diffusion)
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

    # Run multi-view inference
    print("Running multi-view reconstruction and rendering...")
    frames = multiview_render(
        pipe=pipe,
        config=config,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )

    # Save output
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    save_video(frames, args.output_path, fps=args.fps)
    print(f"Done! Output saved to: {args.output_path}")
    return 0


if __name__ == "__main__":
    exit(main())

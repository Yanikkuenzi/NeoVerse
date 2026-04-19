import torch
import os
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import CameraTrajectory, load_video, center_crop, homo_matrix_inverse
from diffsynth.auxiliary_models.worldmirror.utils.render_utils import (
    slerp_quaternions,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)


def _interpolate_c2w_midpoint(c2w):
    R0, R1 = c2w[:-1, :3, :3], c2w[1:, :3, :3]
    t0, t1 = c2w[:-1, :3, 3],  c2w[1:, :3, 3]
    q0 = rotation_matrix_to_quaternion(R0)
    q1 = rotation_matrix_to_quaternion(R1)
    q = slerp_quaternions(q0, q1, 0.5)
    R = quaternion_to_rotation_matrix(q)
    out = torch.eye(4, device=c2w.device, dtype=c2w.dtype).expand(R.shape[0], -1, -1).clone()
    out[:, :3, :3] = R
    out[:, :3, 3] = 0.5 * (t0 + t1)
    return out


@torch.no_grad()
def evaluate_batched(pipe, input_path, output_path, height, width,
                     batch_size, resize_mode, non_static_cameras=False):
    if batch_size % 2 == 0:
        batch_size += 1
    assert batch_size >= 3, "batch_size must be >= 3 (need at least 2 context frames)"

    device = pipe.device
    resolution = (width, height)

    # Load ALL frames from directory, split even/odd
    all_names = sorted(os.listdir(input_path))
    all_names = [n for n in all_names if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    context_names = [all_names[i] for i in range(0, len(all_names), 2)]
    target_names = [all_names[i] for i in range(1, len(all_names), 2)]

    S = len(context_names)
    total_targets = len(target_names)
    print(f"  {len(all_names)} total frames -> {S} context, {total_targets} targets")

    os.makedirs(output_path, exist_ok=True)

    # Sliding window
    stride = batch_size - 1
    start = 0
    while start < S:
        end = min(start + batch_size, S)
        W = end - start
        num_window_targets = W - 1

        if num_window_targets == 0:
            break

        print(f"  Window [{start}:{end}) — {W} context frames, {num_window_targets} targets")

        # Load and resize window's context frames
        window_names = context_names[start:end]
        if resize_mode == "resize":
            images = [Image.open(os.path.join(input_path, n)).convert("RGB")
                      .resize(resolution, resample=Image.LANCZOS) for n in window_names]
        else:
            images = [center_crop(Image.open(os.path.join(input_path, n)).convert("RGB"),
                      resolution) for n in window_names]

        img_tensor = torch.stack(
            [F.to_tensor(img)[None] for img in images], dim=1
        ).to(device)  # [1, W, 3, H, W]

        timestamps = torch.arange(2 * start, 2 * end, 2,
                                  dtype=torch.int64, device=device)

        views = {
            "img": img_tensor,
            "is_target": torch.zeros((1, W), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, W), dtype=torch.bool, device=device),
            "timestamp": timestamps.unsqueeze(0),
        }

        # Reconstruct
        if pipe.vram_management_enabled:
            pipe.reconstructor.to(device)

        with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
            predictions = pipe.reconstructor(views, is_inference=True, use_motion=True)

        if pipe.vram_management_enabled:
            pipe.reconstructor.cpu()
            torch.cuda.empty_cache()

        gaussians = predictions["splats"]
        K = predictions["rendered_intrinsics"][0]       # [W, 3, 3]
        c2w = predictions["rendered_extrinsics"][0]     # [W, 4, 4]

        # Render at odd timestamps from first frame's viewpoint
        eval_timestamps = torch.arange(2 * start + 1, 2 * (end - 1), 2,
                                       dtype=torch.int64, device=device)
        assert len(eval_timestamps) == num_window_targets

        if non_static_cameras:
            eval_c2w = _interpolate_c2w_midpoint(c2w)
        else:
            eval_c2w = c2w[0:1].repeat(num_window_targets, 1, 1)
        eval_w2c = homo_matrix_inverse(eval_c2w)
        eval_K = K[:num_window_targets]

        eval_rgb, _, _ = pipe.reconstructor.gs_renderer.rasterizer.forward(
            gaussians,
            render_viewmats=[eval_w2c],
            render_Ks=[eval_K],
            render_timestamps=[eval_timestamps],
            sh_degree=0, width=width, height=height,
        )

        # Save
        for i in range(num_window_targets):
            name = target_names[start + i]
            frame = (eval_rgb[0, i].clamp(0, 1) * 255).byte().cpu().numpy()
            Image.fromarray(frame).save(os.path.join(output_path, name))

        del gaussians, eval_rgb
        torch.cuda.empty_cache()

        start += stride

    print(f"Saved {total_targets} evaluation frames to {output_path}")


@torch.no_grad()
def generate_video(pipe, input_video, prompt, negative_prompt, cam_traj: CameraTrajectory,
                   output_path="outputs/output.mp4", alpha_threshold=1.0, static_flag=False,
                   seed=42, cfg_scale=1.0, num_inference_steps=4, skip_diffusion=False):
    device = pipe.device
    height, width = input_video[0].size[1], input_video[0].size[0]
    views = {
        "img": torch.stack([F.to_tensor(image)[None] for image in input_video], dim=1).to(device),
        "is_target": torch.zeros((1, len(input_video)), dtype=torch.bool, device=device),
    }
    if static_flag:
        views["is_static"] = torch.ones((1, len(input_video)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, len(input_video)), dtype=torch.int64, device=device)
    else:
        views["is_static"] = torch.zeros((1, len(input_video)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, len(input_video), dtype=torch.int64, device=device).unsqueeze(0)

    # Low-VRAM: load reconstructor to GPU before use
    if pipe.vram_management_enabled:
        pipe.reconstructor.to(device)

    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

    # Low-VRAM: offload reconstructor back to CPU
    if pipe.vram_management_enabled:
        pipe.reconstructor.cpu()
        torch.cuda.empty_cache()

    gaussians = predictions["splats"]
    K = predictions["rendered_intrinsics"][0]
    input_cam2world = predictions["rendered_extrinsics"][0]
    timestamps = predictions["rendered_timestamps"][0]

    if static_flag:
        K = K[:1].repeat(len(cam_traj), 1, 1)
        timestamps = timestamps[:1].repeat(len(cam_traj))

    # Apply per-trajectory zoom_ratio
    ratio = torch.linspace(1, cam_traj.zoom_ratio, K.shape[0], device=device)
    K_zoomed = K.clone()
    K_zoomed[:, 0, 0] *= ratio
    K_zoomed[:, 1, 1] *= ratio

    target_cam2world = cam_traj.c2w.to(device)
    if cam_traj.mode == "relative" and not static_flag:
        target_cam2world = input_cam2world @ target_cam2world
    target_world2cam = homo_matrix_inverse(target_cam2world)
    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[target_world2cam],
        render_Ks=[K_zoomed],
        render_timestamps=[timestamps],
        sh_degree=0, width=width, height=height,
    )
    target_mask = (target_alpha > alpha_threshold).float()
    if cam_traj.use_first_frame:
        target_rgb[0, 0] = views["img"][0, 0].permute(1, 2, 0)
        target_mask[0, 0] = 1.0
    wrapped_data = {
        "source_views": views,
        "target_rgb": target_rgb,
        "target_depth": target_depth,
        "target_mask": target_mask,
        "target_poses": target_cam2world.unsqueeze(0),
        "target_intrs": K_zoomed.unsqueeze(0),
    }
    if skip_diffusion:
        # Bypass the diffusion denoiser and save reconstructor renderings directly.
        # `target_rgb` is expected in float [0,1], shape (batch, frames, H, W, 3).
        trgb = target_rgb.detach().cpu().numpy()
        trgb = np.clip(trgb, 0.0, 1.0)
        frames = []
        for f in trgb[0]:
            img = (f * 255).astype(np.uint8)
            frames.append(Image.fromarray(img))
        save_video(frames, output_path, fps=16)
        return
    generated_frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed, rand_device=pipe.device,
        height=height, width=width, num_frames=len(target_cam2world),
        cfg_scale=cfg_scale, num_inference_steps=num_inference_steps, tiled=False,
        **wrapped_data,
    )
    save_video(generated_frames, output_path, fps=16)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeoVerse Unified Inference",
    )

    # Trajectory specification (mutually exclusive, not required in --evaluate mode)
    traj_group = parser.add_mutually_exclusive_group(required=False)
    traj_group.add_argument("--trajectory",
                            choices=["pan_left", "pan_right", "tilt_up", "tilt_down",
                                     "move_left", "move_right", "push_in", "pull_out",
                                     "boom_up", "boom_down", "orbit_left", "orbit_right",
                                     "static"],
                            help="Predefined trajectory type")
    traj_group.add_argument("--trajectory_file",
                            help="Path to JSON trajectory file")

    # Predefined trajectory parameters
    parser.add_argument("--angle", type=float,
                        help="Override rotation angle for pan/tilt/orbit")
    parser.add_argument("--distance", type=float,
                        help="Override translation distance for move/push/pull/boom")
    parser.add_argument("--orbit_radius", type=float,
                        help="Override orbit radius")
    parser.add_argument("--traj_mode", choices=["relative", "global"], default="relative",
                        help="Trajectory mode (default: relative)")
    parser.add_argument("--zoom_ratio", type=float, default=1.0,
                        help="Zoom factor for zoom_in/zoom_out (default: 1.0)")

    # Validation only
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate trajectory file, don't run inference")

    # Input/output
    parser.add_argument("--input_path", help="Input video or image path")
    parser.add_argument("--output_path", default="outputs/inference.mp4",
                        help="Output video path (default: outputs/inference.mp4)")
    parser.add_argument("--prompt", default="A smooth video with complete scene content. Inpaint any missing regions or margins naturally to match the surrounding scene.",
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", default="",
                        help="Negative text prompt")

    # Model parameters
    parser.add_argument("--model_path", default="models",
                        help="Model directory path (default: models)")
    parser.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt",
                        help="Path to reconstructor checkpoint")
    parser.add_argument("--disable_lora", action="store_true",
                        help="Skip distilled LoRA loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames (default: 81)")

    # Video loading
    parser.add_argument("--height", type=int, default=336,
                        help="Output height (default: 336)")
    parser.add_argument("--width", type=int, default=560,
                        help="Output width (default: 560)")
    parser.add_argument("--resize_mode", choices=["center_crop", "resize"],
                        default="center_crop",
                        help="Video resize mode (default: center_crop)")

    # Advanced
    parser.add_argument("--alpha_threshold", type=float, default=1.0,
                        help="Alpha mask threshold (0.0-1.0)")
    parser.add_argument("--static_scene", action="store_true",
                        help="Enable static scene mode")
    parser.add_argument("--vis_rendering", action="store_true",
                        help="Save intermediate rendering visualizations")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode with model offloading (reduces peak VRAM usage)")
    parser.add_argument("--skip_diffusion", action="store_true",
                        help="Bypass the diffusion model and save reconstructor renderings directly")

    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate frame interpolation: feed even frames, render odd-timestep frames via Gaussian splatting")
    parser.add_argument("--evaluate_output_path", default="outputs/evaluate",
                        help="Directory to save evaluation frames (default: outputs/evalua te)")
    parser.add_argument("--batch_size", type=int, default=41,
                        help="Max context frames per reconstructor call in evaluate mode, must be odd (default: 41)")
    parser.add_argument("--non-static-cameras", dest="non_static_cameras", action="store_true",
                        help="Interpolate predicted c2w between adjacent context frames (eval mode, non-static camera datasets)")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- LoRA / inference params ---
    use_lora = not args.disable_lora
    num_inference_steps = 4 if use_lora else 50
    cfg_scale = 1.0 if use_lora else 5.0

    lora_path = os.path.join(
        args.model_path,
        "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors"
    ) if use_lora else None

    # --- Validate-only mode ---
    if args.validate_only:
        if args.trajectory_file is None:
            print("Error: --validate_only requires --trajectory_file")
            return 1
        print(f"Validating trajectory file: {args.trajectory_file}")
        try:
            data = CameraTrajectory.validate_json(args.trajectory_file)
            fmt = "Keyframe operations" if "keyframes" in data else "Direct matrices"
            count = len(data.get("keyframes", data.get("trajectory", [])))
            print(f"  Format: {fmt}")
            print(f"  Entries: {count}")
            print(f"  Mode: {data.get('mode', 'relative')}")
            print("Validation passed!")
            return 0
        except ValueError as e:
            print(f"Validation failed: {e}")
            return 1

    # --- Normal inference mode ---
    if args.input_path is None:
        print("Error: --input_path is required for inference")
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build trajectory (not needed in --evaluate mode — pose is taken from
    # context frames and optionally interpolated via --non-static-cameras)
    cam_traj = None
    if not args.evaluate:
        if args.trajectory is None and args.trajectory_file is None:
            print("Error: --trajectory or --trajectory_file is required for non-evaluate inference")
            return 1
        if args.trajectory:
            cam_traj = CameraTrajectory.from_predefined(
                args.trajectory,
                num_frames=args.num_frames,
                mode=args.traj_mode,
                angle=args.angle,
                distance=args.distance,
                orbit_radius=args.orbit_radius,
                zoom_ratio=args.zoom_ratio,
            )
        else:
            cam_traj = CameraTrajectory.from_json(args.trajectory_file)

    # Load model
    print(f"Loading model from {args.model_path}...")
    pipe = WanVideoNeoVersePipeline.from_pretrained(
        local_model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        lora_path=lora_path,
        lora_alpha=1.0,
        device="cuda",
        torch_dtype=torch.bfloat16,
        enable_vram_management=args.low_vram,
    )
    print("Model loaded!")

    # --- Evaluate mode: batched sliding window ---
    if args.evaluate:
        if not os.path.isdir(args.input_path):
            print("Error: --evaluate requires --input_path to be a directory of images")
            return 1
        print(f"Running batched evaluation...")
        evaluate_batched(
            pipe=pipe,
            input_path=args.input_path,
            output_path=args.evaluate_output_path,
            height=args.height,
            width=args.width,
            batch_size=args.batch_size,
            resize_mode=args.resize_mode,
            non_static_cameras=args.non_static_cameras,
        )
        print(f"Done! Output saved to: {args.evaluate_output_path}")
        return 0

    # --- Normal inference ---
    print(f"Loading video from {args.input_path}...")
    images = load_video(args.input_path, args.num_frames,
                        resolution=(args.width, args.height),
                        resize_mode=args.resize_mode,
                        static_scene=args.static_scene)

    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.vis_rendering:
        vis_dir = os.path.splitext(output_path)[0]
        os.makedirs(vis_dir, exist_ok=True)
        pipe.save_root = vis_dir

    print(f"Generating with trajectory: {cam_traj.name} (mode={cam_traj.mode})")
    generate_video(
        pipe=pipe,
        input_video=images,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cam_traj=cam_traj,
        output_path=output_path,
        alpha_threshold=args.alpha_threshold,
        static_flag=args.static_scene,
        seed=args.seed,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        skip_diffusion=args.skip_diffusion,
    )
    print(f"Done! Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())

import torch
import os
import argparse
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import CameraTrajectory, load_video, center_crop, homo_matrix_inverse
from PIL import Image


@torch.no_grad()
def generate_video(pipe, input_video, prompt, negative_prompt, cam_traj: CameraTrajectory,
                   output_path="outputs/output.mp4", alpha_threshold=1.0, static_flag=False,
                   seed=42, cfg_scale=1.0, num_inference_steps=4, skip_diffusion=False,
                   evaluate=False, evaluate_output_path=None, evaluate_target_names=None):
    device = pipe.device
    height, width = input_video[0].size[1], input_video[0].size[0]
    views = {
        "img": torch.stack([F.to_tensor(image)[None] for image in input_video], dim=1).to(device),
        "is_target": torch.zeros((1, len(input_video)), dtype=torch.bool, device=device),
    }
    if static_flag:
        views["is_static"] = torch.ones((1, len(input_video)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, len(input_video)), dtype=torch.int64, device=device)
    elif evaluate:
        views["is_static"] = torch.zeros((1, len(input_video)), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, 2 * len(input_video), 2, dtype=torch.int64, device=device).unsqueeze(0)
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

    if evaluate:
        num_targets = len(input_video) - 1
        eval_timestamps = torch.arange(1, 2 * len(input_video) - 1, 2, dtype=torch.int64, device=device)
        eval_c2w = input_cam2world[0:1].repeat(num_targets, 1, 1)
        eval_w2c = homo_matrix_inverse(eval_c2w)
        eval_K = K[:num_targets]

        eval_rgb, _, _ = pipe.reconstructor.gs_renderer.rasterizer.forward(
            gaussians,
            render_viewmats=[eval_w2c],
            render_Ks=[eval_K],
            render_timestamps=[eval_timestamps],
            sh_degree=0, width=width, height=height,
        )

        os.makedirs(evaluate_output_path, exist_ok=True)
        for i, name in enumerate(evaluate_target_names):
            frame = (eval_rgb[0, i].clamp(0, 1) * 255).byte().cpu().numpy()
            Image.fromarray(frame).save(os.path.join(evaluate_output_path, name))
        print(f"Saved {len(evaluate_target_names)} evaluation frames to {evaluate_output_path}")

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

    # Trajectory specification (mutually exclusive)
    traj_group = parser.add_mutually_exclusive_group(required=True)
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
                        help="Directory to save evaluation frames (default: outputs/evaluate)")

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

    # Build trajectory
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

    # Load video
    print(f"Loading video from {args.input_path}...")
    evaluate_target_names = None
    if args.evaluate:
        if not os.path.isdir(args.input_path):
            print("Error: --evaluate requires --input_path to be a directory of images")
            return 1
        all_names = sorted(os.listdir(args.input_path))
        all_names = [n for n in all_names if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        selected = all_names[:2 * args.num_frames]
        if len(selected) < 2 * args.num_frames:
            print(f"Warning: found {len(selected)} images, need {2 * args.num_frames} for evaluation")
        context_paths = [os.path.join(args.input_path, selected[i]) for i in range(0, len(selected), 2)]
        evaluate_target_names = [selected[i] for i in range(1, len(selected), 2)]
        resolution = (args.width, args.height)
        if args.resize_mode == "resize":
            images = [Image.open(p).convert("RGB").resize(resolution, resample=Image.LANCZOS) for p in context_paths]
        else:
            images = [center_crop(Image.open(p).convert("RGB"), resolution) for p in context_paths]
    else:
        images = load_video(args.input_path, args.num_frames,
                            resolution=(args.width, args.height),
                            resize_mode=args.resize_mode,
                            static_scene=args.static_scene)

    # Run inference
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.vis_rendering:
        # Save rendering visualizations to a folder named after the output (without extension)
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
        evaluate=args.evaluate,
        evaluate_output_path=args.evaluate_output_path,
        evaluate_target_names=evaluate_target_names,
    )
    print(f"Done! Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
